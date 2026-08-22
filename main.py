# -*- coding: utf-8 -*-
# main.py — pipeline multi-ano: Tweedie + previsao direta + MinT + 2025/2026
import traceback
import warnings

import numpy as np
import pandas as pd

from config import MODEL_CONFIG, PATHS, ensure_dirs
from data_loading import load_and_analyze_data, perform_eda
from data_preprocessing import split_panel_chronological, to_nixtla_panel
from forecasting import (
    evaluate_on_test,
    fit_direct_models,
    fit_recursive_for_future,
    fitted_insample,
    persistence_multi_year,
    predict_horizon,
    select_tweedie_power,
)
from reconciliation import (
    build_hierarchy,
    leaf_forecasts_only,
    national_total,
    reconcile_forecasts,
)
from results_multiyear import (
    plot_projection,
    plot_test_totals,
    save_comparacao_real_previsto,
    save_leitura,
    save_metrics_table,
    save_model,
    save_projection_csv,
    save_report,
)


def _print_metrics(title, metrics):
    print(
        f"\n{title}\n"
        f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | "
        f"R2: {metrics['R2']:.4f} | MedAE: {metrics['MedAE']:.4f}"
    )


def _total_by_year(df, value_col='y'):
    tmp = df.copy()
    tmp['ano'] = tmp['ds'].dt.year
    return tmp.groupby('ano', as_index=False)[value_col].sum().rename(
        columns={value_col: 'y'}
    )


def main():
    print("Iniciando pipeline multi-ano (Tweedie + MinT)...")
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']

    try:
        print("\n1. Dados e EDA...")
        df = load_and_analyze_data(PATHS['data'])
        df = perform_eda(df)

        print("\n2. Painel Nixtla + hierarquia...")
        panel, quality = to_nixtla_panel(df)
        Y_df, S_df, tags, _spec = build_hierarchy(panel)
        Y_train = Y_df[Y_df['ds'].dt.year <= corte].copy()
        Y_test = Y_df[Y_df['ds'].dt.year > corte].copy()
        print(
            f"Y_df treino={len(Y_train)} teste={len(Y_test)} "
            f"series={Y_df['unique_id'].nunique()}"
        )

        print("\n3. Selecao tweedie_variance_power (CV no treino)...")
        # CV na folha original (mais leve) — amostra de series se necessario
        train_panel, _ = split_panel_chronological(panel)
        power, cv_mae = select_tweedie_power(train_panel)

        print("\n4. Treino direto h=%d (sem lag observado no teste)..." % horizon)
        fcst = fit_direct_models(Y_train, power, horizon)
        Y_hat = predict_horizon(fcst, h=horizon)
        Y_fitted = fitted_insample(fcst)
        print(f"Previsoes base: {len(Y_hat)} linhas, cols={list(Y_hat.columns)}")
        print(f"Fitted in-sample: {len(Y_fitted)} linhas")

        print("\n5. Reconciliacao MinT (nonnegative) + BottomUp...")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Y_rec, base_cols = reconcile_forecasts(
                Y_hat, Y_df, S_df, tags,
                Y_fitted=Y_fitted,
                train_end_year=corte,
            )
        print(f"Cols reconciliadas: {[c for c in Y_rec.columns if c not in ('unique_id','ds')][:8]}...")

        # Colunas de interesse apos reconcile: Model/BottomUp, Model/MinTrace_method-mint_shrink
        rec_cols = [
            c for c in Y_rec.columns
            if c not in ('unique_id', 'ds') and (
                'MinTrace' in c or 'BottomUp' in c or c in base_cols
            )
        ]
        # Preferir MinT do LGBM; fallback BottomUp / base
        preferred = [c for c in rec_cols if 'LGBM' in c and 'MinTrace' in c]
        if not preferred:
            preferred = [c for c in rec_cols if 'LGBM' in c and 'BottomUp' in c]
        if not preferred:
            preferred = [c for c in rec_cols if 'LGBM' in c]
        best_col = preferred[0] if preferred else rec_cols[0]

        print("\n6. Metricas no teste (folha)...")
        leaf_key = list(tags.keys())[-1]
        leaf_ids = set(tags[leaf_key])
        test_leaf = Y_test[Y_test['unique_id'].isin(leaf_ids)].copy()
        pred_leaf = Y_rec[Y_rec['unique_id'].isin(leaf_ids)].copy()

        # Persistencia multi-ano nas folhas
        train_leaf = Y_train[Y_train['unique_id'].isin(leaf_ids)]
        pers = persistence_multi_year(train_leaf, test_leaf)
        pers_metrics = {
            'MAE': float(np.mean(np.abs(pers['y'] - pers['persistencia']))),
            'RMSE': float(np.sqrt(np.mean((pers['y'] - pers['persistencia']) ** 2))),
            'R2': float(1 - np.sum((pers['y'] - pers['persistencia']) ** 2) /
                        np.sum((pers['y'] - pers['y'].mean()) ** 2)),
            'MedAE': float(np.median(np.abs(pers['y'] - pers['persistencia']))),
        }

        metrics = {'persistencia_multi_ano': pers_metrics}
        _print_metrics('persistencia_multi_ano', pers_metrics)

        eval_cols = []
        for c in rec_cols:
            if c in pred_leaf.columns:
                eval_cols.append(c)
        # tambem bases sem reconcile se existirem
        for c in base_cols:
            if c in pred_leaf.columns and c not in eval_cols:
                eval_cols.append(c)

        mdict, merged = evaluate_on_test(pred_leaf, test_leaf, eval_cols)
        metrics.update(mdict)
        for name, m in mdict.items():
            _print_metrics(name, m)

        # melhor por MAE entre modelos LGBM reconciliados; senao menor MAE geral exceto persistencia
        candidates = {
            k: v for k, v in metrics.items()
            if k != 'persistencia_multi_ano'
        }
        best_model = min(candidates, key=lambda k: candidates[k]['MAE'])
        print(f"\nSelecionado (menor MAE): {best_model}")

        # Totais nacionais para grafico
        hist_total = _total_by_year(
            Y_df[Y_df['unique_id'] == list(tags[list(tags.keys())[0]])[0]]
        )
        pred_totals = {}
        for c in [best_model, 'persistencia_multi_ano']:
            if c == 'persistencia_multi_ano':
                pt = pers.copy()
                pt['ano'] = pt['ds'].dt.year
                pred_totals[c] = pt.groupby('ano', as_index=False)['persistencia'].sum().rename(
                    columns={'persistencia': 'yhat'}
                )
            elif c in Y_rec.columns:
                tot = national_total(Y_rec, tags, c)
                tot['ano'] = tot['ds'].dt.year
                pred_totals[c] = tot.rename(columns={c: 'yhat'})[['ano', 'yhat']]

        print("\n7. Salvando relatorio / metricas / modelo...")
        save_metrics_table(metrics)
        save_report(
            quality,
            metrics,
            tweedie_power=power,
            protocol_note=(
                'Previsao direta multi-ano sem emissao observada '
                'nos anos de teste (anti-vazamento).'
            ),
            best_model=best_model,
        )
        save_leitura(quality, metrics, best_model, power)
        save_model(fcst)
        plot_test_totals(hist_total, pred_totals, corte)

        # Predicoes folha teste + comparacao real vs previsto (anos ja no inventario)
        out_pred = merged.copy()
        out_pred['ano'] = out_pred['ds'].dt.year
        out_pred.to_csv(PATHS['predictions'], index=False)
        save_comparacao_real_previsto(merged, best_model, pers_df=pers)

        print("\n8. Projecao 2025–2026 (recursiva a partir de 2024)...")
        Y_full = Y_df[Y_df['ds'].dt.year <= 2024].copy()
        _fcst_fut, Y_hat_fut = fit_recursive_for_future(Y_full, power, h=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Y_rec_fut, _ = reconcile_forecasts(
                Y_hat_fut, Y_df, S_df, tags, train_end_year=2024
            )
        fut_cols = [c for c in Y_rec_fut.columns if 'LGBM' in c and 'MinTrace' in c]
        if not fut_cols:
            fut_cols = [c for c in Y_rec_fut.columns if 'LGBM' in c]
        fut_col = fut_cols[0]
        leaf_fut = leaf_forecasts_only(Y_rec_fut, tags, fut_col)
        leaf_fut['ano'] = leaf_fut['ds'].dt.year
        leaf_fut = leaf_fut.rename(columns={fut_col: 'previsto'})

        last2024 = (
            Y_full[Y_full['unique_id'].isin(leaf_ids) & (Y_full['ds'].dt.year == 2024)]
            [['unique_id', 'y']]
            .rename(columns={'y': 'persistencia_2024'})
        )
        leaf_fut = leaf_fut.merge(last2024, on='unique_id', how='left')
        leaf_fut['persistencia_2024'] = leaf_fut['persistencia_2024'].fillna(0.0)
        leaf_fut['previsto'] = leaf_fut['previsto'].clip(lower=0.0)

        save_projection_csv(leaf_fut[['unique_id', 'ano', 'previsto', 'persistencia_2024']])
        plot_projection(hist_total, leaf_fut)

        print(f"\nProcesso concluido. Relatorio: {PATHS['report']}")
        print(f"Projecao: {PATHS['projection']}")
        print(
            f"Split: treino <= {corte} | teste direto h={horizon} | "
            f"cenario {MODEL_CONFIG['forecast_years']}"
        )

    except Exception:
        print("\nErro durante a execucao:")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
