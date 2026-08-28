# -*- coding: utf-8 -*-
# scripts/run_ensemble_benchmark.py — Ensemble Ridge+NHITS (antes MinT) vs barreira
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import MODEL_CONFIG, PATHS, RIDGE_MAE_BARRIER, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from forecasting import (
    blend_forecast_columns,
    evaluate_on_test,
    fit_direct_models,
    load_lgbm_params,
    predict_horizon,
    tune_blend_weight_leaf,
)
from forecasting_nhits import expand_leaf_to_hierarchy, fit_predict_nhits
from reconciliation import build_hierarchy, reconcile_forecasts


def _cv_split(panel, split_year, horizon):
    cutoff = split_year - horizon
    train = panel[panel['ds'].dt.year <= cutoff].copy()
    val = panel[
        (panel['ds'].dt.year > cutoff) & (panel['ds'].dt.year <= split_year)
    ].copy()
    return train, val, cutoff


def _ridge_and_nhits_hierarchy(train_df, val_df, S_df, leaf_ids, horizon, nhits_steps):
    """Ridge em toda hierarquia + NHITS folha expandida."""
    lgbm_params, _ = load_lgbm_params()
    fcst = fit_direct_models(train_df, lgbm_params, horizon)
    ridge_hat = predict_horizon(fcst, h=horizon)[['unique_id', 'ds', 'Ridge']]

    train_leaf = train_df[train_df['unique_id'].isin(leaf_ids)].copy()
    _, nhits_leaf = fit_predict_nhits(
        train_leaf, horizon, max_steps=nhits_steps, input_size=12
    )
    nhits_hat = expand_leaf_to_hierarchy(nhits_leaf, S_df, model_col='NHITS')
    nhits_hat = nhits_hat.rename(columns={'NHITS': 'NHITS'})

    merged = ridge_hat.merge(nhits_hat, on=['unique_id', 'ds'], how='inner')
    return merged, val_df


def main():
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']
    cfg = MODEL_CONFIG['ensemble_ridge_nhits']
    nhits_steps = 120

    print('1. Painel + hierarquia...')
    df = load_and_analyze_data(PATHS['data'])
    panel, _ = to_nixtla_panel(df)
    Y_df, S_df, tags, _ = build_hierarchy(panel)
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])

    print('\n2. Sintonia peso Ridge+NHITS (CV folha)...')
    tr_cv, va_cv, cutoff = _cv_split(Y_df, corte, horizon)
    print(f'   Treino <= {cutoff}, val {cutoff + 1}–{corte}')
    pred_cv, val_cv = _ridge_and_nhits_hierarchy(
        tr_cv, va_cv, S_df, leaf_ids, horizon, nhits_steps
    )
    best_w, cv_mae, grid = tune_blend_weight_leaf(
        pred_cv.rename(columns={'Ridge': cfg['ridge_col'], 'NHITS': cfg['nhits_col']}),
        val_cv,
        cfg['ridge_col'],
        cfg['nhits_col'],
        leaf_ids,
        weight_grid=cfg['weight_grid'],
    )
    print(f'   Melhor w_ridge={best_w:.1f} MAE folha CV={cv_mae:.4f}')

    print('\n3. Teste cego 2020–2024 (treino <= 2019)...')
    Y_train = Y_df[Y_df['ds'].dt.year <= corte].copy()
    Y_test = Y_df[Y_df['ds'].dt.year > corte].copy()
    test_leaf = Y_test[Y_test['unique_id'].isin(leaf_ids)].copy()

    pred_test, _ = _ridge_and_nhits_hierarchy(
        Y_train, Y_test, S_df, leaf_ids, horizon, nhits_steps
    )
    components = {cfg['ridge_col']: best_w, cfg['nhits_col']: 1.0 - best_w}
    Y_hat = blend_forecast_columns(pred_test, components, cfg['out_col'])

    print('\n4. Reconciliacao MinT...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y_rec, _ = reconcile_forecasts(
            Y_hat, Y_df, S_df, tags, train_end_year=corte
        )

    rec_cols = [
        c for c in Y_rec.columns
        if c not in ('unique_id', 'ds') and cfg['out_col'] in c
    ]
    pred_leaf = Y_rec[Y_rec['unique_id'].isin(leaf_ids)].copy()
    mdict, _ = evaluate_on_test(pred_leaf, test_leaf, rec_cols)

    print('\n=== Ensemble Ridge+NHITS reconciliado ===')
    best_col, best_mae = None, float('inf')
    for name, m in sorted(mdict.items(), key=lambda x: x[1]['MAE']):
        print(f"  {name}: MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f}")
        if m['MAE'] < best_mae:
            best_mae, best_col = m['MAE'], name

    ridge_ref = RIDGE_MAE_BARRIER
    print(f'\nBarreira Ridge/MinTrace(ols): {ridge_ref:.2f} t')
    print(f'Melhor ensemble: {best_col} MAE={best_mae:.2f} t')
    beat = best_mae < ridge_ref
    print(
        'Ensemble superou barreira Ridge.'
        if beat
        else f'Ensemble nao superou barreira (delta={ridge_ref - best_mae:+.2f} t).'
    )

    out = {
        'blend': 'Ridge+NHITS',
        'ridge_weight': best_w,
        'nhits_weight': 1.0 - best_w,
        'cv_mae_leaf': cv_mae,
        'weight_grid': grid,
        'best_reconciled': best_col,
        'best_mae': best_mae,
        'ridge_barrier': ridge_ref,
        'beat_barrier': beat,
        'metrics': mdict,
    }
    out_path = Path(PATHS['ensemble_benchmark'])
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(
            {k: v for k, v in out.items() if k != 'metrics'},
            f,
            indent=2,
        )
    pd.DataFrame(
        [{'metodo': k, **v} for k, v in mdict.items()]
    ).to_csv(Path(PATHS['results_dir']) / 'metricas_ensemble.csv', index=False)
    print(f'\nSalvo: {out_path}')
    return out


if __name__ == '__main__':
    main()
