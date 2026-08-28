# -*- coding: utf-8 -*-
# scripts/run_conformal.py — Split Conformal sobre Ridge/MinTrace
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import CONFORMAL_CONFIG, MODEL_CONFIG, PATHS, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from forecasting_conformal import (
    aggregate_nivel1_intervals,
    apply_intervals,
    build_quantile_table,
    calibration_residuals,
    evaluate_coverage,
    fit_ridge_direct,
    fitted_ridge,
    national_calibration_residuals,
    predict_ridge,
)
from reconciliation import build_hierarchy, national_total, reconcile_forecasts
from results_multiyear import plot_conformal_national


def main():
    ensure_dirs()
    cfg = CONFORMAL_CONFIG
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']
    cal_train = cfg['calibration_train_end']
    cal_end = cfg['calibration_end_year']
    point_base = cfg['point_col_base']
    point_rec = cfg['model_col']

    print('1. Painel + hierarquia...')
    df = load_and_analyze_data(PATHS['data'])
    panel, _ = to_nixtla_panel(df)
    Y_df, S_df, tags, _ = build_hierarchy(panel)
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])
    top_ids = set(tags[list(tags.keys())[0]])

    print(f'\n2. Calibracao conformal (treino <= {cal_train}, val {cal_train + 1}–{cal_end})...')
    cal_leaf, _ = calibration_residuals(
        Y_df, tags, cal_train, cal_end, horizon, point_col=point_base
    )
    cal_nat = national_calibration_residuals(Y_df, tags, cal_leaf, point_col=point_base)
    quantiles = build_quantile_table(cal_leaf, cal_nat, cfg['alphas'])
    print('   Quantis (folha pooled, alpha=0.10):', quantiles[0.10]['pooled_leaf'])

    print(f'\n3. Teste cego {corte + 1}–{corte + horizon} (Ridge + MinT)...')
    Y_train = Y_df[Y_df['ds'].dt.year <= corte].copy()
    Y_test = Y_df[Y_df['ds'].dt.year > corte].copy()
    test_leaf = Y_test[Y_test['unique_id'].isin(leaf_ids)].copy()

    fcst = fit_ridge_direct(Y_train, horizon)
    Y_hat = predict_ridge(fcst, h=horizon)
    Y_fitted = fitted_ridge(fcst)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y_rec, _ = reconcile_forecasts(
            Y_hat, Y_df, S_df, tags,
            Y_fitted=Y_fitted,
            train_end_year=corte,
        )

    pred_leaf = Y_rec[Y_rec['unique_id'].isin(leaf_ids)].copy()
    test_merged = test_leaf[['unique_id', 'ds', 'y']].merge(
        pred_leaf[['unique_id', 'ds', point_rec]],
        on=['unique_id', 'ds'],
        how='inner',
    )
    intervals_leaf = apply_intervals(
        test_merged.rename(columns={point_rec: point_rec}),
        quantiles,
        cutoff_year=corte,
        point_col=point_rec,
        level='leaf',
    )
    intervals_leaf['ano'] = intervals_leaf['ds'].dt.year

    # Folhas com y>0 em algum ano do treino
    active = (
        Y_train[Y_train['unique_id'].isin(leaf_ids)]
        .groupby('unique_id')['y']
        .max()
        .reset_index()
    )
    active_ids = set(active.loc[active['y'] > 0, 'unique_id'])
    intervals_active = intervals_leaf[intervals_leaf['unique_id'].isin(active_ids)]

    nat_point_node = national_total(Y_rec, tags, point_rec)
    nat_obs = (
        Y_test[Y_test['unique_id'].isin(top_ids)]
        .groupby('ds', as_index=False)['y']
        .sum()
    )
    nat_eval = nat_obs.merge(
        nat_point_node.groupby('ds', as_index=False)[point_rec].sum(),
        on='ds',
        how='inner',
    )
    nat_eval['unique_id'] = 'Brasil'
    intervals_nat = apply_intervals(
        nat_eval,
        quantiles,
        cutoff_year=corte,
        point_col=point_rec,
        level='national',
    )
    intervals_nat['ano'] = intervals_nat['ds'].dt.year

    intervals_n1 = aggregate_nivel1_intervals(intervals_leaf, panel)

    print('\n4. Cobertura empirica (teste)...')
    coverage = {
        'protocol': f'cal_{cal_train + 1}_{cal_end}_test_{corte + 1}_{corte + horizon}',
        'quantiles': {
            str(k): {
                qk: (qv if not isinstance(qv, dict) else {str(h): v for h, v in qv.items()})
                for qk, qv in v.items()
            }
            for k, v in quantiles.items()
        },
        'leaf_all': evaluate_coverage(intervals_leaf, point_col=point_rec),
        'leaf_active_y_gt0': evaluate_coverage(intervals_active, point_col=point_rec),
        'national': evaluate_coverage(intervals_nat, point_col=point_rec),
    }
    for n1 in intervals_n1['nivel_1'].unique():
        sub = intervals_n1[intervals_n1['nivel_1'] == n1]
        if 'y' in sub.columns:
            coverage[f'nivel_1_{n1}'] = evaluate_coverage(sub, point_col=point_rec)

    for level, block in coverage.items():
        if not isinstance(block, dict) or 'coverage_90' not in block:
            continue
        print(
            f"   {level}: cov90={block.get('coverage_90', 0):.3f} "
            f"cov95={block.get('coverage_95', 0):.3f} "
            f"width90={block.get('width_mean_90', 0):.1f}"
        )

    intervals_leaf.to_csv(PATHS['conformal_intervals'], index=False)
    with open(PATHS['conformal_coverage'], 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f'\nSalvo: {PATHS["conformal_intervals"]}')
    print(f'Salvo: {PATHS["conformal_coverage"]}')

    hist_nat = Y_df[Y_df['unique_id'].isin(top_ids)].copy()
    hist_nat['ano'] = hist_nat['ds'].dt.year
    hist_nat = hist_nat.groupby('ano', as_index=False)['y'].sum()

    plot_conformal_national(hist_nat, intervals_nat, corte)

    print('\n5. Projecao 2025–2026 com intervalos...')
    Y_full = Y_df[Y_df['ds'].dt.year <= 2024].copy()
    fcst_fut = fit_ridge_direct(Y_full, horizon=2)
    Y_hat_fut = predict_ridge(fcst_fut, h=2)
    Y_fit_fut = fitted_ridge(fcst_fut)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y_rec_fut, _ = reconcile_forecasts(
            Y_hat_fut, Y_df, S_df, tags,
            Y_fitted=Y_fit_fut,
            train_end_year=2024,
        )
    pred_fut_leaf = Y_rec_fut[Y_rec_fut['unique_id'].isin(leaf_ids)][
        ['unique_id', 'ds', point_rec]
    ].copy()
    fut_intervals = apply_intervals(
        pred_fut_leaf,
        quantiles,
        cutoff_year=2024,
        point_col=point_rec,
        level='leaf',
    )
    fut_intervals['ano'] = fut_intervals['ds'].dt.year
    fut_intervals.to_csv(PATHS['conformal_projection'], index=False)
    print(f'Salvo: {PATHS["conformal_projection"]}')

    return coverage


if __name__ == '__main__':
    main()
