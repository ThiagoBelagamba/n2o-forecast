# -*- coding: utf-8 -*-
# scripts/tune_ensemble.py — sintonia peso Ridge+LGBM (CV direta h=5, MAE folha)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import MODEL_CONFIG, PATHS, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from forecasting import (
    fit_direct_models,
    load_lgbm_params,
    predict_horizon,
    save_ensemble_weights,
    tune_blend_weight_leaf,
)
from reconciliation import build_hierarchy


def _cv_split(panel, split_year, horizon):
    cutoff = split_year - horizon
    train = panel[panel['ds'].dt.year <= cutoff].copy()
    val = panel[
        (panel['ds'].dt.year > cutoff) & (panel['ds'].dt.year <= split_year)
    ].copy()
    return train, val, cutoff


def main():
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']
    cfg = MODEL_CONFIG['ensemble_ridge_lgbm']

    print('Carregando painel + hierarquia...')
    df = load_and_analyze_data(PATHS['data'])
    panel, _ = to_nixtla_panel(df)
    Y_df, _, tags, _ = build_hierarchy(panel)
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])

    train, val, cutoff = _cv_split(Y_df, corte, horizon)
    print(f'CV ensemble: treino <= {cutoff}, val {cutoff + 1}–{corte}, h={horizon}')

    lgbm_params, _ = load_lgbm_params()

    fcst = fit_direct_models(train, lgbm_params, horizon)
    pred = predict_horizon(fcst, h=horizon)

    best_w, best_mae, grid = tune_blend_weight_leaf(
        pred,
        val,
        cfg['ridge_col'],
        cfg['lgbm_col'],
        leaf_ids,
        weight_grid=cfg['weight_grid'],
    )

    print('\n=== Grid peso Ridge (folha) ===')
    for row in grid:
        print(
            f"  w_ridge={row['ridge_weight']:.1f} "
            f"w_lgbm={row['lgbm_weight']:.1f} MAE={row['mae_leaf']:.4f}"
        )
    print(f'\nMelhor: ridge_weight={best_w:.1f} MAE folha CV={best_mae:.4f}')

    save_ensemble_weights(
        best_w,
        best_mae,
        extra={
            'cv_protocol': f'direct_h{horizon}_val_{cutoff + 1}_{corte}',
            'weight_grid': grid,
        },
    )


if __name__ == '__main__':
    main()
