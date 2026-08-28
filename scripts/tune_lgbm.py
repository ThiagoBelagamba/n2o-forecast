# -*- coding: utf-8 -*-
# scripts/tune_lgbm.py — RandomizedSearch LGBM Tweedie (CV direta h=5)
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import LGBM_GRID, MODEL_CONFIG, PATHS, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from forecasting import (
    default_lgbm_params,
    eval_lgbm_direct_mae,
    sample_series_for_tune,
    save_lgbm_params,
)


def _random_combos(n_draw, seed):
    rng = np.random.default_rng(seed)
    keys = list(LGBM_GRID.keys())
    all_combos = list(itertools.product(*[LGBM_GRID[k] for k in keys]))
    idx = rng.choice(len(all_combos), size=min(n_draw, len(all_combos)), replace=False)
    for i in idx:
        yield dict(zip(keys, all_combos[i]))


def _cv_split(panel, split_year, horizon):
    """Treino <= split_year - horizon; validacao = proximos horizon anos."""
    cutoff = split_year - horizon
    train = panel[panel['ds'].dt.year <= cutoff].copy()
    val = panel[
        (panel['ds'].dt.year > cutoff) & (panel['ds'].dt.year <= split_year)
    ].copy()
    return train, val, cutoff


def run_tune(refine=False):
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']
    n_random = MODEL_CONFIG.get('lgbm_tune_n_random', 40)
    n_sample = MODEL_CONFIG.get('lgbm_tune_sample_series', 150)

    print("Carregando painel folha...")
    df = load_and_analyze_data(PATHS['data'])
    panel, _ = to_nixtla_panel(df)
    train_full, val_full, cutoff = _cv_split(panel, corte, horizon)
    print(
        f"CV direta h={horizon}: treino <= {cutoff}, "
        f"validacao {cutoff + 1}–{corte}"
    )

    if refine and PATHS.get('lgbm_cv') and Path(PATHS['lgbm_cv']).is_file():
        prev = pd.read_csv(PATHS['lgbm_cv']).sort_values('cv_mae')
        top = prev.head(MODEL_CONFIG.get('lgbm_tune_refine_top', 5))
        combos = []
        for _, row in top.iterrows():
            p = default_lgbm_params()
            for k in LGBM_GRID:
                if k in row:
                    p[k] = row[k]
            combos.append(p)
        eval_panel = panel
        label = "refino top-5 painel completo"
    else:
        combos = list(_random_combos(n_random, MODEL_CONFIG['random_state']))
        eval_panel = sample_series_for_tune(panel, n_sample)
        label = f"RandomizedSearch n={len(combos)} amostra {n_sample} series"

    print(f"\nEtapa: {label}")
    rows = []
    best_mae, best_params = float('inf'), default_lgbm_params()

    tr = eval_panel[eval_panel['ds'].dt.year <= cutoff]
    va = eval_panel[
        (eval_panel['ds'].dt.year > cutoff) & (eval_panel['ds'].dt.year <= corte)
    ]

    for i, combo in enumerate(combos, 1):
        params = default_lgbm_params()
        params.update(combo)
        mae = eval_lgbm_direct_mae(tr, va, params, horizon)
        row = {**combo, 'cv_mae': mae, 'stage': label}
        rows.append(row)
        print(
            f"  [{i}/{len(combos)}] p={params['tweedie_variance_power']:.1f} "
            f"depth={params['max_depth']} leaf={params['min_data_in_leaf']} "
            f"L1={params['reg_alpha']} L2={params['reg_lambda']} -> MAE={mae:.4f}"
        )
        if mae < best_mae:
            best_mae, best_params = mae, params

    out_df = pd.DataFrame(rows).sort_values('cv_mae')
    out_df.to_csv(PATHS['lgbm_cv'], index=False)
    print(f"\nGrid CV: {PATHS['lgbm_cv']}")
    print(f"Melhor CV MAE={best_mae:.4f} params={json.dumps(best_params, indent=2)}")

    if not refine:
        # Refino automatico top-5 no painel completo
        print("\n--- Refino top-5 no painel completo ---")
        top5 = out_df.head(MODEL_CONFIG.get('lgbm_tune_refine_top', 5))
        refine_rows = []
        tr_f = panel[panel['ds'].dt.year <= cutoff]
        va_f = panel[
            (panel['ds'].dt.year > cutoff) & (panel['ds'].dt.year <= corte)
        ]
        for _, row in top5.iterrows():
            params = default_lgbm_params()
            for k in LGBM_GRID:
                params[k] = row[k]
            mae = eval_lgbm_direct_mae(tr_f, va_f, params, horizon)
            refine_rows.append({**{k: params[k] for k in LGBM_GRID}, 'cv_mae': mae, 'stage': 'refino_full'})
            print(f"  refino MAE={mae:.4f} p={params['tweedie_variance_power']}")
            if mae < best_mae:
                best_mae, best_params = mae, params
        refine_df = pd.DataFrame(refine_rows).sort_values('cv_mae')
        combined = pd.concat([out_df, refine_df], ignore_index=True).sort_values('cv_mae')
        combined.to_csv(PATHS['lgbm_cv'], index=False)

    save_lgbm_params(best_params, best_mae, extra={'cv_protocol': f'direct_h{horizon}_val_{cutoff+1}_{corte}'})
    return best_params, best_mae


if __name__ == '__main__':
    refine_only = '--refine' in sys.argv
    run_tune(refine=refine_only)
