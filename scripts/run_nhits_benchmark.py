# -*- coding: utf-8 -*-
# scripts/run_nhits_benchmark.py — NHITS + MinT vs barreira Ridge (83.85 t)
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import MODEL_CONFIG, PATHS, RIDGE_MAE_BARRIER, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from forecasting import compute_metrics, evaluate_on_test
from forecasting_nhits import (
    expand_leaf_to_hierarchy,
    fit_predict_nhits,
    nhits_leaf_mae,
)
from reconciliation import build_hierarchy, reconcile_forecasts


def main():
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']

    print("1. Painel + hierarquia...")
    df = load_and_analyze_data(PATHS['data'])
    panel, _ = to_nixtla_panel(df)
    Y_df, S_df, tags, _ = build_hierarchy(panel)
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])

    Y_train = Y_df[Y_df['ds'].dt.year <= corte].copy()
    Y_test = Y_df[Y_df['ds'].dt.year > corte].copy()
    train_leaf = Y_train[Y_train['unique_id'].isin(leaf_ids)].copy()
    test_leaf = Y_test[Y_test['unique_id'].isin(leaf_ids)].copy()

    print(f"   Folhas treino={len(train_leaf)} teste={len(test_leaf)}")

    print("\n2. NHITS (CPU, folha, h=5)...")
    _nf, pred_leaf = fit_predict_nhits(
        train_leaf, horizon, max_steps=120, input_size=12
    )
    raw_mae, _ = nhits_leaf_mae(pred_leaf, test_leaf)
    print(f"   NHITS folha MAE (sem MinT): {raw_mae:.4f}")

    print("\n3. Expansao hierarquica + MinT...")
    Y_hat = expand_leaf_to_hierarchy(pred_leaf, S_df, model_col='NHITS')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y_rec, _ = reconcile_forecasts(
            Y_hat, Y_df, S_df, tags, train_end_year=corte
        )

    rec_cols = [
        c for c in Y_rec.columns
        if c not in ('unique_id', 'ds') and 'NHITS' in c
    ]
    pred_leaf_rec = Y_rec[Y_rec['unique_id'].isin(leaf_ids)].copy()
    mdict, merged = evaluate_on_test(pred_leaf_rec, test_leaf, rec_cols)

    print("\n=== NHITS reconciliado (teste cego 2020-2024) ===")
    best_col, best_mae = None, float('inf')
    for name, m in sorted(mdict.items(), key=lambda x: x[1]['MAE']):
        print(f"  {name}: MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f}")
        if m['MAE'] < best_mae:
            best_mae, best_col = m['MAE'], name

    ridge_ref = RIDGE_MAE_BARRIER
    print(f"\nBarreira Ridge/MinTrace(ols): {ridge_ref:.2f} t")
    print(f"Melhor NHITS: {best_col} MAE={best_mae:.2f} t")
    if best_mae < ridge_ref:
        print("NHITS superou a barreira Ridge.")
    else:
        print(f"NHITS nao superou barreira (delta={ridge_ref - best_mae:+.2f} t).")

    out = {
        'nhits_leaf_mae_raw': raw_mae,
        'nhits_best_reconciled': best_col,
        'nhits_best_mae': best_mae,
        'ridge_barrier': ridge_ref,
        'beat_barrier': best_mae < ridge_ref,
        'metrics': mdict,
    }
    out_path = Path(PATHS['results_dir']) / 'nhits_benchmark.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'nhits_leaf_mae_raw': raw_mae,
                'nhits_best_reconciled': best_col,
                'nhits_best_mae': best_mae,
                'ridge_barrier': ridge_ref,
                'beat_barrier': best_mae < ridge_ref,
            },
            f,
            indent=2,
        )
    print(f"\nSalvo: {out_path}")

    rows = [{'metodo': k, **v} for k, v in mdict.items()]
    pd.DataFrame(rows).to_csv(
        Path(PATHS['results_dir']) / 'metricas_nhits.csv', index=False
    )
    return out


if __name__ == '__main__':
    main()
