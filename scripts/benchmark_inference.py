# -*- coding: utf-8 -*-
# scripts/benchmark_inference.py — mede wall-clock da inferência multi-ano
# (previsão direta h=5 + MinT clássico + clip ≥ 0)
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from config import MODEL_CONFIG, PATHS, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from forecasting import fitted_insample, predict_horizon
from reconciliation import build_hierarchy, reconcile_forecasts


def main():
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']
    model_path = PATHS['model']
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Modelo nao encontrado: {model_path}. Rode main.py antes."
        )

    print("1. Carregando dados + hierarquia...")
    t0 = time.perf_counter()
    df = load_and_analyze_data(PATHS['data'])
    panel, _quality = to_nixtla_panel(df)
    Y_df, S_df, tags, _ = build_hierarchy(panel)
    Y_train = Y_df[Y_df['ds'].dt.year <= corte].copy()
    load_s = time.perf_counter() - t0
    print(f"   carga+hierarquia: {load_s:.2f}s")

    print("2. Carregando modelo joblib...")
    t0 = time.perf_counter()
    fcst = joblib.load(model_path)
    load_model_s = time.perf_counter() - t0
    print(f"   load modelo: {load_model_s:.2f}s")

    print(f"3. Inferencia direta h={horizon} (predict)...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        Y_hat = predict_horizon(fcst, h=horizon)
        predict_s = time.perf_counter() - t0
    print(f"   predict: {predict_s:.2f}s | linhas={len(Y_hat)}")

    print("4. Fitted in-sample (para MinT mint_shrink)...")
    t0 = time.perf_counter()
    Y_fitted = fitted_insample(fcst)
    fitted_s = time.perf_counter() - t0
    print(f"   fitted: {fitted_s:.2f}s")

    print("5. Reconciliacao MinT classico + clip >= 0...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        Y_rec, _ = reconcile_forecasts(
            Y_hat, Y_df, S_df, tags,
            Y_fitted=Y_fitted,
            train_end_year=corte,
        )
        reconcile_s = time.perf_counter() - t0
    print(f"   reconcile+clip: {reconcile_s:.2f}s | cols={len(Y_rec.columns)}")

    # Inferencia = predict + fitted + MinT+clip (sem carga de dados/modelo)
    infer_core_s = predict_s + fitted_s + reconcile_s
    # Bloco operacional completo apos modelo em memoria
    infer_total_s = load_model_s + infer_core_s

    timing = {
        'protocol': 'direct_h5 + MinTrace(ols/mint_shrink) + clip>=0',
        'n_series_hierarchy': int(Y_df['unique_id'].nunique()),
        'n_leaves': int(S_df.shape[1]),
        'horizon': horizon,
        'seconds': {
            'load_data_hierarchy': round(load_s, 3),
            'load_model': round(load_model_s, 3),
            'predict_horizon': round(predict_s, 3),
            'fitted_insample': round(fitted_s, 3),
            'reconcile_mint_clip': round(reconcile_s, 3),
            'inference_core': round(infer_core_s, 3),
            'inference_total_with_model_load': round(infer_total_s, 3),
        },
        'minutes': {
            'inference_core': round(infer_core_s / 60.0, 3),
            'inference_total_with_model_load': round(infer_total_s / 60.0, 3),
        },
        'margin_note': (
            'Inferencia multi-ano (predict + MinT classico + clip) '
            f'em {infer_core_s:.1f}s (~{infer_core_s/60:.2f} min); '
            f'com load do joblib: {infer_total_s:.1f}s (~{infer_total_s/60:.2f} min).'
        ),
    }

    out = PATHS.get('timing') or os.path.join(
        PATHS['results_dir'], 'tempo_inferencia.json'
    )
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    print(f"\n=== RESULTADO ===\n{timing['margin_note']}")
    print(f"Salvo: {out}")
    return timing


if __name__ == '__main__':
    main()
