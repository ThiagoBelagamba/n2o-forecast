# -*- coding: utf-8 -*-
# forecasting.py — LightGBM Tweedie multi-passo (direto) + baselines
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from config import MODEL_CONFIG


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'MedAE': float(median_absolute_error(y_true, y_pred)),
    }


def _lgbm_tweedie(power, seed=None):
    return LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=power,
        n_estimators=MODEL_CONFIG['lgbm_n_estimators'],
        learning_rate=MODEL_CONFIG['lgbm_learning_rate'],
        num_leaves=MODEL_CONFIG['lgbm_num_leaves'],
        random_state=seed or MODEL_CONFIG['random_state'],
        verbosity=-1,
        n_jobs=MODEL_CONFIG.get('rf_n_jobs', 3),
    )


def build_mlforecast(models, lags=(1, 2)):
    """MLForecast anual; lags gerenciados pelo wrapper (sem vazamento no predict)."""
    return MLForecast(
        models=models,
        freq='YE',
        lags=list(lags),
        date_features=['year'],
        num_threads=MODEL_CONFIG.get('rf_n_jobs', 3),
    )


def _forecast_frame(df):
    """Apenas colunas exigidas pelo MLForecast (evita 'ano' como feature dinamica)."""
    return df[['unique_id', 'ds', 'y']].copy()


def select_tweedie_power(train_df, max_series=120):
    """
    Escolhe tweedie_variance_power no treino via CV temporal simples
    (ultimos anos do treino como validacao de 1 passo).
    Amostra series para caber no tempo de wall-clock.
    """
    powers = MODEL_CONFIG['tweedie_powers']
    uids = train_df['unique_id'].drop_duplicates()
    if len(uids) > max_series:
        uids = uids.sample(max_series, random_state=MODEL_CONFIG['random_state'])
        train_df = train_df[train_df['unique_id'].isin(uids)]
        print(f"CV Tweedie em amostra de {max_series} series.")

    years = sorted(train_df['ds'].dt.year.unique())
    n_val = min(MODEL_CONFIG['cv_splits'], max(1, len(years) // 5))
    val_years = years[-n_val:]
    best_p, best_mae = powers[0], float('inf')

    for p in powers:
        maes = []
        for vy in val_years:
            tr = _forecast_frame(train_df[train_df['ds'].dt.year < vy])
            va = train_df[train_df['ds'].dt.year == vy]
            if tr.empty or va.empty:
                continue
            fcst = build_mlforecast({'LGBM': _lgbm_tweedie(p)})
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fcst.fit(tr, static_features=[])
                pred = fcst.predict(h=1)
            merged = va[['unique_id', 'ds', 'y']].merge(
                pred, on=['unique_id', 'ds'], how='inner'
            )
            if merged.empty:
                continue
            maes.append(mean_absolute_error(merged['y'], merged['LGBM']))
        if maes:
            m = float(np.mean(maes))
            print(f"  Tweedie p={p}: MAE CV 1-passo = {m:.4f}")
            if m < best_mae:
                best_mae, best_p = m, p
    print(f"Selecionado tweedie_variance_power={best_p} (MAE CV={best_mae:.4f})")
    return best_p, best_mae


def fit_direct_models(train_df, power, horizon):
    """
    Previsao direta: max_horizon=H treina um modelo por h=1..H
    a partir so do treino (sem y observado do teste).
    fitted=True para MinT (residuos in-sample).
    """
    models = {
        'LGBM_Tweedie': _lgbm_tweedie(power),
        'Ridge': Ridge(alpha=1.0),
        'RF': RandomForestRegressor(
            n_estimators=MODEL_CONFIG['baseline_rf_estimators'],
            max_depth=MODEL_CONFIG['baseline_rf_max_depth'],
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=MODEL_CONFIG.get('rf_n_jobs', 3),
        ),
    }
    fcst = build_mlforecast(models)
    print(f"Treinando modelos diretos (max_horizon={horizon})...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fcst.fit(
            _forecast_frame(train_df),
            static_features=[],
            max_horizon=horizon,
            fitted=True,
        )
    return fcst


def fitted_insample(fcst):
    """Valores ajustados in-sample (necessarios para MinT shrink)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fcst.forecast_fitted_values()


def fit_recursive_for_future(panel_to_cutoff, power, h):
    """Um modelo 1-passo para projecao recursiva (2025–2026)."""
    models = {'LGBM_Tweedie': _lgbm_tweedie(power)}
    fcst = build_mlforecast(models)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fcst.fit(_forecast_frame(panel_to_cutoff), static_features=[])
        pred = fcst.predict(h=h)
    return fcst, pred


def predict_horizon(fcst, h):
    """Prediz h passos a frente a partir do fim do treino (sem new_df)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fcst.predict(h=h)


def persistence_multi_year(train_df, test_df):
    """Repete o ultimo y do treino por unique_id em todos os anos de teste."""
    last = (
        train_df.sort_values('ds')
        .groupby('unique_id', as_index=False)
        .tail(1)[['unique_id', 'y']]
        .rename(columns={'y': 'persistencia'})
    )
    out = test_df[['unique_id', 'ds', 'y']].merge(last, on='unique_id', how='left')
    out['persistencia'] = out['persistencia'].fillna(0.0)
    return out


def evaluate_on_test(pred_df, test_df, model_cols):
    """Alinha previsoes com y real do teste e calcula metricas por modelo."""
    merged = test_df[['unique_id', 'ds', 'y']].merge(
        pred_df, on=['unique_id', 'ds'], how='inner'
    )
    results = {}
    for col in model_cols:
        if col not in merged.columns:
            continue
        results[col] = compute_metrics(merged['y'], merged[col].clip(lower=0.0))
    return results, merged
