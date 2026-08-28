# -*- coding: utf-8 -*-
# forecasting.py — LightGBM Tweedie multi-passo (direto) + baselines
import json
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from config import LGBM_DEFAULTS, MODEL_CONFIG, PATHS


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'MedAE': float(median_absolute_error(y_true, y_pred)),
    }


def default_lgbm_params():
    return dict(LGBM_DEFAULTS)


def load_lgbm_params(path=None):
    """Carrega hiperparametros tunados ou retorna defaults."""
    path = path or PATHS.get('lgbm_params')
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('params', data)
        out = default_lgbm_params()
        out.update(params)
        return out, True
    return default_lgbm_params(), False


def save_lgbm_params(params, cv_mae, path=None, extra=None):
    path = path or PATHS['lgbm_params']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'params': params,
        'cv_mae_direct_h5': cv_mae,
        **(extra or {}),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"LGBM params salvos: {path}")


def _lgbm_tweedie(params=None, seed=None):
    p = default_lgbm_params()
    if params:
        p.update(params)
    return LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=p['tweedie_variance_power'],
        n_estimators=p.get('n_estimators', MODEL_CONFIG['lgbm_n_estimators']),
        learning_rate=p.get('learning_rate', MODEL_CONFIG['lgbm_learning_rate']),
        num_leaves=p.get('num_leaves', MODEL_CONFIG['lgbm_num_leaves']),
        max_depth=p.get('max_depth', -1),
        min_data_in_leaf=p.get('min_data_in_leaf', 20),
        reg_alpha=p.get('reg_alpha', 0.0),
        reg_lambda=p.get('reg_lambda', 0.0),
        random_state=seed or MODEL_CONFIG['random_state'],
        verbosity=-1,
        n_jobs=MODEL_CONFIG.get('rf_n_jobs', 3),
    )


def _lag_transforms():
    windows = MODEL_CONFIG.get('rolling_windows', [3, 5])
    return {
        1: [RollingMean(window_size=w) for w in windows],
    }


def build_mlforecast(models, lags=None):
    """MLForecast anual com lags + medias moveis."""
    if lags is None:
        lags = MODEL_CONFIG.get('mlforecast_lags', [1, 2, 3, 5])
    return MLForecast(
        models=models,
        freq='YE',
        lags=list(lags),
        lag_transforms=_lag_transforms(),
        date_features=['year'],
        num_threads=MODEL_CONFIG.get('rf_n_jobs', 3),
    )


def _forecast_frame(df):
    return df[['unique_id', 'ds', 'y']].copy()


def eval_lgbm_direct_mae(train_df, val_df, lgbm_params, horizon):
    """
    MAE folha em previsao direta h=horizon (protocolo multi-ano).
    train_df / val_df: painel folha com unique_id, ds, y.
    """
    if train_df.empty or val_df.empty:
        return float('inf')
    fcst = build_mlforecast({'LGBM': _lgbm_tweedie(lgbm_params)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fcst.fit(
            _forecast_frame(train_df),
            static_features=[],
            max_horizon=horizon,
        )
        pred = fcst.predict(h=horizon)
    merged = val_df[['unique_id', 'ds', 'y']].merge(
        pred, on=['unique_id', 'ds'], how='inner'
    )
    if merged.empty:
        return float('inf')
    return float(mean_absolute_error(merged['y'], merged['LGBM'].clip(lower=0.0)))


def sample_series_for_tune(panel, n_sample, seed=None):
    """Amostra series priorizando y>0 em algum ano."""
    rng = np.random.default_rng(seed or MODEL_CONFIG['random_state'])
    uids = panel['unique_id'].unique()
    positive = (
        panel.groupby('unique_id')['y']
        .apply(lambda s: (s > 0).any())
        .reset_index()
    )
    pos_uids = positive[positive['y']]['unique_id'].tolist()
    neg_uids = positive[~positive['y']]['unique_id'].tolist()
    n_pos = min(len(pos_uids), max(1, int(n_sample * 0.85)))
    n_neg = min(len(neg_uids), n_sample - n_pos)
    chosen = list(rng.choice(pos_uids, size=n_pos, replace=False))
    if n_neg > 0 and neg_uids:
        chosen += list(rng.choice(neg_uids, size=n_neg, replace=False))
    return panel[panel['unique_id'].isin(chosen)].copy()


def select_tweedie_power(train_df, max_series=120):
    """Legado: CV 1-passo rapida (main usa lgbm_best_params.json quando existir)."""
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
            params = default_lgbm_params()
            params['tweedie_variance_power'] = p
            fcst = build_mlforecast({'LGBM': _lgbm_tweedie(params)})
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


def fit_direct_models(train_df, lgbm_params, horizon):
    """Previsao direta h=horizon; lgbm_params dict completo."""
    models = {
        'LGBM_Tweedie': _lgbm_tweedie(lgbm_params),
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fcst.forecast_fitted_values()


def fit_recursive_for_future(panel_to_cutoff, lgbm_params, h):
    models = {'LGBM_Tweedie': _lgbm_tweedie(lgbm_params)}
    fcst = build_mlforecast(models)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fcst.fit(_forecast_frame(panel_to_cutoff), static_features=[])
        pred = fcst.predict(h=h)
    return fcst, pred


def predict_horizon(fcst, h):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fcst.predict(h=h)


def persistence_multi_year(train_df, test_df):
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
    merged = test_df[['unique_id', 'ds', 'y']].merge(
        pred_df, on=['unique_id', 'ds'], how='inner'
    )
    results = {}
    for col in model_cols:
        if col not in merged.columns:
            continue
        results[col] = compute_metrics(merged['y'], merged[col].clip(lower=0.0))
    return results, merged


def best_lgbm_reconciled_mae(metrics):
    """Menor MAE entre colunas LGBM reconciliadas."""
    cands = {
        k: v['MAE'] for k, v in metrics.items()
        if k.startswith('LGBM') and k != 'persistencia_multi_ano'
    }
    if not cands:
        return None, float('inf')
    best = min(cands, key=cands.get)
    return best, cands[best]


def blend_forecast_columns(df, components, out_col, clip_zero=True):
    """
    Media ponderada de colunas de previsao (mesmo unique_id, ds).
    components: {nome_coluna: peso} — pesos devem somar 1.
    """
    missing = [c for c in components if c not in df.columns]
    if missing:
        raise KeyError(f'Colunas ausentes para blend: {missing}')
    blend = np.zeros(len(df), dtype=float)
    for col, weight in components.items():
        blend += weight * df[col].astype(float).values
    if clip_zero:
        blend = np.clip(blend, 0.0, None)
    out = df.copy()
    out[out_col] = blend
    return out


def tune_blend_weight_leaf(
    pred_df,
    true_df,
    col_a,
    col_b,
    leaf_ids,
    weight_grid=None,
):
    """
    Sintoniza peso w de w*col_a + (1-w)*col_b por MAE folha.
    Retorna (melhor_w_ridge, melhor_mae, grid_rows).
    """
    weight_grid = weight_grid or MODEL_CONFIG['ensemble_ridge_lgbm']['weight_grid']
    merged = true_df[['unique_id', 'ds', 'y']].merge(
        pred_df[['unique_id', 'ds', col_a, col_b]],
        on=['unique_id', 'ds'],
        how='inner',
    )
    merged = merged[merged['unique_id'].isin(leaf_ids)]
    if merged.empty:
        return 0.5, float('inf'), []

    y = merged['y'].astype(float).values
    a = merged[col_a].astype(float).clip(lower=0.0).values
    b = merged[col_b].astype(float).clip(lower=0.0).values
    rows = []
    best_w, best_mae = 0.5, float('inf')
    for w in weight_grid:
        pred = w * a + (1.0 - w) * b
        mae = float(mean_absolute_error(y, pred))
        rows.append({'ridge_weight': w, 'lgbm_weight': 1.0 - w, 'mae_leaf': mae})
        if mae < best_mae:
            best_mae, best_w = mae, w
    return best_w, best_mae, rows


def load_ensemble_weights(path=None):
    """Carrega pesos tunados Ridge+LGBM ou defaults."""
    cfg = MODEL_CONFIG['ensemble_ridge_lgbm']
    default = {
        'ridge_weight': cfg['default_ridge_weight'],
        'lgbm_weight': 1.0 - cfg['default_ridge_weight'],
        'out_col': cfg['out_col'],
    }
    path = path or PATHS.get('ensemble_weights')
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        ridge_w = data.get('ridge_weight', default['ridge_weight'])
        return {
            'ridge_weight': ridge_w,
            'lgbm_weight': data.get('lgbm_weight', 1.0 - ridge_w),
            'out_col': cfg['out_col'],
            'cv_mae_leaf': data.get('cv_mae_leaf'),
        }, True
    return default, False


def save_ensemble_weights(ridge_weight, cv_mae, path=None, extra=None):
    path = path or PATHS['ensemble_weights']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'ridge_weight': ridge_weight,
        'lgbm_weight': 1.0 - ridge_weight,
        'cv_mae_leaf': cv_mae,
        **(extra or {}),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Ensemble weights salvos: {path}')


def add_ridge_lgbm_ensemble(Y_hat, Y_fitted=None, ridge_weight=None):
    """Adiciona coluna Ensemble_Ridge_LGBM em previsoes e fitted."""
    cfg = MODEL_CONFIG['ensemble_ridge_lgbm']
    weights, _ = load_ensemble_weights()
    w = ridge_weight if ridge_weight is not None else weights['ridge_weight']
    components = {cfg['ridge_col']: w, cfg['lgbm_col']: 1.0 - w}
    out_col = cfg['out_col']
    Y_hat = blend_forecast_columns(Y_hat, components, out_col)
    if Y_fitted is not None:
        Y_fitted = blend_forecast_columns(Y_fitted, components, out_col)
    return Y_hat, Y_fitted, w
