# -*- coding: utf-8 -*-
# forecasting_conformal.py — Split Conformal sobre Ridge reconciliado
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import CONFORMAL_CONFIG, MODEL_CONFIG
from forecasting import build_mlforecast, default_lgbm_params, load_lgbm_params
from reconciliation import national_total


def _forecast_frame(df):
    return df[['unique_id', 'ds', 'y']].copy()


def fit_ridge_direct(train_df, horizon, lgbm_params=None):
    """MLForecast Ridge-only (previsao direta h=horizon)."""
    _ = lgbm_params or default_lgbm_params()
    fcst = build_mlforecast({'Ridge': Ridge(alpha=1.0)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fcst.fit(
            _forecast_frame(train_df),
            static_features=[],
            max_horizon=horizon,
            fitted=True,
        )
    return fcst


def predict_ridge(fcst, h):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fcst.predict(h=h)


def fitted_ridge(fcst):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fcst.forecast_fitted_values()


def horizon_step(ds_series, cutoff_year):
    return (pd.to_datetime(ds_series).dt.year - cutoff_year).astype(int)


def calibration_residuals(
    Y_df,
    tags,
    train_end,
    val_end,
    horizon,
    point_col='Ridge',
):
    """
    Treino <= train_end; validacao (train_end+1)..val_end com previsao direta h.
    Retorna DataFrame folha: unique_id, ds, y, yhat, abs_res, horizon_step.
    """
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])

    train = Y_df[Y_df['ds'].dt.year <= train_end].copy()
    val = Y_df[
        (Y_df['ds'].dt.year > train_end) & (Y_df['ds'].dt.year <= val_end)
    ].copy()
    val_leaf = val[val['unique_id'].isin(leaf_ids)].copy()

    lgbm_params, _ = load_lgbm_params()
    fcst = fit_ridge_direct(train, horizon, lgbm_params)
    pred = predict_ridge(fcst, h=horizon)

    merged = val_leaf[['unique_id', 'ds', 'y']].merge(
        pred[['unique_id', 'ds', point_col]],
        on=['unique_id', 'ds'],
        how='inner',
    )
    merged = merged.rename(columns={point_col: 'yhat'})
    merged['yhat'] = merged['yhat'].clip(lower=0.0)
    merged['abs_res'] = (merged['y'] - merged['yhat']).abs()
    merged['horizon_step'] = horizon_step(merged['ds'], train_end)
    return merged, fcst


def national_calibration_residuals(Y_df, tags, cal_residuals, point_col='Ridge'):
    """Residuos no total nacional (agrega y e yhat por ano)."""
    top_key = list(tags.keys())[0]
    top_ids = set(tags[top_key])
    nat_obs = (
        Y_df[Y_df['unique_id'].isin(top_ids)]
        .groupby('ds', as_index=False)['y']
        .sum()
    )
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])
    pred_leaf = cal_residuals[cal_residuals['unique_id'].isin(leaf_ids)]
    nat_hat = pred_leaf.groupby('ds', as_index=False)['yhat'].sum()
    merged = nat_obs.merge(nat_hat, on='ds', how='inner')
    merged['abs_res'] = (merged['y'] - merged['yhat']).abs()
    train_end = CONFORMAL_CONFIG['calibration_train_end']
    merged['horizon_step'] = horizon_step(merged['ds'], train_end)
    return merged


def conformal_quantile(abs_residuals, alpha):
    """Quantil split conformal simetrico."""
    r = np.asarray(abs_residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return 0.0
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(r, q_level, method='higher'))


def build_quantile_table(cal_leaf, cal_national, alphas=None):
    """
    Quantis pooled e por horizonte (folha e nacional).
    Retorna dict[alpha] -> {pooled_leaf, by_horizon_leaf, pooled_national, by_horizon_national}
    """
    alphas = alphas or CONFORMAL_CONFIG['alphas']
    out = {}
    for alpha in alphas:
        by_h_leaf = {}
        for h in sorted(cal_leaf['horizon_step'].unique()):
            sub = cal_leaf.loc[cal_leaf['horizon_step'] == h, 'abs_res']
            by_h_leaf[int(h)] = conformal_quantile(sub, alpha)

        by_h_nat = {}
        for h in sorted(cal_national['horizon_step'].unique()):
            sub = cal_national.loc[cal_national['horizon_step'] == h, 'abs_res']
            by_h_nat[int(h)] = conformal_quantile(sub, alpha)

        out[float(alpha)] = {
            'pooled_leaf': conformal_quantile(cal_leaf['abs_res'], alpha),
            'by_horizon_leaf': by_h_leaf,
            'pooled_national': conformal_quantile(cal_national['abs_res'], alpha),
            'by_horizon_national': by_h_nat,
        }
    return out


def _q_for_row(quantiles, alpha, horizon_step, level='leaf'):
    block = quantiles[float(alpha)]
    if level == 'national':
        by_h = block['by_horizon_national']
        pooled = block['pooled_national']
    else:
        by_h = block['by_horizon_leaf']
        pooled = block['pooled_leaf']
    h = int(horizon_step)
    return by_h.get(h, pooled)


def apply_intervals(
    pred_df,
    quantiles,
    cutoff_year,
    point_col,
    level='leaf',
    alphas=None,
):
    """
    Adiciona colunas lo/hi por alpha ao pred_df (unique_id, ds, point_col).
    """
    alphas = alphas or CONFORMAL_CONFIG['alphas']
    out = pred_df.copy()
    out['horizon_step'] = horizon_step(out['ds'], cutoff_year)
    for alpha in alphas:
        pct = int((1.0 - alpha) * 100)
        q_vals = out['horizon_step'].map(
            lambda h: _q_for_row(quantiles, alpha, h, level=level)
        )
        out[f'lo_{pct}'] = (out[point_col] - q_vals).clip(lower=0.0)
        out[f'hi_{pct}'] = out[point_col] + q_vals
    return out


def evaluate_coverage(intervals_df, y_col='y', point_col=None):
    """Cobertura empirica e largura media por nivel de confianca."""
    if point_col is None:
        skip = {'unique_id', 'ds', 'y', 'horizon_step', 'ano', 'nivel_1'}
        candidates = [
            c for c in intervals_df.columns
            if c not in skip and not c.startswith('lo_') and not c.startswith('hi_')
        ]
        point_col = candidates[0] if candidates else None
    results = {}
    if 'y' not in intervals_df.columns:
        return results
    for col in intervals_df.columns:
        if not col.startswith('lo_'):
            continue
        pct = col.replace('lo_', '')
        hi_col = f'hi_{pct}'
        if hi_col not in intervals_df.columns:
            continue
        inside = (
            (intervals_df['y'] >= intervals_df[col])
            & (intervals_df['y'] <= intervals_df[hi_col])
        )
        width = intervals_df[hi_col] - intervals_df[col]
        results[f'coverage_{pct}'] = float(inside.mean())
        results[f'width_mean_{pct}'] = float(width.mean())
        results[f'n_{pct}'] = int(len(intervals_df))
    if point_col and point_col in intervals_df.columns:
        results['mae_point'] = float(
            (intervals_df['y'] - intervals_df[point_col]).abs().mean()
        )
    return results


def aggregate_nivel1_intervals(intervals_leaf, panel, group_col='nivel_1'):
    """Soma folhas por setor nivel_1 e ano (ponto + limites)."""
    uid_map = panel[['unique_id', group_col]].drop_duplicates()
    df = intervals_leaf.merge(uid_map, on='unique_id', how='left')
    num_cols = [
        c for c in df.columns
        if c not in ('unique_id', 'ds', group_col, 'horizon_step', 'ano', 'y')
    ]
    agg = df.groupby([group_col, 'ds'], as_index=False)[num_cols].sum()
    if 'y' in df.columns:
        y_agg = (
            df.groupby([group_col, 'ds'], as_index=False)['y']
            .sum()
        )
        agg = agg.merge(y_agg, on=[group_col, 'ds'], how='left')
    return agg


def national_intervals_from_leaf(intervals_leaf, tags, point_col):
    """Total nacional: soma folhas (aproximacao para intervalos)."""
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])
    sub = intervals_leaf[intervals_leaf['unique_id'].isin(leaf_ids)].copy()
    num_cols = [
        c for c in sub.columns
        if c not in ('unique_id', 'ds', 'horizon_step', 'ano', 'y')
    ]
    tot = sub.groupby('ds', as_index=False)[num_cols].sum()
    tot['unique_id'] = 'Brasil'
    return tot
