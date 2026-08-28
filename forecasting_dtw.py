# -*- coding: utf-8 -*-
# forecasting_dtw.py — clustering DTW + modelos por cluster (folha)
import warnings

import numpy as np
import pandas as pd

from config import DTW_CONFIG, MODEL_CONFIG
from forecasting import fit_direct_models, load_lgbm_params, predict_horizon
from forecasting_nhits import expand_leaf_to_hierarchy


def panel_to_hier_uid(uid):
    """Painel folha (|) -> unique_id hierarquico (Brasil/...)."""
    return 'Brasil/' + str(uid).replace('|', '/')


def build_leaf_matrix(Y_df, leaf_ids, train_end_year):
    """
    Matriz (n_series, n_timesteps) alinhada por ano para DTW.
    Usa unique_id da hierarquia (tags folha).
    """
    sub = Y_df[
        (Y_df['unique_id'].isin(leaf_ids))
        & (Y_df['ds'].dt.year <= train_end_year)
    ].copy()
    years = sorted(sub['ds'].dt.year.unique())
    uids = sorted(sub['unique_id'].unique())
    mat = np.zeros((len(uids), len(years)), dtype=float)
    uid_idx = {u: i for i, u in enumerate(uids)}
    year_idx = {y: j for j, y in enumerate(years)}
    for _, row in sub.iterrows():
        mat[uid_idx[row['unique_id']], year_idx[row['ds'].year]] = float(row['y'])
    return mat, uids, years


def transform_series_matrix(mat, method='log1p_zscore'):
    """log1p + z-score por serie (eixo 1)."""
    x = np.log1p(np.clip(mat, 0, None))
    if method == 'log1p_zscore':
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-9, 1.0, sd)
        return (x - mu) / sd
    return x


def cluster_series_dtw(mat, uids, n_clusters, random_state=None):
    """
    K-Means com metrica DTW (tslearn).
    Series com soma zero recebem cluster -1 (tratadas separadamente).
    """
    from tslearn.clustering import TimeSeriesKMeans

    rs = random_state or DTW_CONFIG['random_state']
    total = mat.sum(axis=1)
    zero_mask = total <= 0
    active_idx = np.where(~zero_mask)[0]

    labels = np.full(len(uids), -1, dtype=int)
    if len(active_idx) == 0:
        return pd.DataFrame({'unique_id': uids, 'cluster': labels})

    X = mat[active_idx]
    X = transform_series_matrix(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric='dtw',
        max_iter=10,
        random_state=rs,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        km.fit(X)
    labels[active_idx] = km.labels_

    return pd.DataFrame({'unique_id': uids, 'cluster': labels})


def select_k_silhouette(mat, uids, k_candidates=None, sample_n=None, random_state=None):
    """Escolhe K com silhouette (proxy euclidiano em series normalizadas)."""
    from sklearn.metrics import silhouette_score

    k_candidates = k_candidates or DTW_CONFIG['k_candidates']
    rs = random_state or DTW_CONFIG['random_state']
    total = mat.sum(axis=1)
    active_idx = np.where(total > 0)[0]
    if len(active_idx) < max(k_candidates) + 1:
        return DTW_CONFIG['n_clusters'], {}

    if sample_n and len(active_idx) > sample_n:
        rng = np.random.default_rng(rs)
        active_idx = rng.choice(active_idx, size=sample_n, replace=False)

    from tslearn.clustering import TimeSeriesKMeans

    X = transform_series_matrix(mat[active_idx])
    X3 = X.reshape(X.shape[0], X.shape[1], 1)
    X_flat = X.reshape(X.shape[0], -1)

    scores = {}
    best_k, best_s = k_candidates[0], -1.0
    for k in k_candidates:
        if k >= len(active_idx):
            continue
        km = TimeSeriesKMeans(
            n_clusters=k, metric='dtw', max_iter=8, random_state=rs, n_jobs=1
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lbl = km.fit_predict(X3)
        if len(set(lbl)) < 2:
            continue
        s = float(silhouette_score(X_flat, lbl))
        scores[k] = s
        if s > best_s:
            best_s, best_k = s, k
    return best_k, scores


def fit_predict_per_cluster(
    Y_train,
    leaf_ids,
    cluster_df,
    lgbm_params,
    horizon,
    models=None,
):
    """
    Treina MLForecast por cluster (folhas) e concatena previsoes folha.
    Cluster -1: persistencia (ultimo valor treino).
    """
    models = models or DTW_CONFIG['models']
    train_leaf = Y_train[Y_train['unique_id'].isin(leaf_ids)].copy()
    preds = []

    for cl in sorted(cluster_df['cluster'].unique()):
        uids = cluster_df.loc[cluster_df['cluster'] == cl, 'unique_id'].tolist()
        sub = train_leaf[train_leaf['unique_id'].isin(uids)]
        if sub.empty:
            continue
        if cl == -1:
            last = (
                sub.sort_values('ds')
                .groupby('unique_id', as_index=False)
                .tail(1)[['unique_id', 'y']]
                .set_index('unique_id')['y']
            )
            years = list(range(
                MODEL_CONFIG['split_year'] + 1,
                MODEL_CONFIG['split_year'] + horizon + 1,
            ))
            rows = []
            for y in years:
                ds = pd.Timestamp(f'{y}-12-31')
                for uid in uids:
                    lv = float(last.get(uid, 0.0))
                    row = {'unique_id': uid, 'ds': ds}
                    for m in models:
                        row[m] = lv
                    rows.append(row)
            preds.append(pd.DataFrame(rows))
            continue

        fcst = fit_direct_models(sub, lgbm_params, horizon)
        pred = predict_horizon(fcst, h=horizon)
        keep = ['unique_id', 'ds'] + [m for m in models if m in pred.columns]
        preds.append(pred[keep])

    if not preds:
        raise RuntimeError('Sem previsoes por cluster.')
    out = pd.concat(preds, ignore_index=True)
    return out


def merge_cluster_metadata(cluster_df, panel):
    """Anexa nivel_1 ao CSV de clusters."""
    pmap = panel[['unique_id', 'nivel_1']].drop_duplicates().copy()
    pmap['hier_uid'] = pmap['unique_id'].map(panel_to_hier_uid)
    return cluster_df.merge(
        pmap[['hier_uid', 'nivel_1']],
        left_on='unique_id',
        right_on='hier_uid',
        how='left',
    ).drop(columns=['hier_uid'], errors='ignore')


def expand_and_reconcile_leaf_hat(leaf_hat, S_df, Y_df, tags, Y_fitted=None, train_end_year=None):
    """Expande colunas folha e reconcilia MinT."""
    from reconciliation import reconcile_forecasts

    model_cols = [c for c in leaf_hat.columns if c not in ('unique_id', 'ds')]
    Y_hat = None
    for col in model_cols:
        exp = expand_leaf_to_hierarchy(
            leaf_hat[['unique_id', 'ds', col]], S_df, model_col=col
        )
        if Y_hat is None:
            Y_hat = exp
        else:
            Y_hat = Y_hat.merge(exp, on=['unique_id', 'ds'], how='outer')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y_rec, _ = reconcile_forecasts(
            Y_hat, Y_df, S_df, tags,
            Y_fitted=Y_fitted,
            train_end_year=train_end_year,
        )
    return Y_rec
