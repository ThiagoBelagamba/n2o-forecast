# -*- coding: utf-8 -*-
# forecasting_nhits.py — NHITS multi-ano (NeuralForecast) + expansao hierarquica
import warnings

import numpy as np
import pandas as pd

from config import MODEL_CONFIG


def _nf_panel(df):
    out = df[['unique_id', 'ds', 'y']].copy()
    out['ds'] = pd.to_datetime(out['ds'])
    out['y'] = out['y'].astype(float).clip(lower=0.0)
    return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)


def expand_leaf_to_hierarchy(leaf_hat, S_df, model_col='NHITS'):
    """
    Propaga previsoes folha para todos os nos via matriz S (Bottom-Up).
    leaf_hat: unique_id, ds, model_col (so folhas).
    """
    leaf_hat = leaf_hat.copy()
    leaf_hat['ds'] = pd.to_datetime(leaf_hat['ds'])
    leaf_cols = [c for c in S_df.columns if c != 'unique_id']
    leaf_hat = leaf_hat[leaf_hat['unique_id'].isin(leaf_cols)]
    node_ids = S_df['unique_id'].tolist()
    S = S_df[leaf_cols].astype(float).values

    rows = []
    for ds, grp in leaf_hat.groupby('ds'):
        s = grp.set_index('unique_id')[model_col]
        y_leaf = np.array(
            [float(s.get(uid, 0.0)) for uid in leaf_cols], dtype=float
        )
        y_all = S @ y_leaf
        for uid, val in zip(node_ids, y_all):
            rows.append({'unique_id': uid, 'ds': ds, model_col: float(val)})
    return pd.DataFrame(rows)


def fit_predict_nhits(train_df, horizon, max_steps=150, input_size=12):
    """
    Treina NHITS no painel folha (<= corte) e preve h=horizon passos.
    Retorna Y_hat folha + Y_hat hierarquia expandida.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    panel = _nf_panel(train_df)
    models = [
        NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            early_stop_patience_steps=8,
            learning_rate=1e-3,
            batch_size=32,
            random_seed=MODEL_CONFIG['random_state'],
            accelerator='cpu',
        )
    ]
    nf = NeuralForecast(models=models, freq='YE')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nf.fit(df=panel, val_size=horizon)
        pred = nf.predict()
    pred = pred.reset_index()
    pred.columns = [c if c != 'NHITS' else 'NHITS' for c in pred.columns]
    if 'NHITS' not in pred.columns:
        # neuralforecast pode nomear coluna pelo alias do modelo
        val_cols = [c for c in pred.columns if c not in ('unique_id', 'ds', 'y')]
        if not val_cols:
            raise RuntimeError('NHITS sem coluna de previsao.')
        pred = pred.rename(columns={val_cols[0]: 'NHITS'})
    pred['NHITS'] = pred['NHITS'].clip(lower=0.0)
    return nf, pred


def nhits_leaf_mae(pred_leaf, test_leaf, col='NHITS'):
    merged = test_leaf[['unique_id', 'ds', 'y']].merge(
        pred_leaf[['unique_id', 'ds', col]],
        on=['unique_id', 'ds'],
        how='inner',
    )
    if merged.empty:
        return float('inf'), merged
    err = np.abs(merged['y'] - merged[col].clip(lower=0.0)).mean()
    return float(err), merged
