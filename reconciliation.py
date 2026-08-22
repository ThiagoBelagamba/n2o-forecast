# -*- coding: utf-8 -*-
# reconciliation.py — hierarquia SEEG + MinT / BottomUp (não negativo)
import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate

from config import MODEL_CONFIG


def build_hierarchy(panel):
    """
    Agrega o painel folha em níveis + total.
    Retorna Y_df (todas as series), S_df (matriz de soma), tags.
    """
    group_cols = [c for c in MODEL_CONFIG['group_cols'] if c in panel.columns]
    # Hierarquia: Country → nivel_1 → … → unique_id (folha)
    # aggregate espera colunas de nível; unique_id da folha = última espec.
    leaf = panel.copy()
    leaf['Country'] = 'Brasil'
    # unique_id folha já existe; aggregate cria novos unique_ids por spec
    # Precisamos de uma coluna folha nomeada. Usamos as group_cols + Country.
    spec = [['Country']]
    for i in range(1, min(4, len(group_cols) + 1)):
        spec.append(['Country'] + group_cols[:i])
    # Folha completa
    spec.append(['Country'] + group_cols)

    base = leaf[['Country'] + group_cols + ['ds', 'y']].copy()
    # Preenche NaN textuais
    for c in group_cols:
        base[c] = base[c].fillna('missing').astype(str)

    Y_df, S_df, tags = aggregate(base, spec)
    print(
        f"Hierarquia: {len(tags)} niveis, "
        f"{S_df.shape[1]} folhas, {S_df.shape[0]} nos totais."
    )
    return Y_df, S_df, tags, spec


def reconcile_forecasts(Y_hat_df, Y_df, S_df, tags, Y_fitted=None, train_end_year=None):
    """
    Reconcilia previsoes base com BottomUp e MinTrace(ols).
    Nao-negatividade: clip nas previsoes reconciliadas (NNMinT QP e lento demais
    para ~800 series x 3 modelos neste hardware academico).
    Y_fitted: previsoes in-sample (mint_shrink); se ausente, usa ols/wls_struct.
    """
    Y_hist = Y_df.copy()
    if train_end_year is not None and 'ds' in Y_hist.columns:
        Y_hist = Y_hist[Y_hist['ds'].dt.year <= train_end_year].copy()

    model_cols = [
        c for c in Y_hat_df.columns
        if c not in ('unique_id', 'ds')
    ]
    if not model_cols:
        raise ValueError("Y_hat_df sem colunas de modelo para reconciliar.")

    if Y_fitted is not None:
        fit_cols = [c for c in Y_fitted.columns if c not in ('unique_id', 'ds', 'y')]
        # so as colunas que existem em Y_hat
        fit_cols = [c for c in fit_cols if c in model_cols]
        if fit_cols:
            Y_hist = Y_hist.merge(
                Y_fitted[['unique_id', 'ds'] + fit_cols],
                on=['unique_id', 'ds'],
                how='left',
            )

    reconcilers = [
        BottomUp(),
        MinTrace(method='ols'),
    ]
    if Y_fitted is not None and all(c in Y_hist.columns for c in model_cols):
        reconcilers.append(MinTrace(method='mint_shrink'))

    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    out = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_hist,
        S_df=S_df,
        tags=tags,
    )
    # Restricao fisica: emissao >= 0
    for c in out.columns:
        if c not in ('unique_id', 'ds'):
            out[c] = out[c].clip(lower=0.0)
    return out, model_cols



def clip_nonnegative_leaves_and_reaggregate(Y_hat_df, S_df, tags, model_col):
    """Fallback: clip folhas >= 0 e repropaga Bottom-Up (se MinT falhar)."""
    # tags['Country/nivel_1/...'] — última chave = folhas
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])
    df = Y_hat_df.copy()
    is_leaf = df['unique_id'].isin(leaf_ids)
    df.loc[is_leaf, model_col] = df.loc[is_leaf, model_col].clip(lower=0.0)
    return df


def leaf_forecasts_only(Y_hat_df, tags, model_col):
    """Filtra só as séries folha (nível mais baixo da hierarquia)."""
    leaf_key = list(tags.keys())[-1]
    leaf_ids = tags[leaf_key]
    return Y_hat_df[Y_hat_df['unique_id'].isin(leaf_ids)][
        ['unique_id', 'ds', model_col]
    ].copy()


def national_total(Y_hat_df, tags, model_col):
    """Série do nó Country (total Brasil)."""
    top_key = list(tags.keys())[0]
    top_ids = tags[top_key]
    return Y_hat_df[Y_hat_df['unique_id'].isin(top_ids)][
        ['unique_id', 'ds', model_col]
    ].copy()
