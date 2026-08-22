# -*- coding: utf-8 -*-
# data_preprocessing.py — painel unique_id / ds / y (lags via MLForecast)
import numpy as np
import pandas as pd

from config import MODEL_CONFIG


def make_unique_id(df, group_cols=None):
    """Concatena chaves setoriais em unique_id estável."""
    cols = group_cols or [
        c for c in MODEL_CONFIG['group_cols'] if c in df.columns
    ]
    if not cols:
        raise ValueError("Nenhuma coluna de grupo para unique_id.")
    filled = df[cols].fillna('missing').astype(str)
    return filled.agg('|'.join, axis=1)


def to_nixtla_panel(df):
    """
    Converte o painel nacional SEEG para o contrato Nixtla:
    unique_id, ds (Timestamp anual), y, + colunas de hierarquia.
    Não remove séries só-zero (necessárias para a matriz S).
    """
    target = MODEL_CONFIG['target_col']
    group_cols = [c for c in MODEL_CONFIG['group_cols'] if c in df.columns]

    out = df.copy()
    n_dropped = int(out[target].isna().sum())
    out = out.dropna(subset=[target]).copy()
    out[target] = out[target].astype(float).clip(lower=0.0)

    for c in group_cols:
        out[c] = out[c].fillna('missing').astype(str)

    out['unique_id'] = make_unique_id(out, group_cols)
    out['ds'] = pd.to_datetime(out['ano'].astype(int).astype(str) + '-12-31')
    out['y'] = out[target]

    keep = ['unique_id', 'ds', 'y', 'ano'] + group_cols
    panel = out[keep].sort_values(['unique_id', 'ds']).reset_index(drop=True)

    y = panel['y']
    n_top10 = max(1, int(np.ceil(0.1 * len(y))))
    top10 = y.nlargest(n_top10)
    y_sum = float(y.sum())

    quality = {
        'n_raw': len(df),
        'n_dropped_target': n_dropped,
        'n_clean': len(panel),
        'n_series': int(panel['unique_id'].nunique()),
        'missing_pct': {
            c: 100.0 * float((df[c].isna().mean() if c in df.columns else 0.0))
            for c in ['produto', 'atividade_economica']
        },
        'target_zeros_pct': 100.0 * float((y == 0).mean()),
        'target_median': float(y.median()),
        'target_mean': float(y.mean()),
        'target_max': float(y.max()),
        'target_p90': float(y.quantile(0.90)),
        'target_p99': float(y.quantile(0.99)),
        'share_em_top10_pct': (100.0 * float(top10.sum()) / y_sum) if y_sum else 0.0,
        'year_min': int(panel['ano'].min()),
        'year_max': int(panel['ano'].max()),
        'split_year': MODEL_CONFIG['split_year'],
        'protocol': 'multi_year_direct_no_test_lag',
    }
    print(
        f"Painel Nixtla: {quality['n_clean']} linhas, "
        f"{quality['n_series']} series, "
        f"{quality['year_min']}–{quality['year_max']} "
        f"(zeros={quality['target_zeros_pct']:.1f}%)."
    )
    return panel, quality


def split_panel_chronological(panel):
    """Treino <= split_year, teste > split_year (só para métricas / holdout y)."""
    corte = MODEL_CONFIG['split_year']
    train = panel[panel['ano'] <= corte].copy()
    test = panel[panel['ano'] > corte].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Split inválido em {corte}: treino={len(train)}, teste={len(test)}."
        )
    print(
        f"Split cronologico (corte {corte}): "
        f"treino {len(train)} ({int(train['ano'].min())}–{int(train['ano'].max())}) | "
        f"teste {len(test)} ({int(test['ano'].min())}–{int(test['ano'].max())})"
    )
    return train, test
