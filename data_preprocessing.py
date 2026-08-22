# data_preprocessing.py
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_CONFIG


def add_sector_lags(df):
    """
    Adiciona defasagens por setor (só passado):
    emissao_lag1, emissao_lag2, delta_lag1 = lag1 - lag2.
    Remove linhas sem lag1 (início da série).
    """
    target = MODEL_CONFIG['target_col']
    group_cols = [c for c in MODEL_CONFIG['group_cols'] if c in df.columns]
    if not group_cols:
        raise ValueError("Nenhuma coluna de grupo setorial para calcular lags.")
    if 'ano' not in df.columns:
        raise ValueError("Coluna 'ano' obrigatória para lags.")

    out = df.copy()
    out['_grupo'] = (
        out[group_cols]
        .fillna('missing')
        .astype(str)
        .agg('|'.join, axis=1)
    )
    out = out.sort_values(['_grupo', 'ano'], kind='mergesort')
    g = out.groupby('_grupo', sort=False)[target]
    out['emissao_lag1'] = g.shift(1)
    out['emissao_lag2'] = g.shift(2)
    out['delta_lag1'] = out['emissao_lag1'] - out['emissao_lag2']
    n_before = len(out)
    out = out.dropna(subset=['emissao_lag1']).drop(columns=['_grupo'])
    print(
        f"Lags por setor: {n_before} -> {len(out)} linhas "
        f"(descartadas {n_before - len(out)} sem lag1)."
    )
    return out.reset_index(drop=True)


def preprocess_data(df):
    """Prepara os dados para modelagem (lags + qualidade)."""
    target = MODEL_CONFIG['target_col']
    n_raw = len(df)

    missing_pct = {}
    for col in ['produto', 'atividade_economica']:
        if col in df.columns:
            missing_pct[col] = 100.0 * df[col].isna().mean()

    n_dropped_target = int(df[target].isna().sum())
    df_clean = df.dropna(subset=[target]).copy()
    df_clean = add_sector_lags(df_clean)

    num_cols = [col for col in MODEL_CONFIG['num_cols'] if col in df_clean.columns]
    cat_cols = [col for col in MODEL_CONFIG['cat_cols'] if col in df_clean.columns]

    drop_cols = [c for c in [target, 'gas'] if c in df_clean.columns]
    X = df_clean.drop(columns=drop_cols)
    y = df_clean[target]

    y_clean = df_clean[target]
    n_top10 = max(1, int(np.ceil(0.1 * len(y_clean))))
    top10 = y_clean.nlargest(n_top10)
    y_sum = float(y_clean.sum())

    quality = {
        'n_raw': n_raw,
        'n_dropped_target': n_dropped_target,
        'n_clean': len(df_clean),
        'missing_pct': missing_pct,
        'target_zeros_pct': 100.0 * (y_clean == 0).mean(),
        'target_median': float(y_clean.median()),
        'target_mean': float(y_clean.mean()),
        'target_max': float(y_clean.max()),
        'target_p90': float(y_clean.quantile(0.90)),
        'target_p99': float(y_clean.quantile(0.99)),
        'share_em_top10_pct': (100.0 * float(top10.sum()) / y_sum) if y_sum else 0.0,
        'year_min': int(df_clean['ano'].min()) if 'ano' in df_clean.columns else None,
        'year_max': int(df_clean['ano'].max()) if 'ano' in df_clean.columns else None,
        'has_lags': True,
        'split_year': MODEL_CONFIG['split_year'],
    }

    print(
        f"Qualidade: {n_raw} linhas brutas, {n_dropped_target} alvos nulos "
        f"descartados, {len(df_clean)} linhas limpas (com lags)."
    )
    return X, y, num_cols, cat_cols, quality


def create_preprocessor(num_cols, cat_cols):
    """Cria o pipeline de pré-processamento (ainda não ajustado)."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ],
        remainder='drop',
    )
    return preprocessor


def split_data_chronological(X, y):
    """Divide os dados por ano: treino <= split_year, teste > split_year."""
    if 'ano' not in X.columns:
        raise ValueError("A coluna 'ano' é obrigatória para o split cronológico.")

    corte = MODEL_CONFIG['split_year']
    treino = X['ano'] <= corte
    X_train, X_test = X[treino], X[~treino]
    y_train, y_test = y[treino], y[~treino]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Split cronológico inválido em {corte}: "
            f"treino={len(X_train)}, teste={len(X_test)}."
        )

    print(
        f"Split cronológico (corte {corte}): "
        f"treino {len(X_train)} linhas "
        f"({int(X_train['ano'].min())}–{int(X_train['ano'].max())}) | "
        f"teste {len(X_test)} linhas "
        f"({int(X_test['ano'].min())}–{int(X_test['ano'].max())})"
    )
    return X_train, X_test, y_train, y_test
