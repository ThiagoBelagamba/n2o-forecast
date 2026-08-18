# data_preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_CONFIG


def preprocess_data(df):
    """Prepara os dados para modelagem e registra qualidade."""
    target = MODEL_CONFIG['target_col']
    n_raw = len(df)

    missing_pct = {}
    for col in ['produto', 'atividade_economica']:
        if col in df.columns:
            missing_pct[col] = 100.0 * df[col].isna().mean()

    n_dropped_target = int(df[target].isna().sum())
    df_clean = df.dropna(subset=[target]).copy()

    num_cols = [col for col in MODEL_CONFIG['num_cols'] if col in df_clean.columns]
    cat_cols = [col for col in MODEL_CONFIG['cat_cols'] if col in df_clean.columns]

    drop_cols = [c for c in [target, 'gas'] if c in df_clean.columns]
    X = df_clean.drop(columns=drop_cols)
    y = df_clean[target]

    quality = {
        'n_raw': n_raw,
        'n_dropped_target': n_dropped_target,
        'n_clean': len(df_clean),
        'missing_pct': missing_pct,
        'target_zeros_pct': 100.0 * (df_clean[target] == 0).mean(),
        'target_median': float(df_clean[target].median()),
        'target_mean': float(df_clean[target].mean()),
        'year_min': int(df_clean['ano'].min()) if 'ano' in df_clean.columns else None,
        'year_max': int(df_clean['ano'].max()) if 'ano' in df_clean.columns else None,
    }

    print(
        f"Qualidade: {n_raw} linhas brutas, {n_dropped_target} alvos nulos "
        f"descartados, {len(df_clean)} linhas limpas."
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
