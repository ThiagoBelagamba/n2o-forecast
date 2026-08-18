# model_training.py
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from config import MODEL_CONFIG


def compute_metrics(y_true, y_pred):
    """Métricas em escala interpretável, usadas por modelos e baselines."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'MedAE': float(median_absolute_error(y_true, y_pred)),
    }


def _filled_groups(X):
    """Chaves de grupo com NaN preenchido para o merge funcionar."""
    cols = [c for c in MODEL_CONFIG['cat_cols'] if c in X.columns]
    return X[cols].fillna('missing')


def _predict_group_mean(X_train, y_train, X_test):
    tmp = _filled_groups(X_train)
    tmp['_y'] = np.asarray(y_train)
    group_cols = list(tmp.columns[:-1])
    means = tmp.groupby(group_cols, as_index=False)['_y'].mean()
    merged = _filled_groups(X_test).merge(means, on=group_cols, how='left')
    return merged['_y'].fillna(np.mean(y_train)).to_numpy()


def _predict_persistence(X_train, y_train, X_test):
    """Último valor observado de cada grupo no conjunto de treino."""
    tmp = _filled_groups(X_train)
    group_cols = list(tmp.columns)
    tmp['ano'] = X_train['ano'].to_numpy()
    tmp['_y'] = np.asarray(y_train)
    last = (
        tmp.sort_values('ano')
        .groupby(group_cols, as_index=False)
        .tail(1)
        .drop(columns=['ano'])
    )
    merged = _filled_groups(X_test).merge(last, on=group_cols, how='left')
    return merged['_y'].fillna(np.mean(y_train)).to_numpy()


def create_baselines(X_train, y_train, X_test, y_test):
    """Baselines ingênuos e temporais para comparação justa."""
    baseline_results = {}

    for name, strategy, kwargs in [
        ('mean', 'mean', {}),
        ('median', 'median', {}),
        ('constant_zero', 'constant', {'constant': 0}),
    ]:
        dummy = DummyRegressor(strategy=strategy, **kwargs)
        dummy.fit(X_train, y_train)
        baseline_results[name] = compute_metrics(y_test, dummy.predict(X_test))

    baseline_results['media_por_grupo'] = compute_metrics(
        y_test, _predict_group_mean(X_train, y_train, X_test)
    )
    baseline_results['persistencia'] = compute_metrics(
        y_test, _predict_persistence(X_train, y_train, X_test)
    )
    return baseline_results


def _sort_by_year(X, y):
    order = X['ano'].to_numpy().argsort(kind='mergesort')
    if isinstance(X, pd.DataFrame):
        X_sorted = X.iloc[order]
    else:
        X_sorted = X[order]
    if isinstance(y, pd.Series):
        y_sorted = y.iloc[order]
    else:
        y_sorted = np.asarray(y)[order]
    return X_sorted, y_sorted


def _fit_grid(name, estimator, param_grid, preprocessor, X_train, y_train, tscv):
    pipeline = Pipeline(steps=[
        ('preprocessor', clone(preprocessor)),
        ('model', estimator),
    ])
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring=MODEL_CONFIG['scoring'],
        n_jobs=2,
        verbose=1,
        refit=True,
    )
    print(f"\nTreinando {name}...")
    grid_search.fit(X_train, y_train)
    print(f"{name} — melhores parâmetros: {grid_search.best_params_}")
    print(f"{name} — MAE (CV): {-grid_search.best_score_:.4f}")
    return grid_search


def train_models(X_train, y_train, preprocessor):
    """Treina Random Forest e Ridge com TimeSeriesSplit (treino ordenado por ano)."""
    X_sorted, y_sorted = _sort_by_year(X_train, y_train)
    tscv = TimeSeriesSplit(n_splits=MODEL_CONFIG['cv_splits'])

    searches = {}
    searches['random_forest'] = _fit_grid(
        'Random Forest',
        RandomForestRegressor(random_state=MODEL_CONFIG['random_state']),
        MODEL_CONFIG['rf_param_grid'],
        preprocessor,
        X_sorted,
        y_sorted,
        tscv,
    )
    searches['ridge'] = _fit_grid(
        'Ridge',
        Ridge(),
        MODEL_CONFIG['ridge_param_grid'],
        preprocessor,
        X_sorted,
        y_sorted,
        tscv,
    )
    return searches


def evaluate_model(model, X_test, y_test):
    """Avalia o modelo nos dados de teste."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(
        f"Teste — MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | "
        f"R2: {metrics['R2']:.4f} | MedAE: {metrics['MedAE']:.4f}"
    )
    return y_pred, metrics


if __name__ == "__main__":
    print("Funções disponíveis neste módulo:")
    print("- train_models")
    print("- evaluate_model")
    print("- create_baselines")
    print("- compute_metrics")
