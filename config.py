# config.py
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configurações do pandas
PD_OPTIONS = {
    'display.max_columns': None
}

# Caminhos do projeto (fonte única, sempre relativos à raiz)
PATHS = {
    'root': _ROOT,
    'data': os.path.join(_ROOT, 'data', 'emissao_gases.csv'),
    'results_dir': os.path.join(_ROOT, 'resultados'),
    'plots_dir': os.path.join(_ROOT, 'resultados', 'graficos'),
    'model': os.path.join(_ROOT, 'resultados', 'modelo_n2o.joblib'),
    'predictions': os.path.join(_ROOT, 'resultados', 'emissao_n2o_com_predicoes.csv'),
    'predictions_new': os.path.join(_ROOT, 'resultados', 'emissao_n2o_novas_predicoes.csv'),
    'report': os.path.join(_ROOT, 'resultados', 'relatorio.txt'),
}

# Configurações do modelo
MODEL_CONFIG = {
    'target_col': 'emissao',
    'gas_type': 'N2O (t)',
    'num_cols': ['ano'],
    'cat_cols': [
        'nivel_1', 'nivel_2', 'nivel_3', 'nivel_4',
        'nivel_5', 'nivel_6', 'tipo_emissao',
        'atividade_economica', 'produto'
    ],
    'split_year': 2010,  # treino <= 2010, teste >= 2011
    'cv_splits': 5,      # TimeSeriesSplit
    'random_state': 42,
    'scoring': 'neg_mean_absolute_error',
    # Grid paralelo + árvores em paralelo (Ryzen 6c/12t). Evita n_jobs aninhado
    # saturar memória: 4 processos × 3 threads ≈ 12.
    'grid_n_jobs': 4,
    'rf_n_jobs': 3,
    'rf_param_grid': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 5]
    },
    'ridge_param_grid': {
        'model__alpha': [0.1, 1.0, 10.0, 100.0]
    },
}

# Configurações de visualização
PLOT_STYLE = 'ggplot'


def ensure_dirs():
    """Garante que as pastas de saída existam."""
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    os.makedirs(PATHS['plots_dir'], exist_ok=True)
