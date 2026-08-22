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
    'data': os.path.join(_ROOT, 'data', 'Dados-nacionais-13.0.xlsx'),
    'data_cache': os.path.join(_ROOT, 'data', 'n2o_nacional_longo.csv'),
    'results_dir': os.path.join(_ROOT, 'resultados'),
    'plots_dir': os.path.join(_ROOT, 'resultados', 'graficos'),
    'model': os.path.join(_ROOT, 'resultados', 'modelo_n2o.joblib'),
    'predictions': os.path.join(_ROOT, 'resultados', 'emissao_n2o_com_predicoes.csv'),
    'predictions_new': os.path.join(_ROOT, 'resultados', 'emissao_n2o_novas_predicoes.csv'),
    'report': os.path.join(_ROOT, 'resultados', 'relatorio.txt'),
}

# Mapeamento Coleção 13 (wide) → nomes internos do pipeline
SEEG_C13_COLUMNS = {
    'Emissão/Remoção/Bunker': 'tipo_emissao',
    'Gás': 'gas',
    'Setor de emissão': 'nivel_1',
    'Categoria emissora': 'nivel_2',
    'Sub-categoria emissora': 'nivel_3',
    'Detalhamento': 'nivel_4',
    'Recorte': 'nivel_5',
    'Produto ou sistema': 'produto',
    'Atividade geral': 'atividade_economica',
}

# Colunas de id setorial (Estado/Bioma são somados ao nacional)
SEEG_C13_GROUP_COLS = [
    'tipo_emissao', 'gas', 'nivel_1', 'nivel_2', 'nivel_3',
    'nivel_4', 'nivel_5', 'produto', 'atividade_economica',
]

# Configurações do modelo
MODEL_CONFIG = {
    'target_col': 'emissao',
    'gas_type': 'N2O (t)',
    'emission_type': 'Emissão',
    'num_cols': ['ano', 'emissao_lag1', 'emissao_lag2', 'delta_lag1'],
    'cat_cols': [
        'nivel_1', 'nivel_2', 'nivel_3', 'nivel_4', 'nivel_5',
        'tipo_emissao', 'atividade_economica', 'produto',
    ],
    'group_cols': [
        'nivel_1', 'nivel_2', 'nivel_3', 'nivel_4', 'nivel_5',
        'tipo_emissao', 'atividade_economica', 'produto',
    ],
    'lag_cols': ['emissao_lag1', 'emissao_lag2', 'delta_lag1'],
    'split_year': 2019,  # treino <= 2019, teste 2020–2024
    'cv_splits': 5,
    'random_state': 42,
    'scoring': 'neg_mean_absolute_error',
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
