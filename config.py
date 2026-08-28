# config.py
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

PD_OPTIONS = {
    'display.max_columns': None
}

PATHS = {
    'root': _ROOT,
    'data': os.path.join(_ROOT, 'data', 'Dados-nacionais-13.0.xlsx'),
    'data_cache': os.path.join(_ROOT, 'data', 'n2o_nacional_longo.csv'),
    'results_dir': os.path.join(_ROOT, 'resultados'),
    'plots_dir': os.path.join(_ROOT, 'resultados', 'graficos'),
    'model': os.path.join(_ROOT, 'resultados', 'modelo_n2o.joblib'),
    'predictions': os.path.join(_ROOT, 'resultados', 'emissao_n2o_com_predicoes.csv'),
    'predictions_new': os.path.join(_ROOT, 'resultados', 'emissao_n2o_novas_predicoes.csv'),
    'projection': os.path.join(_ROOT, 'resultados', 'projecao_2025_2026.csv'),
    'comparacao': os.path.join(
        _ROOT, 'resultados', 'comparacao_real_vs_previsto.csv'
    ),
    'report': os.path.join(_ROOT, 'resultados', 'relatorio.txt'),
    'metrics': os.path.join(_ROOT, 'resultados', 'metricas_comparacao.csv'),
    'timing': os.path.join(_ROOT, 'resultados', 'tempo_inferencia.json'),
    'lgbm_params': os.path.join(_ROOT, 'resultados', 'lgbm_best_params.json'),
    'lgbm_cv': os.path.join(_ROOT, 'resultados', 'cv_lgbm_grid.csv'),
    'ensemble_weights': os.path.join(_ROOT, 'resultados', 'ensemble_weights.json'),
    'ensemble_benchmark': os.path.join(_ROOT, 'resultados', 'ensemble_benchmark.json'),
}

# Barreira MAE (Ridge/MinTrace ols no teste cego 2020-2024)
RIDGE_MAE_BARRIER = 83.8493346419633

LGBM_GRID = {
    'tweedie_variance_power': [round(x, 1) for x in __import__('numpy').arange(1.1, 2.0, 0.1)],
    'max_depth': [3, 4, 5],
    'min_data_in_leaf': [10, 25, 50],
    'reg_alpha': [0.1, 1.0, 5.0],
    'reg_lambda': [0.1, 1.0, 5.0],
}

LGBM_DEFAULTS = {
    'tweedie_variance_power': 1.7,
    'max_depth': 4,
    'min_data_in_leaf': 25,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'num_leaves': 15,
    'n_estimators': 200,
    'learning_rate': 0.05,
}

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

SEEG_C13_GROUP_COLS = [
    'tipo_emissao', 'gas', 'nivel_1', 'nivel_2', 'nivel_3',
    'nivel_4', 'nivel_5', 'produto', 'atividade_economica',
]

MODEL_CONFIG = {
    'target_col': 'emissao',
    'gas_type': 'N2O (t)',
    'emission_type': 'Emissão',
    'group_cols': [
        'nivel_1', 'nivel_2', 'nivel_3', 'nivel_4', 'nivel_5',
        'tipo_emissao', 'atividade_economica', 'produto',
    ],
    'cat_cols': [
        'nivel_1', 'nivel_2', 'nivel_3', 'nivel_4', 'nivel_5',
        'tipo_emissao', 'atividade_economica', 'produto',
    ],
    'num_cols': ['ano'],
    'hierarchy_levels': [
        ['nivel_1'],
        ['nivel_1', 'nivel_2'],
        ['nivel_1', 'nivel_2', 'nivel_3'],
    ],
    'split_year': 2019,
    'test_horizon': 5,  # 2020–2024
    'forecast_years': [2025, 2026],
    'cv_splits': 3,
    'random_state': 42,
    'lgbm_n_estimators': 200,
    'lgbm_learning_rate': 0.05,
    'lgbm_num_leaves': 15,
    'tweedie_powers': [round(x, 1) for x in __import__('numpy').arange(1.1, 2.0, 0.1)],
    'mlforecast_lags': [1, 2, 3, 5],
    'rolling_windows': [3, 5],
    'lgbm_tune_n_random': 40,
    'lgbm_tune_sample_series': 150,
    'lgbm_tune_refine_top': 5,
    'grid_n_jobs': 4,
    'rf_n_jobs': 3,
    'baseline_rf_estimators': 100,
    'baseline_rf_max_depth': 10,
    # Ensemble Ridge + LGBM (peso do Ridge; complemento = LGBM)
    'ensemble_ridge_lgbm': {
        'ridge_col': 'Ridge',
        'lgbm_col': 'LGBM_Tweedie',
        'out_col': 'Ensemble_Ridge_LGBM',
        'default_ridge_weight': 0.5,
        'weight_grid': [0.3, 0.4, 0.5, 0.6, 0.7],
    },
    'ensemble_ridge_nhits': {
        'ridge_col': 'Ridge',
        'nhits_col': 'NHITS',
        'out_col': 'Ensemble_Ridge_NHITS',
        'weight_grid': [0.3, 0.4, 0.5, 0.6, 0.7],
    },
}

PLOT_STYLE = 'ggplot'


def ensure_dirs():
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    os.makedirs(PATHS['plots_dir'], exist_ok=True)
