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
    'lgbm_n_estimators': 300,
    'lgbm_learning_rate': 0.05,
    'lgbm_num_leaves': 31,
    'tweedie_powers': [1.3, 1.5, 1.7],
    'baseline_rf_estimators': 100,
    'baseline_rf_max_depth': 10,
}

PLOT_STYLE = 'ggplot'


def ensure_dirs():
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    os.makedirs(PATHS['plots_dir'], exist_ok=True)
