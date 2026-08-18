# -*- coding: utf-8 -*-
# data_loading.py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import MODEL_CONFIG, PATHS, PD_OPTIONS, PLOT_STYLE, ensure_dirs


def load_and_analyze_data(file_path=None):
    """Carrega os dados e filtra pelo gás configurado."""
    file_path = file_path or PATHS['data']
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Dataset não encontrado em:\n  {file_path}\n"
            "Copie emissao_gases.csv para a pasta data/ "
            "(o arquivo não vai no Git)."
        )
    try:
        pd.set_option('display.max_columns', PD_OPTIONS['display.max_columns'])
        plt.style.use(PLOT_STYLE)

        df = pd.read_csv(file_path)
        df_gas = df[df['gas'] == MODEL_CONFIG['gas_type']].copy()

        if len(df_gas) == 0:
            raise ValueError(
                f"Nenhum dado encontrado para {MODEL_CONFIG['gas_type']}. "
                "Verifique o nome do gás no dataset."
            )

        return df_gas

    except Exception as e:
        print(f"Erro no carregamento dos dados: {e}")
        raise


def perform_eda(df):
    """Realiza análise exploratória e salva gráficos em disco (sem plt.show)."""
    print("\n=== ANÁLISE EXPLORATÓRIA ===\n")
    ensure_dirs()

    print(f"Total de linhas: {len(df)}")
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

    print("\nResumo Estatístico das Variáveis Numéricas:")
    print(df.describe())

    plot_missing_values(df)
    plot_emission_distribution(df)
    plot_emission_boxplot(df)
    plot_correlation_matrix(df)
    plot_categorical_distributions(df)

    print("\n=== FIM DA ANÁLISE EXPLORATÓRIA ===\n")
    return df


def _save_plot(filename):
    path = os.path.join(PATHS['plots_dir'], filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo: {path}")


def plot_missing_values(df):
    """Plot valores ausentes."""
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Valores Ausentes no Dataset")
    _save_plot('eda_valores_ausentes.png')


def plot_emission_distribution(df):
    """Plot distribuição da emissão."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df[MODEL_CONFIG['target_col']], bins=30, kde=True)
    plt.title(f"Distribuição da Emissão de {MODEL_CONFIG['gas_type']}")
    plt.xlabel("Emissão (t)")
    plt.ylabel("Frequência")
    _save_plot('eda_distribuicao_emissao.png')


def plot_emission_boxplot(df):
    """Plot boxplot da emissão."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[MODEL_CONFIG['target_col']])
    plt.title(f"Boxplot da Emissão de {MODEL_CONFIG['gas_type']}")
    _save_plot('eda_boxplot_emissao.png')


def plot_correlation_matrix(df):
    """Plot matriz de correlação."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 6))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matriz de Correlação")
        _save_plot('eda_correlacao.png')


def plot_categorical_distributions(df):
    """Plot distribuição de variáveis categóricas."""
    for col in MODEL_CONFIG['cat_cols']:
        if col in df.columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(y=df[col], order=df[col].value_counts().iloc[:20].index)
            plt.title(f"Distribuição de {col} (Top 20)")
            _save_plot(f'eda_cat_{col}.png')
