# -*- coding: utf-8 -*-
# results_saving.py
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MODEL_CONFIG, PATHS, PLOT_STYLE, ensure_dirs
from model_training import compute_metrics


def save_results(
    best_model,
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    num_cols,
    cat_cols,
    baseline_results=None,
    data_quality=None,
    model_metrics=None,
    best_model_name=None,
):
    """Salva modelo, predições, relatório e gráficos."""
    ensure_dirs()

    joblib.dump(
        best_model.best_estimator_ if hasattr(best_model, 'best_estimator_') else best_model,
        PATHS['model'],
    )
    save_predictions(best_model, X_train, y_train, X_test, y_test)
    save_report(
        best_model,
        y_test,
        y_pred,
        baseline_results=baseline_results,
        data_quality=data_quality,
        model_metrics=model_metrics,
        best_model_name=best_model_name,
    )
    save_plots(best_model, X_test, y_test, y_pred)
    print(f"Arquivos salvos em '{PATHS['results_dir']}'")


def save_predictions(model, X_train, y_train, X_test, y_test):
    """Salva predições de treino e teste com coluna conjunto."""
    frames = []
    for nome, X, y in [('treino', X_train, y_train), ('teste', X_test, y_test)]:
        df = X.copy()
        df[MODEL_CONFIG['target_col']] = np.asarray(y)
        df['emissao_predita'] = np.round(model.predict(X), 8)
        df['conjunto'] = nome
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(PATHS['predictions'], index=False)


def _format_metrics_block(metrics):
    return (
        f"MAE:   {metrics['MAE']:.4f}\n"
        f"RMSE:  {metrics['RMSE']:.4f}\n"
        f"R2:    {metrics['R2']:.4f}\n"
        f"MedAE: {metrics['MedAE']:.4f}\n"
    )


def save_report(
    model,
    y_test,
    y_pred,
    baseline_results=None,
    data_quality=None,
    model_metrics=None,
    best_model_name=None,
):
    """Salva relatório de métricas, qualidade de dados e comparação justa."""
    final_metrics = compute_metrics(y_test, y_pred)

    with open(PATHS['report'], 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DO MODELO ===\n\n")
        f.write(
            f"Split cronológico: treino anos <= {MODEL_CONFIG['split_year']}, "
            f"teste anos > {MODEL_CONFIG['split_year']}\n"
        )
        f.write(f"Validação: TimeSeriesSplit (n_splits={MODEL_CONFIG['cv_splits']})\n")
        f.write(f"Scoring do grid: {MODEL_CONFIG['scoring']}\n\n")

        if data_quality:
            f.write("=== QUALIDADE DOS DADOS (N2O) ===\n")
            f.write(f"Linhas após filtro de gás: {data_quality['n_raw']}\n")
            f.write(
                f"Alvos nulos descartados: {data_quality['n_dropped_target']}\n"
            )
            f.write(f"Linhas usadas na modelagem: {data_quality['n_clean']}\n")
            for col, pct in data_quality.get('missing_pct', {}).items():
                f.write(f"Ausência em {col}: {pct:.2f}%\n")
            f.write(f"Zeros no alvo: {data_quality['target_zeros_pct']:.2f}%\n")
            f.write(f"Mediana do alvo: {data_quality['target_median']:.4f} t\n")
            f.write(f"Média do alvo: {data_quality['target_mean']:.4f} t\n")
            f.write(
                f"Período: {data_quality['year_min']}–{data_quality['year_max']}\n\n"
            )

        nome = best_model_name or 'modelo'
        f.write(f"Modelo selecionado (menor MAE de CV): {nome}\n")
        if hasattr(model, 'best_params_'):
            f.write(f"Melhores parâmetros: {model.best_params_}\n")
            f.write(f"MAE de CV: {-model.best_score_:.4f}\n")
        f.write("\nDesempenho no teste:\n")
        f.write(_format_metrics_block(final_metrics))

        if model_metrics:
            f.write("\n=== MODELOS COMPARADOS NO TESTE ===\n")
            for name, metrics in model_metrics.items():
                f.write(f"\n{name}\n")
                f.write(_format_metrics_block(metrics))

        if baseline_results:
            f.write("\n=== COMPARAÇÃO COM BASELINES (TESTE) ===\n")
            for name, metrics in baseline_results.items():
                f.write(f"\n{name}\n")
                f.write(_format_metrics_block(metrics))

            persist = baseline_results.get('persistencia')
            if persist:
                mae_delta = persist['MAE'] - final_metrics['MAE']
                rmse_delta = persist['RMSE'] - final_metrics['RMSE']
                f.write("\n=== GANHO VERSUS PERSISTÊNCIA ===\n")
                f.write(
                    "Valores positivos = o modelo erra menos que repetir "
                    "o último valor observado de cada setor (2010).\n"
                )
                f.write(f"Δ MAE  (persistência − modelo): {mae_delta:.4f} t\n")
                f.write(f"Δ RMSE (persistência − modelo): {rmse_delta:.4f} t\n")
                if mae_delta <= 0:
                    f.write(
                        "\nO modelo não supera a persistência em MAE. "
                        "Árvores e Ridge com só 'ano' + categorias tendem a "
                        "repetir o nível conhecido do setor e não extrapolam "
                        "tendência além de 2010.\n"
                    )


def save_plots(model, X_test, y_test, y_pred):
    """Salva gráficos de avaliação."""
    plt.style.use(PLOT_STYLE)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    y_min = min(np.min(y_test), np.min(y_pred))
    y_max = max(np.max(y_test), np.max(y_pred))
    plt.plot([y_min, y_max], [y_min, y_max], '--r')
    plt.xlabel("Valores Reais (t)")
    plt.ylabel("Valores Preditos (t)")
    plt.title("Valores Reais vs Preditos — conjunto de teste")
    path = os.path.join(PATHS['plots_dir'], 'reais_vs_preditos.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    save_feature_importance_plot(model)


def save_feature_importance_plot(model):
    """Salva gráfico de importância das features via get_feature_names_out."""
    try:
        estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        preprocessor = estimator.named_steps['preprocessor']
        inner = estimator.named_steps['model']

        if not hasattr(inner, 'feature_importances_'):
            print("Modelo sem feature_importances_; gráfico de importância omitido.")
            return

        feature_names = preprocessor.get_feature_names_out()
        importances = inner.feature_importances_
        n = min(20, len(importances))
        indices = np.argsort(importances)[-n:]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top {n} — Importância das Features")
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importância")
        path = os.path.join(PATHS['plots_dir'], 'feature_importances.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Não foi possível plotar feature importances: {e}")


if __name__ == "__main__":
    print("Funções disponíveis neste módulo:")
    print("- save_results")
    print("- save_predictions")
    print("- save_report")
    print("- save_plots")
    print("- save_feature_importance_plot")
