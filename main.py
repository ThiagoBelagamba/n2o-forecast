# main.py
import traceback

from config import MODEL_CONFIG, PATHS
from data_loading import load_and_analyze_data, perform_eda
from data_preprocessing import create_preprocessor, preprocess_data, split_data_chronological
from model_training import create_baselines, evaluate_model, train_models
from results_saving import save_results


def _print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | "
          f"R2: {metrics['R2']:.4f} | MedAE: {metrics['MedAE']:.4f}")


def main():
    print("Iniciando pipeline de modelagem...")

    try:
        print("\n1. Carregando dados e realizando análise exploratória...")
        df = load_and_analyze_data(PATHS['data'])
        df = perform_eda(df)

        print("\n2. Pré-processando dados...")
        X, y, num_cols, cat_cols, data_quality = preprocess_data(df)
        preprocessor = create_preprocessor(num_cols, cat_cols)
        X_train, X_test, y_train, y_test = split_data_chronological(X, y)

        print("\n2.1 Criando baselines para comparação...")
        baseline_results = create_baselines(X_train, y_train, X_test, y_test)
        print("\nDesempenho dos baselines no teste:")
        for name, metrics in baseline_results.items():
            _print_metrics(name, metrics)

        print("\n3. Treinando modelos (Random Forest e Ridge)...")
        searches = train_models(X_train, y_train, preprocessor)

        model_metrics = {}
        preds = {}
        for name, grid in searches.items():
            print(f"\nAvaliando {name} no teste...")
            y_hat, metrics = evaluate_model(grid.best_estimator_, X_test, y_test)
            model_metrics[name] = metrics
            preds[name] = y_hat

        best_name = max(searches, key=lambda n: searches[n].best_score_)
        best_grid = searches[best_name]
        y_pred = preds[best_name]
        print(f"\nSelecionado por MAE de CV: {best_name}")

        print("\n4. Salvando resultados...")
        save_results(
            best_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            y_pred,
            num_cols,
            cat_cols,
            baseline_results=baseline_results,
            data_quality=data_quality,
            model_metrics=model_metrics,
            best_model_name=best_name,
        )

        print(f"\nProcesso concluído. Relatório: {PATHS['report']}")
        print(
            f"Split: treino <= {MODEL_CONFIG['split_year']} | "
            f"teste > {MODEL_CONFIG['split_year']}"
        )

    except Exception:
        print("\nOcorreu um erro durante a execução:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
