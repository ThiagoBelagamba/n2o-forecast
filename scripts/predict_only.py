# scripts/predict_only.py
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np

from config import MODEL_CONFIG, PATHS, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import preprocess_data


def load_model_and_predict():
    print("\nCarregando dados...")
    df = load_and_analyze_data(PATHS['data'])

    print("\nPré-processando dados...")
    X, y, num_cols, cat_cols, _quality = preprocess_data(df)

    print("\nCarregando modelo treinado...")
    if not os.path.exists(PATHS['model']):
        print(f"Erro: modelo não encontrado em {PATHS['model']}")
        print("Execute main.py primeiro para treinar e salvar o modelo.")
        sys.exit(1)

    model = joblib.load(PATHS['model'])

    print("\nFazendo previsões...")
    predictions = model.predict(X)

    print("\nSalvando resultados...")
    ensure_dirs()
    df_results = X.copy()
    df_results[MODEL_CONFIG['target_col']] = np.asarray(y)
    df_results['emissao_predita'] = np.round(predictions, 8)
    corte = MODEL_CONFIG['split_year']
    df_results['conjunto'] = np.where(
        df_results['ano'] <= corte, 'treino', 'teste'
    )

    df_results.to_csv(PATHS['predictions_new'], index=False)
    print(f"\nPrevisões salvas em: {PATHS['predictions_new']}")


if __name__ == "__main__":
    print("Iniciando predições usando modelo pré-treinado...")
    load_model_and_predict()
