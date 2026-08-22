# scripts/predict_only.py — projecao com modelo multi-ano ja treinado
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import warnings

from config import MODEL_CONFIG, PATHS, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from reconciliation import build_hierarchy, leaf_forecasts_only, reconcile_forecasts


def main():
    print("Carregando painel...")
    df = load_and_analyze_data(PATHS['data'])
    panel, _ = to_nixtla_panel(df)
    Y_df, S_df, tags, _ = build_hierarchy(panel)
    Y_full = Y_df[Y_df['ds'].dt.year <= 2024].copy()

    if not os.path.exists(PATHS['model']):
        print(f"Modelo nao encontrado: {PATHS['model']}")
        print("Execute main.py primeiro.")
        sys.exit(1)

    print("Carregando MLForecast...")
    fcst = joblib.load(PATHS['model'])
    h = len(MODEL_CONFIG['forecast_years'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Modelo salvo e o de avaliacao direta; para cenario use main.py.
        # Aqui tentamos predict(h) se o objeto permitir.
        try:
            Y_hat = fcst.predict(h=h)
        except Exception as e:
            print(f"Predict falhou ({e}). Rode main.py para gerar projecao_2025_2026.csv")
            sys.exit(1)
        Y_rec, _ = reconcile_forecasts(Y_hat, Y_df, S_df, tags, train_end_year=2024)

    cols = [c for c in Y_rec.columns if 'MinTrace' in c or 'BottomUp' in c or c == 'LGBM_Tweedie']
    col = cols[0]
    leaf = leaf_forecasts_only(Y_rec, tags, col)
    leaf['ano'] = leaf['ds'].dt.year
    ensure_dirs()
    out = PATHS['predictions_new']
    leaf.to_csv(out, index=False)
    print(f"Salvo: {out}")


if __name__ == '__main__':
    main()
