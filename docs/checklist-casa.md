# Checklist — seção 9 (multi-ano)

- [x] Coleção 13 + painel `unique_id`/`ds`/`y`
- [x] Previsão direta h=5 sem lag observado no teste
- [x] LightGBM Tweedie + Ridge + RF (MLForecast)
- [x] Reconciliação BottomUp / MinTrace + clip ≥ 0
- [x] Métricas vs persistência multi-ano
- [x] Projeção 2025–2026 + gráficos
- [x] Docs atualizados

Reproduzir: `.\.venv\Scripts\python.exe -u main.py`

Melhor no teste cego: **Ridge/MinTrace(ols)** (MAE 83,85 vs persistência 102,74).
