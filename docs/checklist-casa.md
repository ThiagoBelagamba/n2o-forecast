# Checklist — frente de dados e modelo (seção 9)

Status com **Coleção 13 + lags** (treino já executado nesta máquina).

| Campo | Valor |
| --- | --- |
| Nome | Thiago Belagamba Bueno |
| Função | Análise de dados e mecanismo preditivo (seção 9) |
| Repositório | https://github.com/ThiagoBelagamba/n2o-forecast |

## Status

- [x] `Dados-nacionais-13.0.xlsx` em `data/` + cache `n2o_nacional_longo.csv`
- [x] Lags por setor + split 2019 / 2020–2024
- [x] Baselines: persistência multi-ano e **1 passo**
- [x] Grid RF + Ridge até o fim; relatório e gráficos
- [x] `docs/resultados.md` com números do teste

Resultado (teste 2020–2024): **persistência 1 passo MAE 54,23 t; Ridge 57,60 t; RF 78,11 t; persistência multi-ano 102,74 t**. Ridge ganha da multi-ano e do RMSE da 1 passo; perde ~3 t de MAE para copiar t−1.

## Reproduzir

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe -u main.py
.\.venv\Scripts\python.exe scripts\grafico.py
.\.venv\Scripts\python.exe scripts\predict_only.py
```

O xlsx (~136 MB) e o `.joblib` ficam locais. Relatório e PNGs vão ao Git.
