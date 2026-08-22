# Previsão de emissão de N2O

Pipeline de **previsão multi-ano** de **N2O (t)** (SEEG Coleção 13, 1970–2024): LightGBM Tweedie + Ridge/RF, previsão **direta** sem lag observado no teste, reconciliação hierárquica **MinT**, cenário **2025–2026**.

Python **3.11+**. Resultados: [`docs/resultados.md`](docs/resultados.md). Escopo: [`docs/escopo.md`](docs/escopo.md).

## Ideia central

O teste **não** usa a emissão real de 2020 para prever 2021. Treino até 2019; horizonte 2020–2024 cego. Referência justa: **persistência multi-ano** (copiar 2019). Isso é o protocolo certo para projetar 2025–2026.

No experimento: **Ridge + MinTrace** MAE 83,85 t vs persistência 102,74 t (ganho +18,9 t).

## Dataset

```
data/Dados-nacionais-13.0.xlsx   # local (~136 MB), fora do Git
```

Filtro `N2O (t)` + `Emissão`, agregação nacional, cache `data/n2o_nacional_longo.csv`.

## Estrutura

```
main.py                 orquestra multi-ano + MinT + projecao
forecasting.py          MLForecast + LightGBM Tweedie / Ridge / RF
reconciliation.py       hierarquia + BottomUp / MinTrace
data_seeg.py            leitura Colecao 13
data_preprocessing.py   painel unique_id / ds / y
results_multiyear.py    relatorio, metricas, graficos 2025-2026
```

## Setup e execução

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe -u main.py
```

Saídas: `resultados/relatorio.txt`, `metricas_comparacao.csv`, `comparacao_real_vs_previsto.csv` (SEEG vs modelo em 2020–2024), `projecao_2025_2026.csv`, gráficos.

## Limitações

Cenario 2025–2026 não substitui o SEEG. NNMinT QP foi substituído por MinT + clip ≥ 0 por tempo de computação.
