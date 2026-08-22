# Previsão de emissão de N2O

Pipeline de regressão para estimar emissão de **N2O (t)** a partir do inventário brasileiro SEEG (**Coleção 13**, 1970–2024).

O teste é o futuro: treino até **2019** e teste em **2020–2024**. Features: hierarquia setorial + **defasagens por setor** (`emissao_lag1`, `emissao_lag2`, `delta_lag1`). A referência justa com lags é a **persistência de 1 passo** (copiar t−1).

Desenvolvido em **Python 3.11+** (`requirements.txt`). Rodou neste repositório em 3.14.

Resultados: [`docs/resultados.md`](docs/resultados.md). Escopo: [`docs/escopo.md`](docs/escopo.md).

## Dataset

Fonte: **SEEG** (Observatório do Clima) — [seeg.eco.br/dados](https://seeg.eco.br/dados/).

Coloque em `data/`:

```
data/Dados-nacionais-13.0.xlsx
```

(~136 MB, **não** vai ao Git). O pipeline filtra `N2O (t)` + `Emissão`, agrega UF/bioma ao **nacional** e grava o cache `data/n2o_nacional_longo.csv`.

| Interno | Origem Coleção 13 |
| --- | --- |
| `nivel_1` … `nivel_5` | Setor / categoria / subcategoria / detalhamento / recorte |
| `produto`, `atividade_economica`, `tipo_emissao` | Produto ou sistema, Atividade geral, Emissão/Remoção/Bunker |
| `ano`, `emissao` | melt das colunas 1970–2024 |
| `emissao_lag1`, `emissao_lag2`, `delta_lag1` | defasagens por setor (só passado) |

O alvo continua assimétrico (muitos zeros, cauda longa). O grid otimiza **MAE**.

## Estrutura

```
main.py                  treino, avaliação e relatório
config.py                caminhos, corte 2019, hiperparâmetros
data_seeg.py             leitura xlsx Coleção 13 → painel nacional
data_loading.py          carga + EDA
data_preprocessing.py    limpeza, lags, preprocessor, split
model_training.py        baselines (incl. 1 passo), RF, Ridge
results_saving.py        .joblib, CSV, relatório e gráficos
scripts/grafico.py       série anual real vs prevista
scripts/predict_only.py  inferência com modelo já treinado
data/Dados-nacionais-13.0.xlsx   local (não versionado)
resultados/              gerado na execução
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Como executar

```powershell
.\.venv\Scripts\python.exe -u main.py
.\.venv\Scripts\python.exe scripts\grafico.py
.\.venv\Scripts\python.exe scripts\predict_only.py
```

1. `main.py` — EDA, lags, split, baselines, GridSearch, relatório e modelo.
2. `scripts/grafico.py` — marca o corte 2019/2020.
3. `scripts/predict_only.py` — exige `resultados/modelo_n2o.joblib`.

## Como ler as métricas

Referência **justa com lags**: `persistencia_1passo` (y(t) ≈ y(t−1) observado). Também reportamos `persistencia` multi-ano (repetir 2019 em 2020–2024).

**Valor positivo** no Δ MAE (referência − modelo) = o modelo erra menos que a referência.

Números do teste 2020–2024 (experimento já rodado):

| Método | MAE (t) | RMSE (t) |
| --- | ---: | ---: |
| Persistência 1 passo | 54,23 | 463,64 |
| Ridge (CV) | 57,60 | 446,15 |
| Persistência multi-ano | 102,74 | 892,84 |
| Random Forest | 78,11 | 705,11 |

Δ MAE (1 passo − Ridge) = −3,37 t. Δ RMSE = +17,49 t (Ridge melhor em RMSE). Detalhe: [`docs/resultados.md`](docs/resultados.md).

## Limitações

- Avaliação cronológica (não split aleatório).
- Previsão de **um passo** no inventário anual: no teste, `lag1` de 2021 usa a emissão **observada** de 2020.
- Não é projeção multi-ano sem observar os anos intermediários.
- Agregação nacional (UF/bioma somados); `data/cidades/` fica fora deste recorte.
