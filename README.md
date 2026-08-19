# Previsão de emissão de N2O

Pipeline de regressão para estimar emissão de **N2O (t)** a partir do inventário brasileiro de gases de efeito estufa (1970–2019), no formato SEEG.

O objetivo **não** é interpolar setores já vistos. O teste é o futuro: treino em 1970–2010 e teste em 2011–2019. Assim o mesmo setor não aparece nos dois lados do split só porque o ano mudou.

Desenvolvido em **Python 3.11+** (`requirements.txt`). Rodou neste repositório em 3.14.

Resultados do teste 2011–2019 (números e vieses): [`docs/resultados.md`](docs/resultados.md). Escopo acadêmico: [`docs/escopo.md`](docs/escopo.md).

## Dataset

Fonte: **SEEG** (Sistema de Estimativas de Emissões e Remoções de Gases de Efeito Estufa), iniciativa do [Observatório do Clima](https://www.oc.eco.br/).

- Site e download: [seeg.eco.br/dados](https://seeg.eco.br/dados/)
- Plataforma interativa: [plataforma.seeg.eco.br](https://plataforma.seeg.eco.br/)

Este CSV cobre **1970–2019** (sem UF/município), típico da **Coleção 8**. A planilha oficial de hoje é a Coleção 13 (série até 2024) e vem com mais colunas; o schema daqui é a versão nacional “longa” (`ano` em linhas, não em colunas).

Citação sugerida pelo SEEG: *“Fonte: SEEG – Sistema de Estimativa de Emissões e Remoções de Gases de Efeito Estufa, Observatório do Clima – seeg.eco.br”*.

O CSV **não está no Git** (~82 MB). Coloque-o em:

```
data/emissao_gases.csv
```

O arquivo original contém todos os gases do inventário. O pipeline filtra `gas == "N2O (t)"` (26.400 linhas: 528 setores × 50 anos). 160 linhas sem alvo são descartadas.

| Coluna | Papel |
| --- | --- |
| `ano` | Única feature numérica (1970–2019) |
| `nivel_1` … `nivel_6` | Hierarquia setorial |
| `tipo_emissao` | Tipo da emissão |
| `atividade_economica` | Atividade (há ~0,6% de ausência no N2O) |
| `produto` | Produto (há ~54% de ausência no N2O; imputado como `missing`) |
| `gas` | Tipo de gás (filtrado para N2O) |
| `emissao` | Alvo, em toneladas |

O alvo é assimétrico: cerca de 46% zeros, mediana ≈ 0,04 t, média ≈ 907 t, máximo ≈ 138.619 t. Por isso o grid otimiza **MAE**, não R².

## Estrutura

```
main.py                  treino, avaliação e relatório
config.py                caminhos, corte temporal e hiperparâmetros
data_loading.py          carga, filtro N2O e EDA (gráficos em disco)
data_preprocessing.py    limpeza, preprocessor sklearn, split cronológico
model_training.py        baselines, Random Forest, Ridge, métricas
results_saving.py        modelo .joblib, CSV, relatório e gráficos
scripts/grafico.py       série anual real vs prevista
scripts/predict_only.py  inferência com modelo já treinado
data/emissao_gases.csv   dataset local (não versionado)
resultados/              gerado na execução
```

Hiperparâmetros e o ano de corte (`split_year = 2010`) ficam em `config.py`.

## Setup

Não copie a pasta `.venv` do outro computador. Recrie o ambiente.

Python **3.11 ou mais novo**. Depois:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Linux/macOS: `python3 -m venv .venv` e `.venv/bin/pip install -r requirements.txt`.

O CSV **não está no Git**. Copie `emissao_gases.csv` para `data/emissao_gases.csv` antes de rodar. Sem esse arquivo o `main.py` para com erro explícito.

## Como executar

Na **raiz** do projeto:

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe scripts\grafico.py
.\.venv\Scripts\python.exe scripts\predict_only.py
```

1. `main.py` — EDA, split, baselines, GridSearch (Random Forest e Ridge com `TimeSeriesSplit`), relatório e modelo.
2. `scripts/grafico.py` — exige o CSV de predições gerado pelo treino; marca o corte 2010/2011.
3. `scripts/predict_only.py` — exige `resultados/modelo_n2o.joblib`; grava um CSV novo, sem sobrescrever o do treino.

O GridSearch do Random Forest é a etapa lenta (12 combinações × 5 folds). O paralelismo fica em `config.py` (`grid_n_jobs=4`, `rf_n_jobs=3`). Em um PC comum pode levar dezenas de minutos.

## Saídas (`resultados/`)

| Arquivo | Conteúdo |
| --- | --- |
| `relatorio.txt` | Qualidade dos dados, métricas no teste, RF vs Ridge vs persistência, vieses, §9.2 |
| `leitura-secao9.txt` | Página curta da evidência |
| `metricas_comparacao.csv` | Tabela RF / Ridge / baselines |
| `cv_random_forest.csv`, `cv_ridge.csv` | Grid completo (12 + 4 combinações) |
| `vies_por_faixa.csv`, `vies_por_nivel1.csv`, `vies_por_ano.csv` | Erro no teste por recorte |
| `modelo_n2o.joblib` | Melhor modelo (RF). **Local**, ~103 MB — o GitHub recusa; gere com `main.py` |
| `emissao_n2o_com_predicoes.csv` | Predições de treino e teste. Local (não vai ao Git) |
| `graficos/` | EDA, reais vs preditos, importância, MAE por faixa/setor/ano, série anual |

## Como ler as métricas

O modelo escolhido é o de **menor MAE na validação cruzada temporal**. No teste reportamos MAE, RMSE, MedAE e R².

| Métrica | Interpretação |
| --- | --- |
| MAE | Erro absoluto médio, em toneladas |
| RMSE | Penaliza erros nos grandes emissores |
| MedAE | Erro típico, pouco afetado por outliers |
| R² | Fração da variância no teste (2011–2019) |

Baselines no mesmo teste:

- média, mediana e zero (`DummyRegressor`);
- **média por grupo** — média histórica do setor no treino;
- **persistência** — último valor do setor em 2010. Este é o benchmark correto para previsão.

O relatório traz Δ MAE e Δ RMSE contra a persistência. **Valor positivo = o modelo erra menos que repetir 2010.**

No teste 2011–2019 (experimento já rodado):

| Método | MAE (t) | RMSE (t) |
| --- | ---: | ---: |
| Persistência | 219,73 | 1.390,27 |
| Random Forest | 232,78 | 1.437,71 |
| Ridge | 1.593,26 | 5.589,09 |

Δ MAE (persistência − RF) = −13,05 t. A persistência ganha; o RF só ganha de Ridge e dos dummies. Detalhe e vieses: [`docs/resultados.md`](docs/resultados.md).

## Limitações

Um split aleatório 80/20 neste painel (528 grupos × 50 anos) coloca o mesmo setor em treino e teste. O R² fica alto porque o modelo memoriza o nível do setor, não porque prevê o futuro. Por isso a avaliação é cronológica.

As features são só `ano` + categorias. Random Forest e Ridge **não extrapolam tendência** além de 2010: a previsão média no teste ficou constante (1.205,59 t) enquanto a emissão real subiu. Perder para a persistência é o resultado medido, não um bug do código. Superar esse teto exigiria defasagens e tendência por setor — fora do escopo deste pipeline.
