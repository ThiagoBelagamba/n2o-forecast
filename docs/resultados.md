# Resultados — previsão multi-ano (Coleção 13 + Tweedie + MinT)

Fonte: [`resultados/relatorio.txt`](../resultados/relatorio.txt). Projeção: [`resultados/projecao_2025_2026.csv`](../resultados/projecao_2025_2026.csv).

**Pergunta:** com dados até 2019, prever 2020–2024 **sem** ver a emissão observada nos anos intermediários (protocolo adequado a 2025–2026)?

**Resposta medida:** sim — **Ridge + MinTrace(ols)** MAE **83,85 t** vs persistência multi-ano **102,74 t** (Δ MAE = **+18,89 t**). O modelo erra menos que copiar 2019 no horizonte cego.

---

## Protocolo (anti-vazamento)

| Item | Valor |
| --- | --- |
| Treino | ≤ 2019 |
| Teste | 2020–2024, previsão **direta** h=5 (MLForecast `max_horizon`) |
| Lags no teste | geridos pelo MLForecast a partir do fim do treino — **sem** y observado de 2020+ |
| Referência justa | persistência multi-ano (copiar 2019) |
| Modelo base | LightGBM Tweedie (sintonizado), Ridge, RF; lags `[1,2,3,5]` + `RollingMean(3,5)` |
| Reconciliação | BottomUp + MinTrace(ols) + MinTrace(mint_shrink); clip ≥ 0 |

`persistencia_1passo` (copiar t−1 observado) **não** é a referência deste protocolo.

---

## Métricas no teste (séries folha)

| Método | MAE | RMSE |
| --- | ---: | ---: |
| Persistência multi-ano | 102,74 | 892,84 |
| **Ridge / MinTrace(ols)** | **83,85** | **704,53** |
| Ridge / MinTrace(mint_shrink) | 88,03 | 673,50 |
| Ridge (base) | 89,90 | 707,21 |
| LGBM Tweedie / MinTrace(mint_shrink) | 108,25 | 833,60 |
| LGBM Tweedie (base, sintonizado) | 101,37 | 800,32 |
| RF / MinTrace(mint_shrink) | 97,07 | 732,79 |

LightGBM foi sintonizado com `scripts/tune_lgbm.py` (RandomizedSearch + refino, CV direta h=5 em 2015–2019). Melhor config em `resultados/lgbm_best_params.json` (p=1.1, `max_depth=5`, `min_data_in_leaf=25`, L1/L2). No teste cego multi-ano o **Ridge reconciliado** permanece melhor em MAE; LGBM melhorou de ~126 t para ~101 t (base) mas não superou a barreira **83,85 t**.

---

## Sintonia LGBM (Fase 1)

| Artefato | Conteúdo |
| --- | --- |
| `resultados/lgbm_best_params.json` | Hiperparâmetros vencedores (CV MAE ≈ 105,5 t) |
| `resultados/cv_lgbm_grid.csv` | Malha RandomizedSearch + refino top-5 |

Features MLForecast: lags `[1, 2, 3, 5]` + `RollingMean(3)` e `RollingMean(5)` no lag 1.

---

## NHITS (Fase 2 — extensão experimental)

**Gate:** acionado porque LGBM reconciliado MAE ≥ 83,85 t. Benchmark: `scripts/run_nhits_benchmark.py`.

| Método | MAE | RMSE |
| --- | ---: | ---: |
| NHITS (folha, sem MinT) | 85,67 | — |
| NHITS / MinTrace(ols) | 85,67 | 695,35 |

NHITS **não** superou Ridge/MinTrace(ols) (Δ ≈ −1,82 t). Projeção 2025–2026 permanece com Ridge/LGBM do pipeline principal. Artefatos: `resultados/nhits_benchmark.json`, `resultados/metricas_nhits.csv`.

---

## Ensemble (Ridge + LGBM / Ridge + NHITS)

Blend **antes** da reconciliação MinT. Pesos sintonizados por CV folha (2015–2019, h=5).

| Método | MAE teste cego | vs Ridge/MinTrace(ols) |
| --- | ---: | --- |
| **Ridge / MinTrace(ols)** | **83,85** | — |
| Ridge / MinTrace(mint_shrink) | 88,03 | referência atual no `main.py` |
| Ensemble Ridge+LGBM (base, w=0,7/0,3) | 94,60 | pior |
| Ensemble Ridge+LGBM / MinTrace(ols) | 102,32 | pior |
| Ensemble Ridge+NHITS (w=0,3/0,7) | 88,47 | pior (−4,6 t) |

O ensemble Ridge+LGBM melhora levemente o Ridge **sem** reconciliar (95,35 → 94,60 t), mas **não** supera Ridge+MinT. Ridge+NHITS fica entre NHITS isolado e Ridge, ainda longe da barreira.

Scripts: `scripts/tune_ensemble.py`, `scripts/run_ensemble_benchmark.py`. Pesos: `resultados/ensemble_weights.json`.

---

## Conformal Prediction (incerteza)

Split conformal simétrico sobre **Ridge/MinTrace(ols)**; calibração 2015–2019 (treino ≤ 2014), teste cego 2020–2024.

| Nível | Cobertura 90% | Cobertura 95% | Largura média IC 90% |
| --- | ---: | ---: | ---: |
| Folhas (todas) | **91,4%** | **94,8%** | ~118 t |
| Folhas (y>0 no treino) | 91,3% | 94,8% | ~119 t |

Intervalos no nível **folha** ficam próximos da cobertura nominal. No total nacional a calibração com poucos pontos (5 anos) ainda sub-cobre — ver gráfico `conformal_nacional.png`.

Artefatos: `resultados/conformal_coverage.json`, `conformal_intervals.csv`, `conformal_projecao_2025_2026.csv`. Script: `scripts/run_conformal.py`.

---

## Piloto DTW + modelos por cluster

K-Means com métrica **DTW** (K=3 por silhouette) nas 770 folhas; Ridge/LGBM treinados por cluster vs Ridge global.

| Método | MAE folha teste |
| --- | ---: |
| Ridge global / MinTrace(ols) | **89,82** |
| Ridge por cluster DTW | 95,51 |
| LGBM por cluster DTW | 115,30 |

O piloto **não** superou o Ridge global nem a barreira 83,85 t. Clusters em `resultados/dtw_clusters.csv`; benchmark em `dtw_benchmark.json`. Script: `scripts/run_dtw_cluster_benchmark.py`.

---

## Projeção 2025–2026

1. Refit até 2024  
2. Forecast recursivo h=2  
3. Reconciliação + clip ≥ 0  
4. CSV por setor + gráfico do total nacional  

Aviso no arquivo: **cenário, não inventário oficial SEEG**.

---

## Tempo de inferência (MinT clássico + clip)

Medido com `scripts/benchmark_inference.py` sobre o `modelo_n2o.joblib` (h=5 + BottomUp/MinTrace + clip ≥ 0):

| Etapa | Tempo |
| --- | ---: |
| Predict horizonte | ~0,2 s |
| Reconcile MinT + clip | ~1,0 s |
| **Núcleo de inferência** | **~1,2 s** (~0,02 min) |
| + load do joblib | ~1,4 s |

Arquivo: [`resultados/tempo_inferencia.json`](../resultados/tempo_inferencia.json).

**Margem:** após abandonar o NNMinT (QP), o tempo total de inferência multi-ano cai para a ordem de **poucos segundos** (**&lt; 1 minuto**).

---

## Limitações

- Cascata de erro na projeção recursiva 2026 (usa o previsto de 2025).  
- NNMinT com programação quadrática ficou lento demais (~800 séries); usamos MinT clássico + clip (inferência ~1,2 s).  
- Sem covariates externas (fertilizante, rebanho).  

---

## Conclusão e platô MAE

O erro pontual **estabilizou** em torno da barreira **83,85 t** (Ridge/MinTrace ols). Experimentos posteriores — LGBM tunado (~101 t), NHITS (~86 t), ensemble Ridge+LGBM/NHITS (~88–95 t), DTW por cluster (~95 t) — **não superaram** Ridge de forma consistente.

**Próximo passo metodológico:** quantificar **incerteza** (Conformal Prediction), não novo tuning pontual. Cobertura folha ~**91% / 95%** valida os intervalos no teste cego.

**Entrega operacional:** cenário 2025–2026 com ponto + IC agregado (`cenario_2025_2026_resumo.csv`), UI Nest/Next, export para slides (`resultados/export/slides.md`).

Pipeline de fechamento:

```powershell
.\.venv\Scripts\python.exe -u main.py
.\.venv\Scripts\python.exe -u scripts\run_conformal.py
.\.venv\Scripts\python.exe -u scripts\export_entrega.py
.\.venv\Scripts\python.exe -u scripts\export_slides.py
```

Artefatos: `mae_por_nivel1.csv`, `conformal_por_nivel1.csv`, `tabela_experimentos.csv`, `sintese_entrega.json`, `leitura-secao9.txt`.
