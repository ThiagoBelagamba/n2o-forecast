# Escopo do projeto

**Título:** Previsão multi-ano de N2O (SEEG Coleção 13) com Tweedie, MinT e cenário 2025–2026

| Campo | Valor |
| --- | --- |
| Curso | Ciência da Computação |
| Disciplina | Tendências em Ciência da Computação |
| Professor | Prof. Dr. Renato Gonçalves Rocha |
| Integrante (esta frente) | Thiago Belagamba Bueno |
| Repositório | https://github.com/ThiagoBelagamba/n2o-forecast |

Números: [`docs/resultados.md`](resultados.md).

---

## Pergunta de pesquisa

> Com o SEEG nacional até o ano de corte, um modelo (Tweedie/boosting ou Ridge) com previsão direta multi-passo e reconciliação hierárquica prevê N2O (t) nos anos seguintes **sem observar o inventário intermediário** melhor do que copiar o último ano do treino?

## Método

| Decisão | Escolha |
| --- | --- |
| Fonte | Coleção 13, 1970–2024, nacional |
| Split | treino ≤ 2019, teste 2020–2024 |
| Protocolo | previsão direta h=5 (anti-vazamento) |
| Modelos | LightGBM Tweedie (sintonizado), Ridge, RF via MLForecast; NHITS experimental se LGBM não bater Ridge |
| Reconciliação | BottomUp + MinTrace (ols / mint_shrink) + clip ≥ 0 |
| Referência | persistência multi-ano |
| Entrega | projeção recursiva 2025–2026 |

## Fora do escopo

Covariates IBGE; município como alvo principal; novos modelos pontuais após platô MAE.

**UI/API** (`api/`, `web/`): galeria mínima Nest + Next lendo artefatos de `resultados/` — entregável de visualização, não app de produção.

**NHITS** entra como extensão experimental (`forecasting_nhits.py`, `scripts/run_nhits_benchmark.py`) somente quando LGBM não supera Ridge no teste cego; não substitui o modelo operacional até provar ganho.

**Conformal Prediction** (`forecasting_conformal.py`, `scripts/run_conformal.py`): intervalos de incerteza sobre Ridge reconciliado — complementa a entrega pontual.

**DTW clustering** (`forecasting_dtw.py`, `scripts/run_dtw_cluster_benchmark.py`): piloto experimental de modelos por cluster; não superou Ridge global no teste medido.

**Fechamento TCC** (`results_entrega.py`, `scripts/export_entrega.py`, `scripts/export_slides.py`): agregações por setor, cenário com IC, tabela de experimentos, síntese JSON e export para slides.

## Resultado (resumo)

Ridge/MinTrace(ols) MAE 83,85 t vs persistência multi-ano 102,74 t no teste cego 2020–2024. Platô MAE confirmado; conformal ~91%/95% nas folhas. Cenário 2025–2026 com IC em `cenario_2025_2026_resumo.csv` e `resultados/export/slides.md`.
