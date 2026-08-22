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
| Modelo base | LightGBM Tweedie (p=1.7), Ridge, RF |
| Reconciliação | BottomUp + MinTrace(ols) + MinTrace(mint_shrink); clip ≥ 0 |

`persistencia_1passo` (copiar t−1 observado) **não** é a referência deste protocolo.

---

## Métricas no teste (séries folha)

| Método | MAE | RMSE |
| --- | ---: | ---: |
| Persistência multi-ano | 102,74 | 892,84 |
| **Ridge / MinTrace(ols)** | **83,85** | **704,53** |
| Ridge / MinTrace(mint_shrink) | 86,86 | 663,18 |
| Ridge (base) | 89,90 | 707,21 |
| RF / MinTrace(mint_shrink) | 97,07 | 732,79 |
| LGBM Tweedie / MinTrace(mint_shrink) | 126,27 | 1.179,25 |

LightGBM Tweedie (p=1.7) foi selecionado por CV no treino; no teste multi-ano o **Ridge reconciliado** venceu em MAE. O Tweedie permanece no pipeline (perda alinhada a zeros + cauda); a entrega operacional usa o melhor MAE no protocolo cego.

---

## Projeção 2025–2026

1. Refit até 2024  
2. Forecast recursivo h=2  
3. Reconciliação + clip ≥ 0  
4. CSV por setor + gráfico do total nacional  

Aviso no arquivo: **cenário, não inventário oficial SEEG**.

---

## Limitações

- Cascata de erro na projeção recursiva 2026 (usa o previsto de 2025).  
- NNMinT com programação quadrática ficou lento demais (~800 séries); usamos MinT clássico + clip.  
- Sem covariates externas (fertilizante, rebanho).  
