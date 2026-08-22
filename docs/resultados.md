# Resultados do experimento (seção 9.2) — Coleção 13 + lags

Evidência gerada por `main.py` + `scripts/grafico.py` + `scripts/predict_only.py`. Fonte: [`resultados/relatorio.txt`](../resultados/relatorio.txt). Página curta: [`resultados/leitura-secao9.txt`](../resultados/leitura-secao9.txt).

**Pergunta:** com o histórico até t−1 (lags por setor), um modelo prevê N2O (t) no ano t (teste 2020–2024) melhor do que copiar t−1?

**Resposta medida:** em MAE agregado, **não** — a persistência de 1 passo ganha por 3,37 t (−6,2%). O Ridge (escolhido por CV) **supera** a persistência multi-ano (repetir 2019) por 45,15 t de MAE e **ganha em RMSE** frente à de 1 passo. Em 2020 e 2021 o Ridge também erra menos que copiar t−1.

---

## 1. Recorte

| Item | Valor |
| --- | --- |
| Fonte | SEEG Coleção 13 (`Dados-nacionais-13.0.xlsx`), agregado nacional |
| Filtro | `N2O (t)` + `Emissão` → painel 42.350 linhas; com lags **41.580** |
| Split | treino ≤ 2019 (**37.730**); teste 2020–2024 (**3.850**) |
| Features | `ano` + categorias + `emissao_lag1`, `emissao_lag2`, `delta_lag1` |
| Validação | `TimeSeriesSplit` (5 folds) |
| Modelo escolhido (CV) | **Ridge** (`alpha=0.1`); MAE de CV = 41,13 t |

---

## 2. Métricas no teste 2020–2024

| Método | MAE | RMSE | MedAE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Persistência 1 passo (referência justa)** | **54,23** | 463,64 | 0,08 | 0,995 |
| Ridge (escolhido por CV) | 57,60 | **446,15** | 2,94 | 0,996 |
| Persistência multi-ano (repetir 2019) | 102,74 | 892,84 | 0,16 | 0,983 |
| Random Forest | 78,11 | 705,11 | 0,16 | 0,989 |
| Média por grupo | 410,12 | 3.168,51 | 0,51 | 0,786 |

- Δ MAE (1 passo − Ridge) = **−3,37 t**
- Δ RMSE (1 passo − Ridge) = **+17,49 t** (Ridge melhor em RMSE)
- Δ MAE (multi-ano − Ridge) = **+45,15 t** (Ridge melhor que copiar 2019)

---

## 3. Interpretação

- Com lags, o teto correto é **copiar t−1**, não copiar 2019.
- O Ridge quase empatou com esse teto em MAE e o **superou em RMSE** (cauda).
- Por ano: Ridge melhor que 1 passo em **2020** e **2021**; empate em 2023; pior em 2022 e 2024.
- Top 10% das linhas do teste: 97% da emissão e 85,5% do erro do modelo.

---

## 4. Relevância (§9.2)

Não basta treinar um algoritmo. Aqui o desempenho relevante se mede contra a persistência de 1 passo. O Ridge não a supera em MAE agregado, mas **melhora a persistência multi-ano** e o RMSE — evidência de que as defasagens adicionam sinal frente a “congelar 2019”.

Limitação: previsão de **um passo** (inventário anual). Integração: `scripts/predict_only.py` + `modelo_n2o.joblib` (local).
