# Resultados do experimento (seção 9.2)

Evidência gerada por `main.py` + `scripts/grafico.py` + `scripts/predict_only.py`. Fonte numérica: [`resultados/relatorio.txt`](../resultados/relatorio.txt). Página curta: [`resultados/leitura-secao9.txt`](../resultados/leitura-secao9.txt).

**Pergunta:** um modelo treinado só até 2010 prevê N2O (t) em 2011–2019 melhor do que repetir, por setor, o valor de 2010?

**Resposta medida:** não. A persistência ganha por 13,05 t de MAE (−5,9%). O método (split cronológico, MAE, baselines) está correto; o teto das features `ano` + categorias ficou documentado.

---

## 1. Recorte

| Item | Valor |
| --- | --- |
| Fonte | SEEG Coleção 8, 1970–2019 |
| Filtro | `gas == "N2O (t)"` → 26.400 linhas; 160 alvos nulos descartados; **26.240** usadas |
| Split | treino 1970–2010 (**21.488** linhas); teste 2011–2019 (**4.752** linhas) |
| Validação | `TimeSeriesSplit` (5 folds) no treino ordenado por ano |
| Scoring do grid | MAE (`neg_mean_absolute_error`) |
| Modelo escolhido (CV) | Random Forest (`max_depth=None`, `min_samples_split=2`, `n_estimators=100`); MAE de CV = 243,93 t |

Qualidade do alvo (N2O completo): 46,19% zeros; mediana 0,04 t; média 907 t; máximo 138.619 t. Os 10% maiores registros concentram **96,6%** da emissão.

---

## 2. Métricas no teste 2011–2019

Valores em toneladas de N2O. Tabela completa: [`resultados/metricas_comparacao.csv`](../resultados/metricas_comparacao.csv).

| Método | MAE | RMSE | MedAE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Persistência (referência)** | **219,73** | **1.390,27** | 0,21 | 0,967 |
| Random Forest (escolhido por CV) | 232,78 | 1.437,71 | 0,39 | 0,965 |
| Média por grupo | 654,48 | 3.668,49 | 0,75 | 0,772 |
| Ridge | 1.593,26 | 5.589,09 | 590,34 | 0,471 |
| Mediana / zero | ~1.343,73 | ~7.798,87 | ~0,39 | −0,03 |
| Média global | 1.877,36 | 7.700,73 | 810,41 | −0,005 |

Δ MAE (persistência − RF) = **−13,05 t**. Δ RMSE = **−47,44 t**. Valor negativo = o modelo erra mais que copiar 2010.

RF ganha de Ridge, da média por grupo e dos dummies. **Não ganha da persistência.** R² alto (~0,97) nos dois (RF e persistência) mede que o nível do setor é estável, não que o RF prevê a trajetória.

Grid até o fim: 12 combinações de RF × 5 folds e 4 valores de `alpha` no Ridge. Ver [`resultados/cv_random_forest.csv`](../resultados/cv_random_forest.csv) e [`resultados/cv_ridge.csv`](../resultados/cv_ridge.csv).

---

## 3. Interpretação

- **MedAE (0,39 t) << MAE (232,78 t):** o erro típico é pequeno; o MAE é a cauda. A linha mediana do inventário não é o problema.
- A previsão média do RF no teste fica **fixa em 1.205,59 t em todos os anos** 2011–2019, enquanto a emissão média real sobe de 1.259 t (2011) para 1.468 t (2019). Sem defasagem/tendência por setor, árvore e Ridge saturam no nível conhecido.
- O MAE anual cresce no horizonte (RF 113 t em 2011 → 404 t em 2019; persistência 90 t → 398 t). Os dois degradam; a persistência degrada um pouco menos.

---

## 4. Vieses (evidência no teste)

Não é só raciocínio: os números estão em [`resultados/vies_por_faixa.csv`](../resultados/vies_por_faixa.csv), [`vies_por_nivel1.csv`](../resultados/vies_por_nivel1.csv) e [`vies_por_ano.csv`](../resultados/vies_por_ano.csv).

**Assimetria / concentração**

- Teste: 37,2% zeros; média 1.343,7 t vs mediana 0,40 t.
- Os 10% maiores valores reais concentram **95,3%** da emissão e **93,6%** do erro absoluto do RF.

**Por faixa de emissão (MAE, t)**

| Faixa | n | RF | Persistência | % da emissão | % do erro RF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero | 1.767 | 1,10 | 0,10 | 0,0 | 0,2 |
| Baixo (positivo até a mediana) | 1.493 | 1,41 | 1,01 | 0,0 | 0,2 |
| Médio (mediana a p90) | 1.016 | 65,73 | 56,13 | 4,6 | 6,0 |
| Alto (> p90) | 476 | 2.175,04 | 2.070,24 | 95,3 | 93,6 |

**Por `nivel_1` (MAE, t)**

| Setor | n | RF | Persistência | % da emissão |
| --- | ---: | ---: | ---: | ---: |
| Mudança de Uso da Terra e Floresta | 108 | 2.426 | 2.537 | 15,3 |
| Agropecuária | 612 | 1.242 | 1.141 | 78,5 |
| Resíduos | 36 | 163 | 147 | 1,2 |
| Processos industriais | 144 | 41 | 11 | 0,3 |
| Energia | 3.852 | 19 | 17 | 4,6 |

MUT é o único setor em que o RF erra um pouco menos que a persistência. Agropecuária (78,5% da emissão) e a faixa alta decidem o MAE agregado — aí a persistência ganha.

---

## 5. Relevância (manual §9.2)

Não basta treinar um algoritmo. No teste futuro, o desempenho só seria operacionalmente relevante se o MAE fosse **menor** que o da persistência. Aqui a persistência é melhor. Conclusão para a banca: o experimento foi concluído; o resultado é negativo e interpretável (features sem série temporal por setor).

Integração: `scripts/predict_only.py` carrega `resultados/modelo_n2o.joblib` (arquivo local, fora do Git por tamanho; gerar com `main.py`).
