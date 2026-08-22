# Escopo do projeto

**Título provisório:** Previsão cronológica de emissão de N2O (SEEG Coleção 13, 1970–2024) com defasagens por setor

Preencher com a equipe (demais nomes):

| Campo | Valor |
| --- | --- |
| Instituição | |
| Curso | Ciência da Computação |
| Turno | Noite |
| Semestre | 8º |
| Disciplina | Tendências em Ciência da Computação |
| Professor | Prof. Dr. Renato Gonçalves Rocha |
| Integrante (esta frente) | Thiago Belagamba Bueno |
| Repositório técnico | https://github.com/ThiagoBelagamba/n2o-forecast |

Este documento delimita **para quem**, **o quê**, **como** e **quem faz o quê**. A frente de **análise de dados e mecanismo preditivo** (Manual de Funções, seção 9) está **executada** com Coleção 13 + lags. Números: [`docs/resultados.md`](resultados.md).

---

## 1. Para quem

O recorte **não é um aplicativo**. O entregável é um **mecanismo preditivo + avaliação honesta**.

| Interessado | Uso esperado |
| --- | --- |
| Engenheiros ambientais / analistas de GEE | Projetar N2O por setor no próximo ano do inventário; comparar com “copiar t−1” |
| Gestores / ESG | Insumo de curto prazo (1 passo), não substituto do inventário oficial |
| Pesquisadores | Reproduzir split cronológico com lags |
| Banca | Pergunta, método, métricas e limitações |

---

## 2. Problema

Split aleatório neste painel coloca o mesmo setor em treino e teste e infla o R². O problema é prever o ano futuro **sem vazamento temporal**, usando o histórico até t−1.

---

## 3. Pergunta de pesquisa

> Com o SEEG Coleção 13 (nacional, 1970–2024) e defasagens por setor, um modelo treinado até 2019 prevê N2O (t) em 2020–2024 melhor do que copiar, para cada setor, o valor observado em t−1?

---

## 4. Objetivos

**Geral:** mecanismo preditivo reproduzível de N2O (t) com validação cronológica e lags.

**Específicos:**

1. Usar a Coleção 13 (xlsx largo → painel nacional longo).
2. Split treino ≤ 2019 / teste 2020–2024 + `TimeSeriesSplit`.
3. Baselines: dummies, média por grupo, persistência multi-ano e **persistência 1 passo**.
4. Comparar Ridge e Random Forest (MAE), com `emissao_lag1/lag2/delta`.
5. Documentar se o ganho frente a copiar t−1 é relevante.

---

## 5. O que entra no escopo

- SEEG Coleção 13 (`Dados-nacionais-13.0.xlsx`), filtro N2O + Emissão, agregação nacional.
- EDA, limpeza, imputação, **lags por setor**.
- Pipeline Python (`main.py`, `data_seeg.py`, módulos, `requirements.txt`).
- Relatório, gráficos, `.joblib`, `predict_only.py`.

## 6. O que fica fora

- App / frontend / API / BD transacional.
- Desagregação operacional por UF/município (`data/cidades/` é outro recorte).
- Outros gases; projeção multi-ano sem observar o intermediário.
- Coleção 8 como fonte principal (substituída).

---

## 7. Como (método)

| Decisão | Escolha | Motivo |
| --- | --- | --- |
| Fonte | Coleção 13, 1970–2024 | Série oficial atual |
| Features | ano + categorias + lags | Passado por setor |
| Split | ≤ 2019 / ≥ 2020 | Anos novos vs Coleção 8 |
| Referência justa | persistência 1 passo | Modelo vê lag1 |
| Referência extra | persistência multi-ano | Comparável ao experimento antigo |
| Métodos | Dummy, grupo, 2 persistências, Ridge, RF | Comparar, não só um algoritmo |

---

## 8. Função principal (concluída)

| Campo | Registro |
| --- | --- |
| Nome | Thiago Belagamba Bueno |
| Função | Análise de dados e mecanismo preditivo (§9) |
| Secundária | Git / repositório (§10) |
| Evidência | repo + `resultados/relatorio.txt` |

## 9. Entregas (§9)

| Entrega | Status |
| --- | --- |
| Dados / limpeza / EDA | Coleção 13 + cache + gráficos |
| Referência | `persistencia_1passo` (+ multi-ano) |
| Modelos | Ridge (CV) e Random Forest |
| Métricas | teste 2020–2024 em `relatorio.txt` / `docs/resultados.md` |
| Integração | `predict_only.py` |

## 10. Premissas

- Xlsx e cache fora do Git.
- Lags no teste usam emissão **observada** de t−1 (previsão de um passo).
- Empatar ou perder para a 1 passo em MAE **não invalida** o método; ganho vs multi-ano e RMSE também contam.
