# Escopo do projeto

**Título provisório:** Previsão cronológica de emissão de N2O a partir do inventário SEEG (1970–2019)

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

Este documento delimita **para quem**, **o quê**, **como** e **quem faz o quê**. A parte já implementada no repositório corresponde à frente de **análise de dados e mecanismo preditivo** (Manual de Funções, seção 9).

---

## 1. Para quem

O recorte **não é um aplicativo** (não há tela de login nem produto comercial). O entregável é um **mecanismo preditivo + avaliação honesta**. Quem se beneficia disso:

| Interessado | Uso esperado |
| --- | --- |
| Engenheiros ambientais, analistas de GEE e equipes de inventário | Projetar emissão de N2O por setor no horizonte seguinte ao último ano disponível; priorizar subsetores (agropecuária, dejetos, solos) para mitigação; testar se “repetir o último ano” já basta ou se um modelo ganha |
| Gestores e consultorias de clima / ESG | Insumo quantitativo (com incerteza) para cenário de curto prazo, não substituto do inventário oficial |
| Pesquisadores | Reproduzir o experimento e comparar métodos em split cronológico |
| Equipe e banca da disciplina | Avaliar pergunta, método, evidências e limitações |

O N2O entra no inventário brasileiro e na conversão para CO2e. Um engenheiro ambiental **usaria** uma previsão setorial se ela for melhor que a persistência e vier com métricas (MAE/RMSE) e recorte temporal explícito. Enquanto o modelo não superar “copiar 2010”, o uso operacional fica limitado — e isso também é informação útil para esse público.

---

## 2. Problema

O inventário SEEG descreve emissões brasileiras por setor e gás, ano a ano. Um modelo treinado com **split aleatório** neste painel (528 setores × 50 anos) coloca o **mesmo setor** no treino e no teste. O R² fica alto porque o algoritmo memoriza o nível típico do setor, não porque prevê o que ainda não aconteceu.

O problema investigado é: **é possível prever a emissão de N2O (t) de cada setor em anos futuros, usando apenas o histórico até um ano de corte, sem vazamento temporal?**

---

## 3. Pergunta de pesquisa

> Com dados nacionais do SEEG (1970–2019), um modelo de regressão treinado só até 2010 consegue prever a emissão de N2O (t) no período 2011–2019 melhor do que repetir, para cada setor, o último valor observado em 2010?

---

## 4. Objetivos

**Geral:** desenvolver e avaliar, de forma reproduzível, um mecanismo preditivo de emissão de N2O (t) com validação cronológica.

**Específicos:**

1. Descrever o recorte N2O do SEEG (volume, ausências, assimetria do alvo).
2. Definir split de treino (1970–2010) e teste (2011–2019) e validação com `TimeSeriesSplit`.
3. Estabelecer baselines justos, em especial a **persistência** (último valor do setor em 2010).
4. Treinar e comparar pelo menos dois métodos (Random Forest e Ridge), otimizando **MAE**.
5. Documentar limitações: o modelo não deve ser apresentado como previsão operacional se não superar a persistência.

---

## 5. O que entra no escopo

- Coleta e uso do CSV SEEG (Coleção 8, série 1970–2019, formato longo).
- Filtro `gas == "N2O (t)"`.
- Análise exploratória, limpeza, imputação de categóricas ausentes.
- Pipeline Python reproduzível (`main.py`, módulos, `requirements.txt`).
- Treino, avaliação no teste futuro, relatório de métricas, modelo serializado (`.joblib`).
- Interpretação frente à persistência e registro das limitações.

## 6. O que fica fora (deste recorte)

Não fazem parte da entrega **desta função**, salvo se a equipe assumir outras frentes e o professor aprovar o escopo ampliado:

- aplicativo, frontend, prototipação UX;
- API/backend e banco de dados transacional;
- previsão para 2020 em diante ou uso da Coleção 13;
- desagregação por UF/município;
- outros gases (CH4, CO2, CO2e);
- engenharia de defasagens/tendência por setor (possível trabalho futuro, não o pipeline atual).

Se a disciplina exigir sistema completo (frontend, backend, BD, testes, segurança), isso é **escopo da equipe**, não desta frente isolada.

---

## 7. Como (método)

| Decisão | Escolha | Motivo |
| --- | --- | --- |
| Alvo | `emissao` em t de N2O | Recorte único e interpretável |
| Features | `ano` + 9 categóricas setoriais | Schema do inventário; sem vazamento do alvo |
| Split | treino ≤ 2010, teste ≥ 2011 | O teste é o futuro |
| Validação | `TimeSeriesSplit` (5 folds) no treino ordenado por ano | Não embaralhar o tempo no grid |
| Métrica do grid | MAE (não R²) | Alvo com ~46% zeros e cauda longa |
| Referência | persistência por setor em 2010 | Benchmark correto de previsão |
| Métodos | Dummy, média por grupo, persistência, Ridge, Random Forest | Comparar, não só “rodar um RF” |
| Reprodução | Python 3.13, `requirements.txt`, Git sem o CSV (~82 MB) | Dataset fora do repositório |

Código correspondente: `data_loading.py`, `data_preprocessing.py`, `model_training.py`, `results_saving.py`, `config.py`, `scripts/`.

---

## 8. Distribuição de responsabilidades

Conforme o Manual de Funções: cada integrante assume uma função **principal**. Abaixo, a função já exercida neste repositório. As demais ficam para a equipe preencher.

### 8.1 Função principal (já em execução)

| Campo | Registro |
| --- | --- |
| Nome | Thiago Belagamba Bueno |
| Função principal | Análise de dados e desenvolvimento do mecanismo preditivo (seção 9) |
| Função secundária | Infraestrutura e versionamento do repositório técnico (seção 10, recorte Git) |
| Evidência | https://github.com/ThiagoBelagamba/n2o-forecast |

**Atribuições desta função (o que já foi feito no código):**

- delimitação do problema e da variável de interesse (`emissao` de N2O);
- avaliação de origem, volume e qualidade (SEEG; 26.400 linhas N2O; 160 alvos nulos; `produto` ~54% ausente);
- limpeza, imputação e EDA;
- split cronológico e checagem de vazamento temporal;
- baselines (média, mediana, zero, média por grupo, persistência);
- treino comparado (Random Forest e Ridge) com MAE;
- documentação técnica no README (métricas, limitações, reprodução);
- organização do repositório (`.gitignore`, `requirements.txt`, `data/` sem o CSV).

**Ainda desta função (pendente de execução em máquina adequada):**

- rodar `main.py` até o fim e arquivar `relatorio.txt`, gráficos e `modelo_n2o.joblib`;
- preencher a seção de resultados do relatório/artigo com MAE/RMSE vs persistência;
- contribuir com método, figuras e limitações na redação coletiva.

### 8.2 Funções da equipe (a alocar)

Não reivindicadas por esta frente. Preencher nome quando a equipe definir:

| Função (manual) | Nome | Observação |
| --- | --- | --- |
| Gestão, planejamento e integração | | plano, cronograma, atas |
| Pesquisa, fundamentação e metodologia | | bibliografia, pergunta, limitações científicas |
| Requisitos | | só se houver sistema além do pipeline |
| UX e prototipação | | fora do recorte atual |
| Frontend | | fora do recorte atual |
| Backend / APIs | | fora do recorte atual |
| Banco de dados | | o armazenamento atual é o CSV SEEG |
| Infraestrutura e implantação | apoio no Git já iniciado | Docker/CI se a equipe decidir |
| Segurança | | se houver sistema com usuários |
| Testes e qualidade | | plano de testes do produto, se houver |
| Documentação institucional | | consolidar este escopo + manuais |
| Produção científica e apresentação | coletiva | artigo, slides, demonstração do pipeline |

---

## 9. Entregas desta frente

| Entrega do manual (seção 9) | Onde está / status |
| --- | --- |
| Descrição do conjunto de dados | README + este escopo |
| Procedimento de limpeza | `data_preprocessing.py` |
| Análise exploratória | `data_loading.py` → `resultados/graficos/` após o treino |
| Gráficos e indicadores | gerados na execução |
| Justificativa dos métodos | README (MAE, persistência, split cronológico) |
| Modelo de referência | persistência em `model_training.py` |
| Modelos testados | Ridge e Random Forest |
| Métricas e comparação | `results_saving.py` / `relatorio.txt` após o treino |
| Código reproduzível | repositório |
| Modelo treinado | `resultados/modelo_n2o.joblib` após o treino |
| Relatório de limitações | README, seção Limitações |
| Documentação para integração | `scripts/predict_only.py` carrega o `.joblib` |

---

## 10. Premissas e riscos

- O CSV local permanece fora do Git; sem `data/emissao_gases.csv` o experimento não roda.
- O GridSearch é computacionalmente pesado; `n_jobs=2` evita saturar a máquina.
- Empatar ou perder para a persistência **não invalida** a pesquisa: é um resultado científico possível e já previsto pelo desenho das features (`ano` + categorias, sem defasagens).
- Mudança de tema, inclusão de sistema web ou troca de recorte temporal deve ser **aprovada pelo professor** antes de entrar no escopo da equipe.
