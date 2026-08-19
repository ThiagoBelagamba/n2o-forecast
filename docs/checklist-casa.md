# O que fazer em casa — frente de dados e modelo (seção 9)

Documento só desta frente. A execução mínima **já foi feita** nesta máquina: `main.py`, `scripts/grafico.py` e `scripts/predict_only.py` terminaram. Números e interpretação: [`docs/resultados.md`](resultados.md).

| Campo | Valor |
| --- | --- |
| Nome | Thiago Belagamba Bueno |
| Curso | Ciência da Computação — 8º semestre, noite |
| Disciplina | Tendências em Ciência da Computação |
| Professor | Prof. Dr. Renato Gonçalves Rocha |
| Função | Análise de dados e mecanismo preditivo (seção 9) |
| Repositório | https://github.com/ThiagoBelagamba/n2o-forecast |

---

## Status (não precisa treinar de novo)

- [x] `main.py` terminou sem erro (grid RF 12×5 e Ridge 4×5 até o fim)
- [x] `relatorio.txt` tem métricas de teste **e** persistência
- [x] Gráficos de EDA, teste, vieses e série anual em `resultados/graficos/`
- [x] `modelo_n2o.joblib` + `scripts/predict_only.py` (CSV novo gerado)
- [x] Página curta: `resultados/leitura-secao9.txt` e `docs/resultados.md`

Resultado no teste 2011–2019: **persistência MAE 219,73 t; RF 232,78 t; Ridge 1.593,26 t**. Δ MAE = −13,05 t (o RF não supera copiar 2010). Isso é o experimento concluído, não um bug.

Se for **reproduzir** em outro PC, use a sequência abaixo. Não copie a pasta `.venv`.

---

## Sequência de execução (reproduzir)

Na **raiz** do projeto, PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe -u main.py
.\.venv\Scripts\python.exe scripts\grafico.py
.\.venv\Scripts\python.exe scripts\predict_only.py
```

1. `main.py` — EDA, split, baselines, GridSearch, `relatorio.txt`, `.joblib`, gráficos de teste.
2. `scripts/grafico.py` — exige `resultados\emissao_n2o_com_predicoes.csv`; gera `emissoes_por_ano.png`.
3. `scripts/predict_only.py` — exige `modelo_n2o.joblib`; gera `emissao_n2o_novas_predicoes.csv` (não sobrescreve o do treino).

O CSV precisa estar em `data\emissao_gases.csv` (~82 MB; fora do Git). Python **3.11+**.

---

## Arquivos que têm que existir

| Arquivo | Para quê |
| --- | --- |
| `resultados\relatorio.txt` | Métricas no teste 2011–2019 vs persistência, vieses, §9.2 |
| `resultados\leitura-secao9.txt` | Página curta (alvo, split, tabela, limitação) |
| `resultados\metricas_comparacao.csv` | RF vs Ridge vs baselines |
| `resultados\modelo_n2o.joblib` | Modelo treinado (Random Forest) |
| `resultados\emissao_n2o_com_predicoes.csv` | Predições com coluna `conjunto` |
| `resultados\graficos\eda_*.png` | Análise exploratória |
| `resultados\graficos\reais_vs_preditos.png` | Teste |
| `resultados\graficos\emissoes_por_ano.png` | Série com corte 2010/2011 |
| `resultados\graficos\feature_importances.png` | RF foi o melhor no CV |
| `resultados\graficos\mae_por_faixa.png` | Viés de escala |
| `docs\resultados.md` | Mesmos números, para a banca ler no Git |

O Git **não** versiona o CSV de 82 MB, o `.venv`, o cache do sklearn, os CSVs grandes de predição nem o `.joblib` (~103 MB, acima do limite do GitHub). Relatório, gráficos e tabelas pequenas de métrica/viés entram no repositório. O modelo se recria com `main.py`.

Empacotar para Moodle/Drive, se pedir: GitHub + `docs/escopo.md` + `docs/resultados.md` + `resultados/relatorio.txt` + PNGs.

---

## O que esta frente NÃO precisa fazer

- Artigo completo, slides da equipe, cronograma, atas.
- Frontend, backend, banco, UX, testes de sistema, segurança.
- Reescrever o pipeline porque a persistência ganhou.
- Treinar de novo só para “tentar um número melhor” com as mesmas features.

---

## Se der erro (reprodução)

| Sintoma | O que fazer |
| --- | --- |
| `Dataset não encontrado` | CSV em `data\emissao_gases.csv` |
| `No module named sklearn` | Usar `.\.venv\Scripts\python.exe`, não o Python global |
| Trava no “Fitting 5 folds…” | Esperar; é o grid. Não feche. |
| Cursor não abre o CSV | Arquivo de ~82 MB; o editor quebra. O Python lê. |
| Pip reclama de versão | Python 3.11+ e venv novo |
