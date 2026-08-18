# O que fazer em casa — frente de dados e modelo (seção 9)

Documento só da **sua** parte. Objetivo: chegar, executar, arquivar evidências e ficar tranquilo. Não precisa de frontend, API nem artigo inteiro da equipe.

| Campo | Valor |
| --- | --- |
| Nome | Thiago Belagamba Bueno |
| Curso | Ciência da Computação — 8º semestre, noite |
| Disciplina | Tendências em Ciência da Computação |
| Professor | Prof. Dr. Renato Gonçalves Rocha |
| Função | Análise de dados e mecanismo preditivo (seção 9) |
| Repositório | https://github.com/ThiagoBelagamba/n2o-forecast |

---

## Antes de começar (5 minutos)

Leve para o outro PC:

- [ ] A pasta do projeto **ou** um `git pull` em `main`
- [ ] O arquivo `emissao_gases.csv` (não está no Git; ~82 MB)

Não leve a pasta `.venv` desta máquina. Ela não funciona em outro Windows/Python.

No PC de casa:

- [ ] Python **3.11 ou mais novo** (`python --version`)
- [ ] `emissao_gases.csv` copiado para `data\emissao_gases.csv` (ao lado de `data\README.md`)

---

## Passo 1 — Ambiente

Na **raiz** do projeto (`n2o-forecast` / `gas`):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Se o `pip` falhar por versão do Python, instale 3.11+ e repita. Não precisa instalar nada globalmente.

Confira:

```powershell
dir data\emissao_gases.csv
```

Se o arquivo não existir, o `main.py` para e avisa. Copie o CSV e siga.

---

## Passo 2 — Treinar (o passo longo)

```powershell
.\.venv\Scripts\python.exe -u main.py
```

O `-u` mostra o progresso na hora. A EDA é rápida. O **Random Forest** (grid 12 × 5 folds) pode levar **dezenas de minutos**. Deixe a janela aberta. `n_jobs=2` é de propósito.

**Pronto quando aparecer:** `Processo concluído. Relatório: ...`

Se travar ou fechar no meio, rode de novo. A pasta `resultados\` é recriada.

---

## Passo 3 — Gráfico anual

Só depois do passo 2:

```powershell
.\.venv\Scripts\python.exe scripts\grafico.py
```

---

## Passo 4 — Conferir se gerou tudo

Tem que existir:

| Arquivo | Para quê |
| --- | --- |
| `resultados\relatorio.txt` | Métricas no teste 2011–2019 vs persistência |
| `resultados\modelo_n2o.joblib` | Modelo treinado |
| `resultados\emissao_n2o_com_predicoes.csv` | Predições com coluna `conjunto` |
| `resultados\graficos\eda_*.png` | Análise exploratória |
| `resultados\graficos\reais_vs_preditos.png` | Teste |
| `resultados\graficos\emissoes_por_ano.png` | Série com corte 2010/2011 |
| `resultados\graficos\feature_importances.png` | Só se o melhor modelo for Random Forest |

Abra `relatorio.txt` e anote:

- MAE / RMSE / MedAE / R² do modelo no **teste**
- MAE da **persistência**
- Δ MAE (persistência − modelo)

Valor **positivo** = você errou menos que copiar 2010. Valor **zero ou negativo** também vale: é resultado, não bug. Guarde o arquivo do jeito que saiu.

Opcional, para testar o `.joblib`:

```powershell
.\.venv\Scripts\python.exe scripts\predict_only.py
```

Gera `resultados\emissao_n2o_novas_predicoes.csv`. Não é obrigatório para a entrega.

---

## Passo 5 — Empacotar a sua evidência (seção 9.2)

Crie uma pasta, por exemplo `entrega-thiago-secao9\`, com:

1. Link do GitHub (código reproduzível).
2. `docs\escopo.md` (já está no repo: pergunta, método, sua função).
3. `resultados\relatorio.txt`.
4. Os PNGs de `resultados\graficos\` (EDA + reais vs preditos + série anual).
5. Uma página curta (pode ser um `.txt` ou slides de 3–5 páginas) com:
   - o que é o alvo (N2O em t);
   - split 1970–2010 / 2011–2019 e por quê;
   - persistência como referência;
   - a tabela de métricas copiada do relatório;
   - limitação: sem defasagens o modelo tende a colar em 2010.

Isso cobre as entregas mínimas do manual: dados, limpeza, EDA, gráficos, métodos, referência, modelos testados, métricas, comparação, código, modelo, limitações, integração (`predict_only.py`).

**Não suba o CSV de 82 MB nem o `.venv` para o Git.** O `.joblib` e o `relatorio.txt` o Git ignora (`resultados/` está no `.gitignore`). Guarde essa pasta **localmente** (Drive, pendrive) e/ou anexe no Moodle/e-mail da disciplina. Se o professor pedir o modelo no Git, avise e aí tiramos `resultados/` do `.gitignore`.

---

## O que você NÃO precisa fazer hoje

- Artigo completo, slides da equipe, cronograma, atas.
- Frontend, backend, banco, UX, testes de sistema, segurança.
- Reescrever o pipeline se a persistência ganhar.
- Treinar de novo neste PC fraco.

Isso é da equipe ou de depois que o professor pedir.

---

## Se der erro

| Sintoma | O que fazer |
| --- | --- |
| `Dataset não encontrado` | CSV em `data\emissao_gases.csv` |
| `No module named sklearn` | Usar `.\.venv\Scripts\python.exe`, não o Python global |
| Trava no “Fitting 5 folds…” | Esperar; é o grid. Não feche. |
| Sem `feature_importances.png` | Ridge ganhou o CV; o scatter e o relatório bastam |
| Pip reclama de versão | Python 3.11+ e venv novo |

---

## Quando pode ficar tranquilo

Marque só quando **os três** forem verdade:

- [ ] `main.py` terminou sem erro
- [ ] `relatorio.txt` tem métricas de teste **e** persistência
- [ ] Pasta de evidência copiada para Drive/pendrive (código no GitHub + relatório + gráficos)

Aí a seção 9 está entregável. O resto é redação coletiva e as outras funções da equipe.
