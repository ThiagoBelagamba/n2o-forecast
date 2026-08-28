# Roteiro de apresentação (8–10 slides)

Use `resultados/export/slides.md` (gerado por `scripts/export_slides.py`) como base. PNGs copiados para `resultados/export/graficos/`.

---

## Slide 1 — Título e pergunta

**Título:** Previsão multi-ano de N₂O (SEEG Coleção 13)

- Com dados até **2019**, prever **2020–2024** sem ver anos intermediários
- Referência: persistência multi-ano
- Repo: https://github.com/ThiagoBelagamba/n2o-forecast

---

## Slide 2 — Protocolo

- Previsão direta h=5 (anti-vazamento)
- Modelos: Ridge, LGBM Tweedie, RF via MLForecast
- Reconciliação hierárquica MinT + clip ≥ 0
- Teste cego 2020–2024; cenário 2025–2026

---

## Slide 3 — Resultado principal

**PNG:** `comparacao_real_vs_previsto.png`

- Ridge/MinTrace: MAE ~84–90 t vs persistência ~103 t
- Ganho Δ MAE positivo vs copiar 2019

---

## Slide 4 — Platô MAE

**Tabela:** `tabela_experimentos.csv`

- LGBM tunado, NHITS, ensemble, DTW: **não** superaram Ridge
- Conclusão: erro pontual estabilizou

---

## Slide 5 — Incerteza (Conformal)

**PNG:** `conformal_nacional.png`

- Cobertura folha ~91% / 95%
- Intervalos simétricos sobre Ridge reconciliado
- Nacional: limitação (5 anos de teste)

---

## Slide 6 — MAE por setor

**PNG:** `mae_por_nivel1.png`

- Heterogeneidade entre setores SEEG (nível 1)
- Agropecuária domina emissão

---

## Slide 7 — Cenário 2025–2026

**PNG:** `cenario_2025_2026_ic.png`, `projecao_2025_2026.png`

- Ponto + IC 90% nacional
- **Cenário — não inventário SEEG oficial**

---

## Slide 8 — Inferência e entrega

**PNG:** `total_nacional_teste.png`

- MinT + clip: ~1,2 s
- API Nest + front Next
- Export: `resultados/export/slides.md`

---

## Slide 9 — Conclusão

1. Ridge reconciliado vence persistência no teste cego
2. Platô MAE: próximo passo = incerteza, não novo tuning
3. Conformal calibrado nas folhas
4. Entrega: cenário 2025–2026 com faixas + UI + slides

---

## Como exportar PDF

1. Rode `scripts/export_slides.py`
2. Abra `resultados/export/slides.md` no Typora ou VS Code
3. Exporte para PDF ou importe PNGs no Canva/PowerPoint

Números-chave em `resultados/export/dados_slides.json`.
