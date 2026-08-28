# -*- coding: utf-8 -*-
"""Exporta artefatos prontos para slides/PDF (markdown + PNGs)."""
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import PATHS, ensure_dirs

PNG_ESSENTIALS = [
    'comparacao_real_vs_previsto.png',
    'conformal_nacional.png',
    'cenario_2025_2026_ic.png',
    'mae_por_nivel1.png',
    'projecao_2025_2026.png',
    'total_nacional_teste.png',
]


def run_export_slides():
    ensure_dirs()
    export_dir = PATHS['export_dir']
    img_dir = os.path.join(export_dir, 'graficos')
    os.makedirs(img_dir, exist_ok=True)

    for name in PNG_ESSENTIALS:
        src = os.path.join(PATHS['plots_dir'], name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(img_dir, name))
            print(f'Copiado: {name}')
        else:
            print(f'Aviso: PNG ausente — {name} (rode export_entrega.py / main.py)')

    sintese = {}
    if os.path.isfile(PATHS['sintese_entrega']):
        with open(PATHS['sintese_entrega'], encoding='utf-8') as f:
            sintese = json.load(f)

    dados = {
        'modelo_campeao': sintese.get('modelo_campeao_main'),
        'mae_campeao': sintese.get('mae_campeao_main'),
        'barreira_ridge_ols': sintese.get('barreira_ridge_ols'),
        'persistencia_mae': sintese.get('persistencia_mae'),
        'delta_mae': sintese.get('delta_mae_vs_persistencia'),
        'conformal_folha_90': sintese.get('conformal', {}).get('coverage_90_folha'),
        'conformal_folha_95': sintese.get('conformal', {}).get('coverage_95_folha'),
    }
    with open(os.path.join(export_dir, 'dados_slides.json'), 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)

    mae = dados.get('mae_campeao')
    delta = dados.get('delta_mae')
    cov90 = dados.get('conformal_folha_90')
    cov_str = f'{cov90 * 100:.1f}%' if cov90 else '—'
    mae_str = f'{mae:.2f} t' if mae is not None else '—'
    delta_str = f'{delta:.2f} t' if delta is not None else '—'
    barreira = dados.get('barreira_ridge_ols', 83.85)
    barreira_str = f'{barreira:.2f} t' if barreira is not None else '—'

    slides = f"""# Apresentação — Previsão N2O (TCC)

> Export gerado automaticamente. Abra no Typora / VS Code / exporte para PDF.

---

## Slide 1 — Pergunta

- Com dados SEEG até **2019**, prever **2020–2024** sem ver anos intermediários
- Referência: persistência multi-ano (copiar 2019)
- Protocolo anti-vazamento: previsão direta h=5 + reconciliação MinT

---

## Slide 2 — Resultado principal

![Comparação real vs previsto](graficos/comparacao_real_vs_previsto.png)

- Modelo campeão: **{dados.get('modelo_campeao', 'Ridge/MinTrace')}**
- MAE teste: **{mae_str}**
- Δ vs persistência: **{delta_str}** (ganho)
- Barreira Ridge/MinTrace(ols): **{barreira_str}**

---

## Slide 3 — Platô MAE (experimentos negativos)

- LGBM tunado, NHITS, ensemble Ridge+LGBM/NHITS, DTW por cluster
- Nenhum superou Ridge de forma consistente
- Próximo passo: **incerteza** (Conformal Prediction), não novo tuning pontual

Ver `resultados/tabela_experimentos.csv`

---

## Slide 4 — Incerteza (Conformal)

![Conformal nacional](graficos/conformal_nacional.png)

- Cobertura folha 90%: **{cov_str}** (próximo do nominal)
- Intervalos simétricos sobre Ridge/MinTrace(ols)
- Nacional: poucos pontos no teste — IC agregado limitado

---

## Slide 5 — MAE por setor

![MAE por nivel_1](graficos/mae_por_nivel1.png)

- Erro heterogêneo entre setores SEEG (nível 1)
- Agropecuária concentra maior parte da emissão

---

## Slide 6 — Cenário 2025–2026

![Cenário com IC](graficos/cenario_2025_2026_ic.png)

![Projeção](graficos/projecao_2025_2026.png)

- Ponto + faixa 90% conformal (total nacional)
- **Cenário de curto prazo — não substitui inventário SEEG oficial**

---

## Slide 7 — Pipeline e inferência

![Total nacional teste](graficos/total_nacional_teste.png)

- MinT clássico + clip: núcleo ~**1,2 s**
- UI: Nest + Next lendo `resultados/`
- Repo: https://github.com/ThiagoBelagamba/n2o-forecast

---

## Slide 8 — Conclusão

1. Ridge reconciliado vence persistência multi-ano no teste cego
2. Erro pontual estabilizou (platô MAE)
3. Conformal entrega intervalos calibrados nas folhas
4. Entrega operacional: cenário 2025–2026 com incerteza
"""
    slides_path = os.path.join(export_dir, 'slides.md')
    with open(slides_path, 'w', encoding='utf-8') as f:
        f.write(slides)
    print(f'Salvo: {slides_path}')
    print(f'Salvo: {os.path.join(export_dir, "dados_slides.json")}')


if __name__ == '__main__':
    run_export_slides()
