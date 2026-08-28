# -*- coding: utf-8 -*-
# results_entrega.py — agregacoes e export para fechamento da entrega (TCC)
import json
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFORMAL_CONFIG, MODEL_CONFIG, PATHS, PLOT_STYLE, RIDGE_MAE_BARRIER, ensure_dirs
from forecasting_conformal import evaluate_coverage


def _hier_nivel1(unique_id):
    parts = str(unique_id).split('/')
    return parts[1] if len(parts) > 1 else 'unknown'


def _require_file(path, name):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f'Artefato ausente: {name} ({path}). '
            'Rode main.py e scripts/run_conformal.py antes.'
        )


def attach_nivel1(df, uid_col='unique_id'):
    out = df.copy()
    out['nivel_1'] = out[uid_col].map(_hier_nivel1)
    return out


def build_mae_por_nivel1(intervals_path=None, point_col=None):
    path = intervals_path or PATHS['conformal_intervals']
    _require_file(path, 'conformal_intervals.csv')
    point_col = point_col or CONFORMAL_CONFIG['model_col']
    df = pd.read_csv(path)
    df = attach_nivel1(df)
    rows = []
    for n1, g in df.groupby('nivel_1'):
        y = g['y'].astype(float)
        pred = g[point_col].astype(float).clip(lower=0.0)
        rows.append({
            'nivel_1': n1,
            'n': len(g),
            'MAE': float((y - pred).abs().mean()),
            'RMSE': float(np.sqrt(((y - pred) ** 2).mean())),
        })
    return pd.DataFrame(rows).sort_values('MAE', ascending=False)


def build_conformal_por_nivel1(intervals_path=None, point_col=None):
    path = intervals_path or PATHS['conformal_intervals']
    _require_file(path, 'conformal_intervals.csv')
    point_col = point_col or CONFORMAL_CONFIG['model_col']
    df = attach_nivel1(pd.read_csv(path))
    rows = []
    for n1, g in df.groupby('nivel_1'):
        cov = evaluate_coverage(g, point_col=point_col)
        rows.append({
            'nivel_1': n1,
            'coverage_90': cov.get('coverage_90'),
            'coverage_95': cov.get('coverage_95'),
            'width_mean_90': cov.get('width_mean_90'),
            'width_mean_95': cov.get('width_mean_95'),
            'n': cov.get('n_90', len(g)),
        })
    return pd.DataFrame(rows).sort_values('nivel_1')


def build_cenario_resumo(conf_proj_path=None, proj_path=None, point_col=None):
    conf_path = conf_proj_path or PATHS['conformal_projection']
    proj_path = proj_path or PATHS['projection']
    _require_file(conf_path, 'conformal_projecao_2025_2026.csv')
    _require_file(proj_path, 'projecao_2025_2026.csv')
    point_col = point_col or CONFORMAL_CONFIG['model_col']

    conf = attach_nivel1(pd.read_csv(conf_path))
    proj = attach_nivel1(pd.read_csv(proj_path))

    agg_cols = [point_col, 'lo_90', 'hi_90', 'lo_95', 'hi_95']
    agg_cols = [c for c in agg_cols if c in conf.columns]

    by_sector = (
        conf.groupby(['nivel_1', 'ano'], as_index=False)[agg_cols]
        .sum()
    )
    pers = (
        proj.groupby(['nivel_1', 'ano'], as_index=False)['persistencia_2024']
        .sum()
    )
    by_sector = by_sector.merge(pers, on=['nivel_1', 'ano'], how='left')
    by_sector['escopo'] = 'setor'

    national = (
        conf.groupby('ano', as_index=False)[agg_cols]
        .sum()
    )
    nat_pers = proj.groupby('ano', as_index=False)['persistencia_2024'].sum()
    national = national.merge(nat_pers, on='ano', how='left')
    national['nivel_1'] = 'Brasil'
    national['escopo'] = 'nacional'

    out = pd.concat([national, by_sector], ignore_index=True)
    out = out.rename(columns={point_col: 'previsto'})
    return out


def build_tabela_experimentos(metrics_path=None):
    metrics_path = metrics_path or PATHS['metrics']
    _require_file(metrics_path, 'metricas_comparacao.csv')
    m = pd.read_csv(metrics_path)
    barrier = RIDGE_MAE_BARRIER

    experiments = [
        ('Ridge / MinTrace(ols)', 'Ridge/MinTrace_method-ols', 'baseline'),
        ('Ridge / MinTrace(mint_shrink)', 'Ridge/MinTrace_method-mint_shrink', 'baseline'),
        ('LGBM Tweedie (tunado)', 'LGBM_Tweedie', 'tuning'),
        ('LGBM / MinTrace(mint_shrink)', 'LGBM_Tweedie/MinTrace_method-mint_shrink', 'tuning'),
        ('NHITS', None, 'nhits_benchmark.json'),
        ('Ensemble Ridge+NHITS', None, 'ensemble_benchmark.json'),
        ('DTW cluster Ridge', None, 'dtw_benchmark.json'),
    ]

    rows = []
    mdict = m.set_index('metodo')['MAE'].to_dict()
    for label, col, src in experiments:
        mae = None
        beat = False
        if col and col in mdict:
            mae = float(mdict[col])
            beat = mae < barrier
        elif src == 'nhits_benchmark.json' and os.path.isfile(PATHS.get('nhits_benchmark', '')):
            with open(PATHS['nhits_benchmark'], encoding='utf-8') as f:
                d = json.load(f)
            mae = d.get('nhits_best_mae')
            beat = bool(d.get('beat_barrier'))
        elif src == 'ensemble_benchmark.json' and os.path.isfile(PATHS.get('ensemble_benchmark', '')):
            with open(PATHS['ensemble_benchmark'], encoding='utf-8') as f:
                d = json.load(f)
            mae = d.get('best_mae')
            beat = bool(d.get('beat_barrier'))
        elif src == 'dtw_benchmark.json' and os.path.isfile(PATHS.get('dtw_benchmark', '')):
            with open(PATHS['dtw_benchmark'], encoding='utf-8') as f:
                d = json.load(f)
            mae = d.get('best_dtw_mae')
            beat = bool(d.get('beat_barrier'))
        if mae is not None:
            rows.append({
                'experimento': label,
                'MAE': mae,
                'barreira_ridge_ols': barrier,
                'superou_barreira': beat,
                'delta_vs_barreira': barrier - mae,
            })
    return pd.DataFrame(rows)


def build_sintese_entrega(metrics_path=None):
    metrics_path = metrics_path or PATHS['metrics']
    _require_file(metrics_path, 'metricas_comparacao.csv')
    m = pd.read_csv(metrics_path).set_index('metodo')

    champ_ols = 'Ridge/MinTrace_method-ols'
    champ_shrink = 'Ridge/MinTrace_method-mint_shrink'
    pers = 'persistencia_multi_ano'

    best_main = champ_shrink
    if champ_shrink in m.index and champ_ols in m.index:
        if m.loc[champ_ols, 'MAE'] < m.loc[champ_shrink, 'MAE']:
            best_main = champ_ols

    sintese = {
        'modelo_campeao_main': best_main,
        'mae_campeao_main': float(m.loc[best_main, 'MAE']),
        'barreira_ridge_ols': RIDGE_MAE_BARRIER,
        'mae_ridge_ols': float(m.loc[champ_ols, 'MAE']) if champ_ols in m.index else None,
        'persistencia_mae': float(m.loc[pers, 'MAE']) if pers in m.index else None,
        'delta_mae_vs_persistencia': (
            float(m.loc[pers, 'MAE'] - m.loc[best_main, 'MAE'])
            if pers in m.index and best_main in m.index else None
        ),
        'nota_modelo': (
            'Barreira historica 83.85 t refere-se a Ridge/MinTrace(ols) em run anterior; '
            'ultimo main.py pode diferir levemente (88.03 mint_shrink vs 89.82 ols).'
        ),
    }

    if os.path.isfile(PATHS.get('conformal_coverage', '')):
        with open(PATHS['conformal_coverage'], encoding='utf-8') as f:
            cov = json.load(f)
        leaf = cov.get('leaf_all', {})
        sintese['conformal'] = {
            'coverage_90_folha': leaf.get('coverage_90'),
            'coverage_95_folha': leaf.get('coverage_95'),
            'width_mean_90_folha': leaf.get('width_mean_90'),
        }

    if os.path.isfile(PATHS.get('timing', '')):
        with open(PATHS['timing'], encoding='utf-8') as f:
            sintese['inferencia'] = json.load(f)

    return sintese


def plot_mae_por_nivel1(mae_df, path=None):
    ensure_dirs()
    path = path or os.path.join(PATHS['plots_dir'], 'mae_por_nivel1.png')
    plt.style.use(PLOT_STYLE)
    df = mae_df.sort_values('MAE', ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(df))))
    ax.barh(df['nivel_1'], df['MAE'], color='steelblue')
    ax.set_xlabel('MAE (t)')
    ax.set_title('MAE por setor (nivel_1) — teste cego 2020–2024')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Grafico: {path}')


def plot_cenario_ic(cenario_df, path=None):
    ensure_dirs()
    path = path or os.path.join(PATHS['plots_dir'], 'cenario_2025_2026_ic.png')
    nat = cenario_df[cenario_df['escopo'] == 'nacional'].sort_values('ano')
    if nat.empty:
        return
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = nat['ano'].astype(int)
    ax.plot(x, nat['previsto'], 'o--', color='darkorange', label='Cenario (ponto)')
    if 'lo_90' in nat.columns and 'hi_90' in nat.columns:
        ax.fill_between(x, nat['lo_90'], nat['hi_90'], alpha=0.25, color='darkorange',
                        label='IC 90% conformal')
    if 'persistencia_2024' in nat.columns:
        ax.plot(x, nat['persistencia_2024'], 's:', color='gray', label='Persistencia 2024')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Emissao N2O total (t)')
    ax.set_title('Cenario 2025–2026 — total nacional com incerteza')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Grafico: {path}')


def write_leitura_completa(sintese, exp_df, conformal_df=None):
    """Gera leitura-secao9.txt com narrativa de fechamento."""
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    path = os.path.join(PATHS['results_dir'], 'leitura-secao9.txt')
    conf = sintese.get('conformal', {})

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Previsao multi-ano de N2O (t) — secao 9.2 (fechamento)\n')
        f.write('SEEG Colecao 13 + Ridge/MinT + Conformal Prediction\n')
        f.write('https://github.com/ThiagoBelagamba/n2o-forecast\n\n')

        f.write('1. PERGUNTA E PROTOCOLO\n')
        f.write(
            f'   Com dados ate {corte}, prever {corte + 1}–'
            f'{corte + MODEL_CONFIG["test_horizon"]} sem ver anos intermediarios.\n'
        )
        f.write('   Previsao direta h=5; referencia: persistencia multi-ano.\n\n')

        f.write('2. RESULTADO PRINCIPAL (erro pontual)\n')
        f.write(f'   Modelo campeao (ultimo main): {sintese.get("modelo_campeao_main")}\n')
        f.write(f'   MAE: {sintese.get("mae_campeao_main", 0):.2f} t\n')
        f.write(f'   Persistencia multi-ano: {sintese.get("persistencia_mae", 0):.2f} t\n')
        f.write(
            f'   Delta MAE (ganho): {sintese.get("delta_mae_vs_persistencia", 0):.2f} t\n'
        )
        f.write(f'   Barreira Ridge/MinTrace(ols): {RIDGE_MAE_BARRIER:.2f} t\n')
        f.write(f'   {sintese.get("nota_modelo", "")}\n\n')

        f.write('3. EXPERIMENTOS (platô MAE)\n')
        for _, row in exp_df.iterrows():
            flag = 'sim' if row['superou_barreira'] else 'nao'
            f.write(
                f'   {row["experimento"]}: MAE {row["MAE"]:.2f} t | '
                f'superou barreira: {flag}\n'
            )
        f.write(
            '\n   Conclusao: erro pontual estabilizou; LGBM/NHITS/ensemble/DTW '
            'nao superaram Ridge de forma consistente.\n\n'
        )

        f.write('4. INCERTEZA (Conformal Prediction)\n')
        if conf:
            f.write(
                f'   Cobertura folha 90%: {conf.get("coverage_90_folha", 0) * 100:.1f}%\n'
            )
            f.write(
                f'   Cobertura folha 95%: {conf.get("coverage_95_folha", 0) * 100:.1f}%\n'
            )
            f.write(
                f'   Largura media IC 90% (folha): {conf.get("width_mean_90_folha", 0):.1f} t\n'
            )
        f.write('   Nacional: poucos pontos no teste; IC agregado ainda limitado.\n\n')

        if conformal_df is not None and len(conformal_df):
            f.write('5. COBERTURA POR SETOR (nivel_1)\n')
            for _, row in conformal_df.iterrows():
                c90 = row.get('coverage_90')
                if c90 is not None and not pd.isna(c90):
                    f.write(
                        f'   {row["nivel_1"]}: cov90={c90 * 100:.1f}% | '
                        f'width90={row.get("width_mean_90", 0):.0f} t\n'
                    )
            f.write('\n')

        f.write('6. CENARIO 2025–2026\n')
        f.write('   Ponto + IC 90% agregado por setor e nacional.\n')
        f.write('   Arquivo: cenario_2025_2026_resumo.csv (nao e inventario SEEG).\n\n')

        f.write('7. INFERENCIA\n')
        f.write('   MinT classico + clip: nucleo ~1.2 s (ver tempo_inferencia.json).\n')

    print(f'Leitura: {path}')


def run_export_entrega():
    """Pipeline completo de exportacao para entrega."""
    ensure_dirs()
    mae_df = build_mae_por_nivel1()
    conf_df = build_conformal_por_nivel1()
    cenario_df = build_cenario_resumo()
    exp_df = build_tabela_experimentos()
    sintese = build_sintese_entrega()

    mae_df.to_csv(PATHS['mae_nivel1'], index=False)
    conf_df.to_csv(PATHS['conformal_nivel1'], index=False)
    cenario_df.to_csv(PATHS['cenario_resumo'], index=False)
    exp_df.to_csv(PATHS['tabela_experimentos'], index=False)
    with open(PATHS['sintese_entrega'], 'w', encoding='utf-8') as f:
        json.dump(sintese, f, indent=2, ensure_ascii=False)

    plot_mae_por_nivel1(mae_df)
    plot_cenario_ic(cenario_df)
    write_leitura_completa(sintese, exp_df, conf_df)

    print(f'Salvo: {PATHS["mae_nivel1"]}')
    print(f'Salvo: {PATHS["conformal_nivel1"]}')
    print(f'Salvo: {PATHS["cenario_resumo"]}')
    print(f'Salvo: {PATHS["tabela_experimentos"]}')
    print(f'Salvo: {PATHS["sintese_entrega"]}')
    return sintese
