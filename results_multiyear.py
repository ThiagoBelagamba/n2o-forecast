# -*- coding: utf-8 -*-
# results_multiyear.py — relatório, métricas e projeção 2025–2026
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MODEL_CONFIG, PATHS, PLOT_STYLE, ensure_dirs
from forecasting import compute_metrics


def save_metrics_table(metrics_dict, path=None):
    ensure_dirs()
    path = path or PATHS['metrics']
    rows = []
    for name, m in metrics_dict.items():
        rows.append({'metodo': name, **m})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Metricas: {path}")


def save_report(
    quality,
    metrics,
    tweedie_power,
    protocol_note,
    best_model,
):
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    ref = metrics.get('persistencia_multi_ano') or metrics.get('persistencia')
    best = metrics.get(best_model, {})

    with open(PATHS['report'], 'w', encoding='utf-8') as f:
        f.write("=== RELATORIO MULTI-ANO (Colecao 13 + Tweedie + MinT) ===\n\n")
        f.write(
            f"Protocolo: {protocol_note}\n"
            f"Split: treino <= {corte}, teste {corte + 1}–"
            f"{corte + MODEL_CONFIG['test_horizon']}\n"
            f"Previsao direta h={MODEL_CONFIG['test_horizon']} "
            f"(sem lag observado no teste).\n"
            f"LightGBM Tweedie power={tweedie_power}\n"
            f"Referencia justa deste protocolo: persistencia multi-ano "
            f"(copiar {corte}).\n\n"
        )
        f.write("=== QUALIDADE ===\n")
        for k, v in quality.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== METRICAS NO TESTE (series folha reconciliadas) ===\n")
        for name, m in metrics.items():
            f.write(
                f"\n{name}\n"
                f"MAE:   {m['MAE']:.4f}\n"
                f"RMSE:  {m['RMSE']:.4f}\n"
                f"R2:    {m['R2']:.4f}\n"
                f"MedAE: {m['MedAE']:.4f}\n"
            )
        if ref and best:
            dmae = ref['MAE'] - best['MAE']
            drmse = ref['RMSE'] - best['RMSE']
            f.write("\n=== GANHO vs PERSISTENCIA MULTI-ANO ===\n")
            f.write(
                "Positivo = modelo erra menos que copiar o ultimo ano do treino.\n"
            )
            f.write(f"Melhor modelo: {best_model}\n")
            f.write(f"Delta MAE:  {dmae:.4f} t\n")
            f.write(f"Delta RMSE: {drmse:.4f} t\n")
            if dmae > 0:
                f.write(
                    "\nConclusao: o modelo supera a persistencia multi-ano em MAE "
                    "no horizonte sem inventário intermediário — "
                    "protocolo adequado para cenario 2025–2026.\n"
                )
            else:
                f.write(
                    "\nConclusao: a persistencia multi-ano ainda ganha em MAE; "
                    "o metodo e o teto ficam documentados.\n"
                )
        f.write(
            "\nNota: persistencia_1passo (copiar t-1 observado) NAO e a "
            "referencia deste protocolo; mediria inventário já publicado.\n"
        )
        timing_path = PATHS.get('timing')
        if timing_path and os.path.isfile(timing_path):
            import json
            with open(timing_path, encoding='utf-8') as tf:
                timing = json.load(tf)
            f.write("\n=== TEMPO DE INFERENCIA (MinT classico + clip) ===\n")
            f.write(timing.get('margin_note', '') + "\n")
            secs = timing.get('seconds', {})
            f.write(
                f"predict: {secs.get('predict_horizon')}s | "
                f"reconcile+clip: {secs.get('reconcile_mint_clip')}s | "
                f"nucleo: {secs.get('inference_core')}s\n"
            )
            f.write(
                "NNMinT QP foi abandonado; com MinT + clip a inferencia "
                "cai para a margem de poucos segundos (< 1 min).\n"
            )
    print(f"Relatorio: {PATHS['report']}")


def save_leitura(quality, metrics, best_model, tweedie_power):
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    ref = metrics.get('persistencia_multi_ano', {})
    best = metrics.get(best_model, {})
    path = os.path.join(PATHS['results_dir'], 'leitura-secao9.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Previsao multi-ano de N2O (t) — secao 9.2\n")
        f.write("SEEG Colecao 13 + LightGBM Tweedie + MinT\n")
        f.write("https://github.com/ThiagoBelagamba/n2o-forecast\n\n")
        f.write(
            f"1. Pergunta: com dados ate {corte}, prever "
            f"{corte + 1}–{corte + MODEL_CONFIG['test_horizon']} "
            f"SEM ver os anos intermediarios?\n\n"
        )
        f.write(
            f"2. Metodo: previsao direta h={MODEL_CONFIG['test_horizon']}, "
            f"Tweedie p={tweedie_power}, reconciliacao MinT (nonnegative).\n\n"
        )
        f.write("3. Metricas (folha reconciliada):\n")
        for name, m in metrics.items():
            f.write(f"   {name}: MAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f}\n")
        if ref and best:
            f.write(
                f"\n4. Delta MAE (persistencia multi-ano - {best_model}): "
                f"{ref['MAE'] - best['MAE']:.2f} t\n"
            )
        f.write(
            "\n5. Entrega operacional: projecao 2025–2026 "
            "(cenario, nao inventario oficial).\n"
        )
        f.write(
            "\n6. Tempo de inferencia (MinT classico + clip): "
            "ver resultados/tempo_inferencia.json "
            "(nucleo ~1.2 s; margem de poucos segundos).\n"
        )
        f.write(f"\nSeries: {quality.get('n_series')} | "
                f"periodo {quality.get('year_min')}–{quality.get('year_max')}\n")
    print(f"Leitura: {path}")


def plot_test_totals(hist_total, pred_totals, corte, path=None):
    ensure_dirs()
    path = path or os.path.join(PATHS['plots_dir'], 'total_nacional_teste.png')
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(10, 5))
    h = hist_total.sort_values('ano')
    plt.plot(h['ano'], h['y'], label='Observado', marker='o')
    for name, df in pred_totals.items():
        d = df.sort_values('ano')
        plt.plot(d['ano'], d['yhat'], label=name, linestyle='--', marker='s')
    plt.axvline(corte + 0.5, color='gray', linestyle=':', label=f'Corte {corte}')
    plt.xlabel('Ano')
    plt.ylabel('Emissao N2O total (t)')
    plt.title('Total nacional — observado vs previsoes multi-ano')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico: {path}")


def plot_projection(hist_total, proj_df, path=None):
    ensure_dirs()
    path = path or os.path.join(PATHS['plots_dir'], 'projecao_2025_2026.png')
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(10, 5))
    h = hist_total.sort_values('ano')
    plt.plot(h['ano'], h['y'], label='Historico', color='steelblue')
    p = proj_df.groupby('ano', as_index=False)['previsto'].sum()
    pers = proj_df.groupby('ano', as_index=False)['persistencia_2024'].sum()
    plt.plot(p['ano'], p['previsto'], 'o--', label='Projecao modelo', color='darkorange')
    plt.plot(pers['ano'], pers['persistencia_2024'], 's:', label='Persistencia 2024', color='gray')
    plt.axvline(2024.5, color='gray', linestyle=':')
    plt.xlabel('Ano')
    plt.ylabel('Emissao N2O total (t)')
    plt.title('Cenario 2025–2026 (nao e inventario oficial)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico projecao: {path}")


def save_projection_csv(leaf_proj, path=None):
    ensure_dirs()
    path = path or PATHS['projection']
    out = leaf_proj.copy()
    out['aviso'] = (
        'Cenario de curto prazo a partir do inventario ate 2024; '
        'nao substitui o SEEG oficial.'
    )
    out.to_csv(path, index=False)
    print(f"Projecao CSV: {path}")


def save_comparacao_real_previsto(
    merged,
    best_model,
    pers_df=None,
    path=None,
):
    """Tabela ano a ano: inventário SEEG (y) vs o que o modelo teria previsto."""
    ensure_dirs()
    path = path or PATHS['comparacao']
    df = merged.copy()
    if 'ano' not in df.columns:
        df['ano'] = pd.to_datetime(df['ds']).dt.year

    if best_model not in df.columns:
        raise KeyError(f"Coluna do modelo ausente: {best_model}")

    # Total nacional (soma das folhas)
    tot = (
        df.groupby('ano', as_index=False)
        .agg(real_seeg=('y', 'sum'), previsto=(best_model, 'sum'))
    )
    tot['erro'] = tot['previsto'] - tot['real_seeg']
    tot['erro_abs'] = tot['erro'].abs()
    tot['erro_pct'] = np.where(
        tot['real_seeg'] != 0,
        100.0 * tot['erro'] / tot['real_seeg'],
        np.nan,
    )
    tot['modelo'] = best_model

    if pers_df is not None and len(pers_df):
        p = pers_df.copy()
        if 'ano' not in p.columns:
            p['ano'] = pd.to_datetime(p['ds']).dt.year
        p_tot = p.groupby('ano', as_index=False)['persistencia'].sum()
        tot = tot.merge(p_tot, on='ano', how='left')
        tot = tot.rename(columns={'persistencia': 'persistencia_multi_ano'})
        tot['erro_persistencia'] = tot['persistencia_multi_ano'] - tot['real_seeg']

    tot.to_csv(path, index=False)
    print(f"Comparacao real vs previsto: {path}")

    # Gráfico barras + linha
    plot_path = os.path.join(PATHS['plots_dir'], 'comparacao_real_vs_previsto.png')
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = tot['ano'].astype(int)
    w = 0.35
    ax.bar(x - w / 2, tot['real_seeg'], width=w, label='Real (SEEG)', color='steelblue')
    ax.bar(x + w / 2, tot['previsto'], width=w, label=f'Previsto ({best_model})',
           color='darkorange')
    if 'persistencia_multi_ano' in tot.columns:
        ax.plot(
            x, tot['persistencia_multi_ano'], 's--', color='gray',
            label='Persistencia multi-ano',
        )
    ax.set_xlabel('Ano')
    ax.set_ylabel('Emissao N2O total (t)')
    ax.set_title('Anos ja inventariados: real SEEG vs previsao do modelo')
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grafico comparacao: {plot_path}")
    return tot


def save_model(fcst, path=None):
    ensure_dirs()
    path = path or PATHS['model']
    joblib.dump(fcst, path)
    print(f"Modelo: {path}")
