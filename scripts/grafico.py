# scripts/grafico.py — emissão média por ano (treino + teste + projeção 2025–2026)
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import MODEL_CONFIG, PATHS, ensure_dirs

plt.style.use('seaborn-v0_8-darkgrid')
palette = sns.color_palette("husl", 3)


def _load_historico():
    """Painel com ano / emissao / emissao_predita (histórico até 2024)."""
    path = PATHS['predictions_new']
    if not os.path.exists(path):
        path = PATHS['predictions']
    df = pd.read_csv(path)
    # Schema antigo
    if {'ano', 'emissao', 'emissao_predita'}.issubset(df.columns):
        return df[['ano', 'emissao', 'emissao_predita']].copy()
    # Schema multi-ano (só teste): y + melhor coluna Ridge/MinTrace
    if 'y' in df.columns and 'ano' in df.columns:
        pred_col = None
        for c in (
            'Ridge/MinTrace_method-ols',
            'Ridge/BottomUp',
            'Ridge',
        ):
            if c in df.columns:
                pred_col = c
                break
        if pred_col is None:
            raise ValueError('Nenhuma coluna de previsao encontrada.')
        out = df.rename(columns={'y': 'emissao', pred_col: 'emissao_predita'})
        return out[['ano', 'emissao', 'emissao_predita']].copy()
    raise ValueError(
        "Arquivo precisa de ('ano','emissao','emissao_predita') "
        "ou schema multi-ano com y + modelo."
    )


def _append_projecao(out_sample):
    """Acrescenta médias de 2025–2026 na série fora da amostra."""
    proj_path = PATHS['projection']
    if not os.path.exists(proj_path):
        print(f"Aviso: projecao nao encontrada ({proj_path}); grafico sem 2025–2026.")
        return out_sample

    proj = pd.read_csv(proj_path)
    if 'previsto' not in proj.columns or 'ano' not in proj.columns:
        print('Aviso: projecao sem colunas ano/previsto.')
        return out_sample

    fut = (
        proj.groupby('ano', as_index=False)['previsto']
        .mean()
        .rename(columns={'previsto': 'emissao_predita'})
    )
    fut = fut[fut['ano'].isin([2025, 2026])].copy()
    fut['emissao'] = float('nan')
    return pd.concat([out_sample, fut], ignore_index=True).sort_values('ano')


def main():
    print('Carregando dados...')
    try:
        df = _load_historico()
        print(f'Total de linhas carregadas: {len(df)}')
    except Exception as e:
        print(f'Erro ao carregar dados: {e}')
        sys.exit(1)

    print('\nProcessando dados...')
    corte = MODEL_CONFIG['split_year']

    df_grouped = df.groupby('ano', as_index=False).agg({
        'emissao': 'mean',
        'emissao_predita': 'mean',
    })

    in_sample = df_grouped[df_grouped['ano'] <= corte]
    out_sample = df_grouped[df_grouped['ano'] > corte].copy()
    out_sample = _append_projecao(out_sample)

    anos_xticks = sorted(
        set(df_grouped['ano'].astype(int).tolist())
        | set(out_sample['ano'].astype(int).tolist())
    )

    print('\nCriando visualizacao...')
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_grouped['ano'], df_grouped['emissao'],
        label='Emissão real',
        color=palette[0],
        linewidth=2.5,
        marker='o',
    )
    if not in_sample.empty:
        plt.plot(
            in_sample['ano'], in_sample['emissao_predita'],
            label='Prevista (dentro da amostra)',
            color=palette[1],
            linewidth=2.0,
            linestyle='--',
            marker='s',
        )
    if not out_sample.empty:
        plt.plot(
            out_sample['ano'], out_sample['emissao_predita'],
            label='Prevista (fora da amostra)',
            color=palette[2],
            linewidth=2.5,
            linestyle='--',
            marker='D',
        )

    plt.axvline(
        corte + 0.5,
        color='gray',
        linestyle=':',
        linewidth=1.5,
        label=f'Corte temporal ({corte}/{corte + 1})',
    )

    plt.title('Emissões médias de N2O (t) por ano', fontsize=16, pad=20)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Emissão média (t)', fontsize=12)
    plt.legend(fontsize=10)
    plt.xticks(anos_xticks, rotation=45)
    plt.grid(True, alpha=0.3)

    ensure_dirs()
    output_path = os.path.join(PATHS['plots_dir'], 'emissoes_por_ano.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nGrafico salvo em: {output_path}')
    print('Fora da amostra (anos):', sorted(out_sample['ano'].astype(int).unique().tolist()))


if __name__ == '__main__':
    main()
