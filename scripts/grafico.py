# scripts/grafico.py
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


def main():
    print("Carregando dados...")
    try:
        df = pd.read_csv(PATHS['predictions'])
        print(f"Total de linhas carregadas: {len(df)}")
        required = ['ano', 'emissao', 'emissao_predita']
        if not all(col in df.columns for col in required):
            raise ValueError(
                "Arquivo não contém as colunas necessárias "
                "('ano', 'emissao', 'emissao_predita')"
            )
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        sys.exit(1)

    print("\nProcessando dados...")
    corte = MODEL_CONFIG['split_year']

    df_grouped = df.groupby('ano', as_index=False).agg({
        'emissao': 'mean',
        'emissao_predita': 'mean',
    })

    in_sample = df_grouped[df_grouped['ano'] <= corte]
    out_sample = df_grouped[df_grouped['ano'] > corte]

    print("\nCriando visualização...")
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
    plt.xticks(df_grouped['ano'].unique(), rotation=45)
    plt.grid(True, alpha=0.3)

    ensure_dirs()
    output_path = os.path.join(PATHS['plots_dir'], 'emissoes_por_ano.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo em: {output_path}")


if __name__ == "__main__":
    main()
