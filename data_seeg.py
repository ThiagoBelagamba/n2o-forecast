# -*- coding: utf-8 -*-
# data_seeg.py — carga SEEG Coleção 13 (xlsx largo → painel nacional longo)
import os

import pandas as pd

from config import (
    MODEL_CONFIG,
    PATHS,
    SEEG_C13_COLUMNS,
    SEEG_C13_GROUP_COLS,
)


def _year_columns(df):
    years = []
    for c in df.columns:
        try:
            y = int(c)
        except (TypeError, ValueError):
            continue
        if 1900 <= y <= 2100:
            years.append(c)
    return sorted(years, key=lambda x: int(x))


def load_seeg_collection13(xlsx_path=None, cache_path=None, use_cache=True):
    """
    Lê Dados-nacionais-13.0.xlsx, filtra N2O + Emissão, melt 1970–2024,
    agrega UF/bioma ao nacional e grava cache CSV.
    """
    xlsx_path = xlsx_path or PATHS['data']
    cache_path = cache_path or PATHS['data_cache']

    if use_cache and os.path.isfile(cache_path):
        print(f"Cache encontrado: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"Linhas no cache: {len(df)}")
        return df

    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(
            f"Dataset não encontrado em:\n  {xlsx_path}\n"
            "Coloque Dados-nacionais-13.0.xlsx em data/ "
            "(o arquivo não vai no Git)."
        )

    print(f"Lendo Coleção 13: {xlsx_path}")
    wide = pd.read_excel(xlsx_path, sheet_name='Dados', engine='openpyxl')
    print(f"Planilha larga: {wide.shape[0]} linhas × {wide.shape[1]} colunas")

    missing = [c for c in SEEG_C13_COLUMNS if c not in wide.columns]
    if missing:
        raise ValueError(
            f"Colunas esperadas ausentes no xlsx: {missing}. "
            f"Encontradas: {list(wide.columns)[:20]}..."
        )

    gas = MODEL_CONFIG['gas_type']
    tipo = MODEL_CONFIG['emission_type']
    tipo_col = 'Emissão/Remoção/Bunker'
    gas_col = 'Gás'

    mask = (wide[gas_col] == gas) & (wide[tipo_col] == tipo)
    wide = wide.loc[mask].copy()
    print(f"Após filtro {gas} + {tipo}: {len(wide)} linhas")

    if len(wide) == 0:
        raise ValueError(
            f"Nenhuma linha para gás={gas!r} e tipo={tipo!r}. "
            "Verifique os rótulos na planilha."
        )

    year_cols = _year_columns(wide)
    if not year_cols:
        raise ValueError("Nenhuma coluna de ano (1970–2024) encontrada.")

    keep = list(SEEG_C13_COLUMNS.keys()) + year_cols
    wide = wide[keep].rename(columns=SEEG_C13_COLUMNS)

    id_vars = [SEEG_C13_COLUMNS[c] for c in SEEG_C13_COLUMNS]
    long = wide.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name='ano',
        value_name=MODEL_CONFIG['target_col'],
    )
    long['ano'] = long['ano'].astype(int)
    long[MODEL_CONFIG['target_col']] = pd.to_numeric(
        long[MODEL_CONFIG['target_col']], errors='coerce'
    )

    # Agrega Estado/Bioma (já removidos) → série nacional por setor
    group_cols = [c for c in SEEG_C13_GROUP_COLS if c in long.columns]
    national = (
        long.groupby(group_cols + ['ano'], as_index=False, dropna=False)
        [MODEL_CONFIG['target_col']]
        .sum(min_count=1)
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    national.to_csv(cache_path, index=False)
    print(
        f"Painel nacional: {len(national)} linhas "
        f"({national['ano'].min()}–{national['ano'].max()}), "
        f"cache em {cache_path}"
    )
    return national
