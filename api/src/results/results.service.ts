import { Injectable, NotFoundException } from '@nestjs/common';
import { existsSync, readFileSync } from 'fs';
import { join } from 'path';
import { parse } from 'csv-parse/sync';
import { graficosRoot, resultadosRoot } from '../paths';

export type ChartCategory = 'previsao' | 'avaliacao' | 'eda';

export interface ChartItem {
  id: string;
  title: string;
  category: ChartCategory;
  url: string;
  available: boolean;
}

const CHART_CATALOG: Omit<ChartItem, 'url' | 'available'>[] = [
  {
    id: 'emissoes_por_ano',
    title: 'Emissões médias por ano (com 2025–2026)',
    category: 'previsao',
  },
  {
    id: 'comparacao_real_vs_previsto',
    title: 'Comparação real SEEG vs previsto (2020–2024)',
    category: 'previsao',
  },
  {
    id: 'total_nacional_teste',
    title: 'Total nacional — teste multi-ano',
    category: 'previsao',
  },
  {
    id: 'projecao_2025_2026',
    title: 'Projeção 2025–2026',
    category: 'previsao',
  },
  {
    id: 'mae_por_ano_teste',
    title: 'MAE por ano (teste)',
    category: 'avaliacao',
  },
  {
    id: 'mae_por_nivel1',
    title: 'MAE por nível 1',
    category: 'avaliacao',
  },
  {
    id: 'reais_vs_preditos',
    title: 'Reais vs preditos',
    category: 'avaliacao',
  },
  {
    id: 'eda_distribuicao_emissao',
    title: 'EDA — distribuição da emissão',
    category: 'eda',
  },
  {
    id: 'eda_concentracao_emissao',
    title: 'EDA — concentração da emissão',
    category: 'eda',
  },
  {
    id: 'eda_valores_ausentes',
    title: 'EDA — valores ausentes',
    category: 'eda',
  },
  {
    id: 'eda_correlacao',
    title: 'EDA — correlação',
    category: 'eda',
  },
];

@Injectable()
export class ResultsService {
  listCharts(): ChartItem[] {
    const dir = graficosRoot();
    return CHART_CATALOG.map((c) => {
      const file = `${c.id}.png`;
      const available = existsSync(join(dir, file));
      return {
        ...c,
        url: `/static/graficos/${file}`,
        available,
      };
    });
  }

  readMetrics() {
    const rows = this.readCsv('metricas_comparacao.csv');
    const parsed = rows.map((r) => ({
      metodo: String(r.metodo ?? ''),
      MAE: Number(r.MAE),
      RMSE: Number(r.RMSE),
      R2: Number(r.R2),
      MedAE: Number(r.MedAE),
    }));
    const pers = parsed.find((m) => m.metodo === 'persistencia_multi_ano');
    const models = parsed.filter((m) => m.metodo !== 'persistencia_multi_ano');
    const best = models.length
      ? models.reduce((a, b) => (a.MAE <= b.MAE ? a : b))
      : null;
    return {
      rows: parsed,
      best,
      persistencia: pers ?? null,
      deltaMae:
        best && pers ? pers.MAE - best.MAE : null,
    };
  }

  readComparacao() {
    const rows = this.readCsv('comparacao_real_vs_previsto.csv');
    return rows.map((r) => ({
      ano: Number(r.ano),
      real_seeg: Number(r.real_seeg),
      previsto: Number(r.previsto),
      erro: Number(r.erro),
      erro_pct: Number(r.erro_pct),
      modelo: String(r.modelo ?? ''),
      persistencia_multi_ano: r.persistencia_multi_ano
        ? Number(r.persistencia_multi_ano)
        : null,
    }));
  }

  readProjecaoResumo() {
    const rows = this.readCsv('projecao_2025_2026.csv');
    const byYear = new Map<number, { previsto: number; persistencia: number }>();
    for (const r of rows) {
      const ano = Number(r.ano);
      if (!byYear.has(ano)) {
        byYear.set(ano, { previsto: 0, persistencia: 0 });
      }
      const acc = byYear.get(ano)!;
      acc.previsto += Number(r.previsto) || 0;
      acc.persistencia += Number(r.persistencia_2024) || 0;
    }
    return Array.from(byYear.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([ano, v]) => ({
        ano,
        previsto_total: v.previsto,
        persistencia_2024_total: v.persistencia,
      }));
  }

  private readCsv(filename: string): Record<string, string>[] {
    const path = join(resultadosRoot(), filename);
    if (!existsSync(path)) {
      throw new NotFoundException(`Arquivo nao encontrado: ${filename}`);
    }
    const text = readFileSync(path, 'utf-8');
    return parse(text, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
    }) as Record<string, string>[];
  }
}
