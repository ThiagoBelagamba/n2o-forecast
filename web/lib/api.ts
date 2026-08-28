export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:3001';

export type ChartCategory = 'previsao' | 'avaliacao' | 'eda';

export interface ChartItem {
  id: string;
  title: string;
  category: ChartCategory;
  url: string;
  available: boolean;
}

export interface MetricsResponse {
  rows: { metodo: string; MAE: number; RMSE: number; R2: number; MedAE: number }[];
  best: { metodo: string; MAE: number; RMSE: number } | null;
  persistencia: { metodo: string; MAE: number; RMSE: number } | null;
  deltaMae: number | null;
}

export interface ComparacaoRow {
  ano: number;
  real_seeg: number;
  previsto: number;
  erro: number;
  erro_pct: number;
  modelo: string;
}

export interface ProjecaoResumo {
  ano: number;
  previsto_total: number;
  persistencia_2024_total: number;
}

async function getJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_URL}${path}`, { cache: 'no-store' });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export const fetchCharts = () => getJson<ChartItem[]>('/charts');
export const fetchMetrics = () => getJson<MetricsResponse>('/metrics');
export const fetchComparacao = () => getJson<ComparacaoRow[]>('/comparacao');
export const fetchProjecao = () => getJson<ProjecaoResumo[]>('/projecao-resumo');
