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

export interface ConformalResumo {
  protocol: string | null;
  national: {
    coverage90: number | null;
    coverage95: number | null;
    width90: number | null;
  };
  leaf: {
    coverage90: number | null;
    coverage95: number | null;
    width90: number | null;
  };
}

export interface SetorRow {
  nivel_1: string;
  n: number;
  MAE: number;
  RMSE: number;
  coverage90: number | null;
  coverage95: number | null;
  width90: number | null;
}

export interface CenarioResumoRow {
  escopo: string;
  nivel_1: string;
  ano: number;
  previsto: number;
  lo_90: number | null;
  hi_90: number | null;
  lo_95: number | null;
  hi_95: number | null;
  persistencia_2024: number | null;
}

export interface ExperimentoRow {
  experimento: string;
  MAE: number;
  barreira_ridge_ols: number;
  superou_barreira: boolean;
  delta_vs_barreira: number;
}

export interface SinteseEntrega {
  modelo_campeao_main: string;
  mae_campeao_main: number;
  barreira_ridge_ols: number;
  mae_ridge_ols: number | null;
  persistencia_mae: number | null;
  delta_mae_vs_persistencia: number | null;
  nota_modelo?: string;
  conformal?: {
    coverage_90_folha?: number;
    coverage_95_folha?: number;
    width_mean_90_folha?: number;
  };
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
export const fetchConformal = () => getJson<ConformalResumo>('/conformal-resumo');
export const fetchSetores = () => getJson<SetorRow[]>('/setores');
export const fetchCenarioResumo = () => getJson<CenarioResumoRow[]>('/cenario-resumo');
export const fetchExperimentos = () => getJson<ExperimentoRow[]>('/experimentos');
export const fetchSintese = () => getJson<SinteseEntrega>('/sintese');
