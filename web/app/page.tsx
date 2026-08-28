import {
  API_URL,
  type ChartCategory,
  type ChartItem,
  fetchCenarioResumo,
  fetchCharts,
  fetchComparacao,
  fetchConformal,
  fetchExperimentos,
  fetchMetrics,
  fetchProjecao,
  fetchSetores,
  fetchSintese,
} from '@/lib/api';

const CATEGORY_ORDER: { key: ChartCategory; title: string; lead: string }[] = [
  {
    key: 'previsao',
    title: 'Previsão',
    lead: 'Série temporal, teste cego 2020–2024 e cenário 2025–2026.',
  },
  {
    key: 'avaliacao',
    title: 'Avaliação',
    lead: 'Erros por ano / setor e dispersão reais vs preditos.',
  },
  {
    key: 'eda',
    title: 'EDA',
    lead: 'Exploração inicial dos dados SEEG.',
  },
];

function fmt(n: number, digits = 1) {
  return n.toLocaleString('pt-BR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function pct(n: number | null) {
  return n != null ? fmt(n * 100, 1) + '%' : '—';
}

function ChartsByCategory({ charts }: { charts: ChartItem[] }) {
  return (
    <>
      {CATEGORY_ORDER.map((cat) => {
        const items = charts.filter((c) => c.category === cat.key && c.available);
        if (!items.length) return null;
        return (
          <section key={cat.key}>
            <h2>{cat.title}</h2>
            <p className="lead">{cat.lead}</p>
            {items.map((c) => (
              <figure key={c.id} className="chart-block">
                <h3>{c.title}</h3>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${API_URL}${c.url}`} alt={c.title} />
              </figure>
            ))}
          </section>
        );
      })}
    </>
  );
}

export default async function HomePage() {
  const [
    metrics,
    charts,
    comparacao,
    projecao,
    conformal,
    sintese,
    setores,
    cenario,
    experimentos,
  ] = await Promise.all([
    fetchMetrics(),
    fetchCharts(),
    fetchComparacao(),
    fetchProjecao(),
    fetchConformal(),
    fetchSintese(),
    fetchSetores(),
    fetchCenarioResumo(),
    fetchExperimentos(),
  ]);

  const apiDown = !metrics && !charts;
  const cenarioNacional = (cenario ?? []).filter((r) => r.escopo === 'nacional');

  return (
    <main>
      <header className="hero">
        <h1>N2O Forecast</h1>
        <p>
          Visualização dos resultados do pipeline multi-ano (SEEG Coleção 13):
          Ridge + MinTrace, teste cego 2020–2024, conformal e cenário 2025–2026.
        </p>
        <div className="notice">
          O cenário 2025–2026 <strong>não</strong> substitui o inventário oficial do SEEG.
        </div>
      </header>

      {apiDown && (
        <div className="error-banner">
          Não foi possível falar com a API em <code>{API_URL}</code>. Suba o Nest
          com <code>cd api; npm run start:dev</code> e recarregue.
        </div>
      )}

      {sintese && (
        <section>
          <h2>Síntese da entrega</h2>
          <div className="metrics">
            <div className="metric">
              <span className="label">Modelo campeão</span>
              <span className="value">{fmt(sintese.mae_campeao_main, 2)}</span>
              <span className="sub">{sintese.modelo_campeao_main}</span>
            </div>
            <div className="metric">
              <span className="label">Δ MAE vs persistência</span>
              <span className="value">
                {sintese.delta_mae_vs_persistencia != null
                  ? fmt(sintese.delta_mae_vs_persistencia, 2)
                  : '—'}
              </span>
              <span className="sub">positivo = modelo melhor</span>
            </div>
            <div className="metric">
              <span className="label">Barreira Ridge(ols)</span>
              <span className="value">{fmt(sintese.barreira_ridge_ols, 2)}</span>
              <span className="sub">referência histórica</span>
            </div>
            {sintese.conformal && (
              <div className="metric">
                <span className="label">Conformal folha 90%</span>
                <span className="value">
                  {pct(sintese.conformal.coverage_90_folha ?? null)}
                </span>
                <span className="sub">cobertura empírica</span>
              </div>
            )}
          </div>
        </section>
      )}

      {!sintese && metrics?.best && metrics.persistencia && (
        <div className="metrics">
          <div className="metric">
            <span className="label">Melhor modelo (MAE)</span>
            <span className="value">{fmt(metrics.best.MAE, 2)}</span>
            <span className="sub">{metrics.best.metodo}</span>
          </div>
          <div className="metric">
            <span className="label">Persistência multi-ano</span>
            <span className="value">{fmt(metrics.persistencia.MAE, 2)}</span>
            <span className="sub">copiar 2019 no horizonte cego</span>
          </div>
          <div className="metric">
            <span className="label">Δ MAE (ganho)</span>
            <span className="value">
              {metrics.deltaMae != null ? fmt(metrics.deltaMae, 2) : '—'}
            </span>
            <span className="sub">positivo = modelo melhor</span>
          </div>
        </div>
      )}

      {conformal && (
        <section>
          <h2>Incerteza (Conformal Prediction)</h2>
          <p className="lead">
            Cobertura empírica no teste cego — intervalos simétricos sobre Ridge/MinTrace.
          </p>
          <div className="metrics">
            <div className="metric">
              <span className="label">Folha — cobertura 90%</span>
              <span className="value">{pct(conformal.leaf.coverage90)}</span>
            </div>
            <div className="metric">
              <span className="label">Folha — cobertura 95%</span>
              <span className="value">{pct(conformal.leaf.coverage95)}</span>
            </div>
            <div className="metric">
              <span className="label">Nacional — cobertura 90%</span>
              <span className="value">{pct(conformal.national.coverage90)}</span>
            </div>
          </div>
        </section>
      )}

      {setores && setores.length > 0 && (
        <section>
          <h2>Setores (nível 1)</h2>
          <p className="lead">MAE no teste cego e cobertura conformal 90% por setor.</p>
          <table>
            <thead>
              <tr>
                <th>Setor</th>
                <th>MAE (t)</th>
                <th>RMSE (t)</th>
                <th>Cob. 90%</th>
                <th>Largura IC 90%</th>
              </tr>
            </thead>
            <tbody>
              {setores.map((r) => (
                <tr key={r.nivel_1}>
                  <td>{r.nivel_1}</td>
                  <td>{fmt(r.MAE, 2)}</td>
                  <td>{fmt(r.RMSE, 2)}</td>
                  <td>{pct(r.coverage90)}</td>
                  <td>{r.width90 != null ? fmt(r.width90, 0) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {comparacao && comparacao.length > 0 && (
        <section>
          <h2>Comparação nacional (teste)</h2>
          <p className="lead">Total SEEG vs previsto do melhor modelo, por ano.</p>
          <table>
            <thead>
              <tr>
                <th>Ano</th>
                <th>Real SEEG (t)</th>
                <th>Previsto (t)</th>
                <th>Erro %</th>
              </tr>
            </thead>
            <tbody>
              {comparacao.map((r) => (
                <tr key={r.ano}>
                  <td>{r.ano}</td>
                  <td>{fmt(r.real_seeg, 0)}</td>
                  <td>{fmt(r.previsto, 0)}</td>
                  <td>{fmt(r.erro_pct, 2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {cenarioNacional.length > 0 && (
        <section>
          <h2>Cenário 2025–2026 (com IC 90%)</h2>
          <p className="lead">
            Total nacional: ponto, faixa conformal e persistência 2024.
          </p>
          <table>
            <thead>
              <tr>
                <th>Ano</th>
                <th>Previsto (t)</th>
                <th>IC 90% inferior</th>
                <th>IC 90% superior</th>
                <th>Persistência 2024 (t)</th>
              </tr>
            </thead>
            <tbody>
              {cenarioNacional.map((r) => (
                <tr key={r.ano}>
                  <td>{r.ano}</td>
                  <td>{fmt(r.previsto, 0)}</td>
                  <td>{r.lo_90 != null ? fmt(r.lo_90, 0) : '—'}</td>
                  <td>{r.hi_90 != null ? fmt(r.hi_90, 0) : '—'}</td>
                  <td>
                    {r.persistencia_2024 != null
                      ? fmt(r.persistencia_2024, 0)
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {!cenarioNacional.length && projecao && projecao.length > 0 && (
        <section>
          <h2>Projeção 2025–2026 (totais)</h2>
          <p className="lead">Soma setorial do cenário e da persistência 2024.</p>
          <table>
            <thead>
              <tr>
                <th>Ano</th>
                <th>Previsto total (t)</th>
                <th>Persistência 2024 (t)</th>
              </tr>
            </thead>
            <tbody>
              {projecao.map((r) => (
                <tr key={r.ano}>
                  <td>{r.ano}</td>
                  <td>{fmt(r.previsto_total, 0)}</td>
                  <td>{fmt(r.persistencia_2024_total, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {experimentos && experimentos.length > 0 && (
        <section>
          <h2>Experimentos (platô MAE)</h2>
          <p className="lead">
            Tentativas que não superaram a barreira Ridge/MinTrace(ols).
          </p>
          <table>
            <thead>
              <tr>
                <th>Experimento</th>
                <th>MAE (t)</th>
                <th>Superou barreira?</th>
                <th>Δ vs barreira</th>
              </tr>
            </thead>
            <tbody>
              {experimentos.map((r) => (
                <tr key={r.experimento}>
                  <td>{r.experimento}</td>
                  <td>{fmt(r.MAE, 2)}</td>
                  <td>{r.superou_barreira ? 'sim' : 'não'}</td>
                  <td>{fmt(r.delta_vs_barreira, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {charts && <ChartsByCategory charts={charts} />}

      <p className="footer">
        Dados servidos por Nest em {API_URL} a partir de <code>resultados/</code>.
        Pipeline: <code>main.py</code> → <code>run_conformal.py</code> →{' '}
        <code>export_entrega.py</code>.
      </p>
    </main>
  );
}
