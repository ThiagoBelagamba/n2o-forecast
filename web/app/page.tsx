import {
  API_URL,
  type ChartCategory,
  type ChartItem,
  fetchCharts,
  fetchComparacao,
  fetchMetrics,
  fetchProjecao,
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
  const [metrics, charts, comparacao, projecao] = await Promise.all([
    fetchMetrics(),
    fetchCharts(),
    fetchComparacao(),
    fetchProjecao(),
  ]);

  const apiDown = !metrics && !charts;

  return (
    <main>
      <header className="hero">
        <h1>N2O Forecast</h1>
        <p>
          Visualização mínima dos resultados do pipeline multi-ano (SEEG Coleção 13):
          Ridge + MinTrace, teste cego 2020–2024 e projeção 2025–2026.
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

      {metrics?.best && metrics.persistencia && (
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

      {projecao && projecao.length > 0 && (
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

      {charts && <ChartsByCategory charts={charts} />}

      <p className="footer">
        Dados servidos por Nest em {API_URL} a partir de <code>resultados/</code>.
        Pipeline Python: <code>main.py</code>.
      </p>
    </main>
  );
}
