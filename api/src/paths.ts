import { existsSync } from 'fs';
import { join } from 'path';

/** Resolve `resultados/` whether running from api/ (tsx) or api/dist (compiled). */
export function resultadosRoot(): string {
  const candidates = [
    join(process.cwd(), '..', 'resultados'),
    join(process.cwd(), 'resultados'),
    join(__dirname, '..', '..', 'resultados'),
    join(__dirname, '..', '..', '..', 'resultados'),
  ];
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return candidates[0];
}

export function graficosRoot(): string {
  return join(resultadosRoot(), 'graficos');
}
