# -*- coding: utf-8 -*-
# results_saving.py
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MODEL_CONFIG, PATHS, PLOT_STYLE, ensure_dirs
from model_training import (
    _predict_persistence,
    _predict_persistence_1passo,
    compute_metrics,
)


def _primary_baseline_name(baseline_results):
    """Com lags, a referência justa é persistencia_1passo; senão, persistencia multi-ano."""
    if baseline_results and 'persistencia_1passo' in baseline_results:
        return 'persistencia_1passo'
    return 'persistencia'


def save_results(
    best_model,
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    num_cols,
    cat_cols,
    baseline_results=None,
    data_quality=None,
    model_metrics=None,
    best_model_name=None,
):
    """Salva modelo, predições, relatório e gráficos."""
    ensure_dirs()

    estimator = (
        best_model.best_estimator_ if hasattr(best_model, 'best_estimator_') else best_model
    )
    if hasattr(estimator, 'memory'):
        estimator.memory = None
    joblib.dump(estimator, PATHS['model'])
    save_predictions(best_model, X_train, y_train, X_test, y_test)

    if 'emissao_lag1' in X_test.columns:
        y_persist = _predict_persistence_1passo(X_test, y_train)
    else:
        y_persist = _predict_persistence(X_train, y_train, X_test)
    bias = compute_bias_evidence(X_test, y_test, y_pred, y_persist)
    save_bias_tables(bias)

    save_report(
        best_model,
        y_test,
        y_pred,
        baseline_results=baseline_results,
        data_quality=data_quality,
        model_metrics=model_metrics,
        best_model_name=best_model_name,
        bias=bias,
    )
    save_leitura_secao9(
        y_test,
        y_pred,
        baseline_results=baseline_results,
        data_quality=data_quality,
        model_metrics=model_metrics,
        best_model_name=best_model_name,
        bias=bias,
        model=best_model,
    )
    save_plots(best_model, X_test, y_test, y_pred, bias=bias)
    print(f"Arquivos salvos em '{PATHS['results_dir']}'")



def save_predictions(model, X_train, y_train, X_test, y_test):
    """Salva predições de treino e teste com coluna conjunto."""
    frames = []
    for nome, X, y in [('treino', X_train, y_train), ('teste', X_test, y_test)]:
        df = X.copy()
        df[MODEL_CONFIG['target_col']] = np.asarray(y)
        df['emissao_predita'] = np.round(model.predict(X), 8)
        df['conjunto'] = nome
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(PATHS['predictions'], index=False)


def _format_metrics_block(metrics):
    return (
        f"MAE:   {metrics['MAE']:.4f}\n"
        f"RMSE:  {metrics['RMSE']:.4f}\n"
        f"R2:    {metrics['R2']:.4f}\n"
        f"MedAE: {metrics['MedAE']:.4f}\n"
    )


def compute_bias_evidence(X_test, y_test, y_pred, y_persist):
    """Vieses no teste: assimetria, concentração e erro por faixa/setor/ano."""
    y_true = np.asarray(y_test, dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)
    y_p = np.asarray(y_persist, dtype=float)
    abs_err = np.abs(y_true - y_hat)
    abs_persist = np.abs(y_true - y_p)

    n = len(y_true)
    zeros = y_true == 0
    total_em = float(y_true.sum())
    total_err = float(abs_err.sum())
    k = max(1, int(np.ceil(0.1 * n)))
    top_idx = np.argsort(y_true)[-k:]
    pos = y_true[y_true > 0]
    median_pos = float(np.median(pos)) if len(pos) else 0.0
    p90 = float(np.percentile(y_true, 90))

    masks = [
        ('zero', zeros),
        ('baixo (positivo até a mediana dos positivos)', (y_true > 0) & (y_true <= median_pos)),
        ('medio (acima da mediana até o p90)', (y_true > median_pos) & (y_true <= p90)),
        ('alto (acima do p90)', y_true > p90),
    ]
    by_bin = []
    for name, mask in masks:
        if int(mask.sum()) == 0:
            continue
        by_bin.append({
            'faixa': name,
            'n': int(mask.sum()),
            'mae': float(abs_err[mask].mean()),
            'mae_persistencia': float(abs_persist[mask].mean()),
            'emissao_media': float(y_true[mask].mean()),
            'share_erro_pct': (100.0 * float(abs_err[mask].sum()) / total_err) if total_err else 0.0,
            'share_emissao_pct': (100.0 * float(y_true[mask].sum()) / total_em) if total_em else 0.0,
        })

    by_nivel = []
    if 'nivel_1' in X_test.columns:
        tmp = pd.DataFrame({
            'nivel_1': X_test['nivel_1'].astype(str).to_numpy(),
            'y': y_true,
            'err': abs_err,
            'err_p': abs_persist,
        })
        for nivel, g in tmp.groupby('nivel_1'):
            by_nivel.append({
                'nivel_1': nivel,
                'n': int(len(g)),
                'mae': float(g['err'].mean()),
                'mae_persistencia': float(g['err_p'].mean()),
                'emissao_media': float(g['y'].mean()),
                'share_erro_pct': (100.0 * float(g['err'].sum()) / total_err) if total_err else 0.0,
                'share_emissao_pct': (100.0 * float(g['y'].sum()) / total_em) if total_em else 0.0,
            })
        by_nivel.sort(key=lambda r: -r['mae'])

    by_year = []
    if 'ano' in X_test.columns:
        tmp = pd.DataFrame({
            'ano': X_test['ano'].to_numpy(),
            'y': y_true,
            'hat': y_hat,
            'err': abs_err,
            'err_p': abs_persist,
        })
        for ano, g in tmp.groupby('ano'):
            by_year.append({
                'ano': int(ano),
                'mae': float(g['err'].mean()),
                'mae_persistencia': float(g['err_p'].mean()),
                'emissao_media': float(g['y'].mean()),
                'pred_media': float(g['hat'].mean()),
            })

    return {
        'n_test': n,
        'zeros_pct': 100.0 * float(zeros.mean()),
        'mean': float(y_true.mean()),
        'median': float(np.median(y_true)),
        'max': float(y_true.max()),
        'p90': p90,
        'median_pos': median_pos,
        'share_em_top10': (100.0 * float(y_true[top_idx].sum()) / total_em) if total_em else 0.0,
        'share_err_top10': (100.0 * float(abs_err[top_idx].sum()) / total_err) if total_err else 0.0,
        'by_bin': by_bin,
        'by_nivel': by_nivel,
        'by_year': by_year,
    }


def save_bias_tables(bias):
    """CSV pequenos para a banca (não são o CSV de 26 mil linhas)."""
    pd.DataFrame(bias['by_bin']).to_csv(
        os.path.join(PATHS['results_dir'], 'vies_por_faixa.csv'), index=False
    )
    if bias['by_nivel']:
        pd.DataFrame(bias['by_nivel']).to_csv(
            os.path.join(PATHS['results_dir'], 'vies_por_nivel1.csv'), index=False
        )
    if bias['by_year']:
        pd.DataFrame(bias['by_year']).to_csv(
            os.path.join(PATHS['results_dir'], 'vies_por_ano.csv'), index=False
        )


def save_comparison_csv(model_metrics, baseline_results, best_model_name, final_metrics):
    rows = []
    if model_metrics:
        for name, metrics in model_metrics.items():
            row = {'metodo': name, 'tipo': 'modelo', **metrics}
            row['selecionado'] = name == best_model_name
            rows.append(row)
    if baseline_results:
        for name, metrics in baseline_results.items():
            rows.append({'metodo': name, 'tipo': 'baseline', **metrics, 'selecionado': False})
    if not rows:
        rows.append({'metodo': best_model_name or 'modelo', 'tipo': 'modelo', **final_metrics, 'selecionado': True})
    pd.DataFrame(rows).to_csv(
        os.path.join(PATHS['results_dir'], 'metricas_comparacao.csv'), index=False
    )


def _write_relevance_section(f, final_metrics, baseline_results, model_metrics, best_model_name):
    corte = MODEL_CONFIG['split_year']
    ref_name = _primary_baseline_name(baseline_results)
    persist = (baseline_results or {}).get(ref_name)
    f.write("=== RELEVÂNCIA DO DESEMPENHO (manual §9.2) ===\n")
    f.write(
        f"Pergunta: com o histórico até t−1, o modelo prevê o ano t "
        f"(teste > {corte}) melhor do que copiar t−1?\n"
        f"Referência justa com lags: {ref_name}. "
        "Também reportamos persistência multi-ano (último valor do treino).\n\n"
    )
    if model_metrics:
        for name, metrics in model_metrics.items():
            mark = " ← selecionado por CV" if name == best_model_name else ""
            f.write(
                f"{name}{mark}: MAE={metrics['MAE']:.4f} t | "
                f"RMSE={metrics['RMSE']:.4f} t | "
                f"MedAE={metrics['MedAE']:.4f} t | R²={metrics['R2']:.4f}\n"
            )
    if baseline_results:
        for bname in ('persistencia_1passo', 'persistencia'):
            if bname in baseline_results:
                m = baseline_results[bname]
                f.write(
                    f"{bname}: MAE={m['MAE']:.4f} t | "
                    f"RMSE={m['RMSE']:.4f} t | "
                    f"MedAE={m['MedAE']:.4f} t | R²={m['R2']:.4f}\n"
                )
    if persist:
        mae_delta = persist['MAE'] - final_metrics['MAE']
        rmse_delta = persist['RMSE'] - final_metrics['RMSE']
        mae_pct = (100.0 * mae_delta / persist['MAE']) if persist['MAE'] else 0.0
        f.write(
            f"\nΔ MAE  ({ref_name} − {best_model_name}): "
            f"{mae_delta:.4f} t ({mae_pct:+.2f}%)\n"
        )
        f.write(
            f"Δ RMSE ({ref_name} − {best_model_name}): {rmse_delta:.4f} t\n"
        )
        if mae_delta > 1e-6:
            f.write(
                "\nConclusão: o modelo erra menos que a referência em MAE no teste. "
                "Há ganho preditivo mensurável com lags + categorias.\n"
            )
        elif mae_delta < -1e-6:
            f.write(
                "\nConclusão: o modelo NÃO supera a referência em MAE. "
                "Copiar t−1 (ou o último ano do treino) é igual ou melhor neste recorte.\n"
            )
        else:
            f.write(
                "\nConclusão: empate prático com a referência em MAE.\n"
            )
    f.write("\n")


def _write_interpretation(f, final_metrics, baseline_results, bias):
    ref_name = _primary_baseline_name(baseline_results)
    persist = (baseline_results or {}).get(ref_name)
    corte = MODEL_CONFIG['split_year']
    f.write(f"=== INTERPRETAÇÃO DOS NÚMEROS (TESTE > {corte}) ===\n")
    f.write(
        f"n={bias['n_test']} linhas. "
        f"Média={bias['mean']:.2f} t, mediana={bias['median']:.4f} t, "
        f"máximo={bias['max']:.2f} t, p90={bias['p90']:.2f} t.\n"
    )
    f.write(
        f"MAE={final_metrics['MAE']:.4f} t e MedAE={final_metrics['MedAE']:.4f} t. "
    )
    if final_metrics['MedAE'] < 0.1 * max(final_metrics['MAE'], 1e-9):
        f.write(
            "MedAE << MAE: o erro típico (mediana) é pequeno; o MAE é puxado "
            "pela cauda dos grandes emissores, não pelo setor mediano.\n"
        )
    else:
        f.write(
            "A distância entre MedAE e MAE descreve o peso da cauda no erro médio.\n"
        )
    f.write(
        f"R²={final_metrics['R2']:.4f} no teste cronológico. "
        "Não comparar com R² de split aleatório.\n"
    )
    if persist:
        f.write(
            f"Referência {ref_name}: MAE={persist['MAE']:.4f} t, "
            f"RMSE={persist['RMSE']:.4f} t.\n"
        )
    f.write("\n")


def _write_bias_section(f, bias, data_quality):
    f.write("=== VIESES COM EVIDÊNCIA (TESTE) ===\n")
    if data_quality:
        f.write(
            f"No recorte N2O completo: {data_quality['target_zeros_pct']:.2f}% zeros, "
            f"média {data_quality['target_mean']:.2f} t vs mediana "
            f"{data_quality['target_median']:.4f} t, máximo "
            f"{data_quality.get('target_max', float('nan')):.2f} t. "
            f"Os 10% maiores registros concentram "
            f"{data_quality.get('share_em_top10_pct', float('nan')):.1f}% da emissão.\n"
        )
    f.write(
        f"No teste: {bias['zeros_pct']:.2f}% zeros. "
        f"Os 10% maiores valores reais concentram {bias['share_em_top10']:.1f}% da "
        f"emissão e {bias['share_err_top10']:.1f}% do erro absoluto do modelo.\n"
        "Isso confirma o viés de escala: o MAE é decidido por poucos grandes emissores.\n\n"
    )
    f.write("Erro por faixa de emissão:\n")
    for row in bias['by_bin']:
        f.write(
            f"  - {row['faixa']}: n={row['n']}, MAE={row['mae']:.4f} t "
            f"(persistência {row['mae_persistencia']:.4f} t), "
            f"{row['share_emissao_pct']:.1f}% da emissão, "
            f"{row['share_erro_pct']:.1f}% do erro.\n"
        )
    if bias['by_nivel']:
        f.write("\nErro por nivel_1 (maior MAE primeiro):\n")
        for row in bias['by_nivel']:
            f.write(
                f"  - {row['nivel_1']}: n={row['n']}, MAE={row['mae']:.4f} t "
                f"(persistência {row['mae_persistencia']:.4f} t), "
                f"{row['share_emissao_pct']:.1f}% da emissão.\n"
            )
    if bias['by_year']:
        f.write("\nMAE por ano do teste (modelo vs persistência):\n")
        for row in bias['by_year']:
            f.write(
                f"  - {row['ano']}: modelo {row['mae']:.4f} t | "
                f"persistência {row['mae_persistencia']:.4f} t | "
                f"emissão média {row['emissao_media']:.2f} t | "
                f"previsão média {row['pred_media']:.2f} t\n"
            )
        first, last = bias['by_year'][0], bias['by_year'][-1]
        if last['mae'] > first['mae'] * 1.1:
            f.write(
                "\nO MAE cresce ao longo do horizonte de teste: "
                "o erro aumenta nos anos mais distantes do corte.\n"
            )
        elif last['mae'] < first['mae'] * 0.9:
            f.write("\nO MAE não aumenta de forma sistemática no horizonte de teste.\n")
        else:
            f.write(
                "\nO MAE permanece da mesma ordem ao longo do horizonte de teste.\n"
            )
    f.write("\n")


def save_report(
    model,
    y_test,
    y_pred,
    baseline_results=None,
    data_quality=None,
    model_metrics=None,
    best_model_name=None,
    bias=None,
):
    """Salva relatório de métricas, qualidade de dados e comparação justa."""
    final_metrics = compute_metrics(y_test, y_pred)
    save_comparison_csv(model_metrics, baseline_results, best_model_name, final_metrics)

    with open(PATHS['report'], 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DO MODELO ===\n\n")
        f.write(
            f"Split cronológico: treino anos <= {MODEL_CONFIG['split_year']}, "
            f"teste anos > {MODEL_CONFIG['split_year']}\n"
        )
        f.write(f"Validação: TimeSeriesSplit (n_splits={MODEL_CONFIG['cv_splits']})\n")
        f.write(f"Scoring do grid: {MODEL_CONFIG['scoring']}\n\n")

        if data_quality:
            f.write("=== QUALIDADE DOS DADOS (N2O) ===\n")
            f.write(f"Linhas após filtro de gás: {data_quality['n_raw']}\n")
            f.write(
                f"Alvos nulos descartados: {data_quality['n_dropped_target']}\n"
            )
            f.write(f"Linhas usadas na modelagem: {data_quality['n_clean']}\n")
            for col, pct in data_quality.get('missing_pct', {}).items():
                f.write(f"Ausência em {col}: {pct:.2f}%\n")
            f.write(f"Zeros no alvo: {data_quality['target_zeros_pct']:.2f}%\n")
            f.write(f"Mediana do alvo: {data_quality['target_median']:.4f} t\n")
            f.write(f"Média do alvo: {data_quality['target_mean']:.4f} t\n")
            if 'target_max' in data_quality:
                f.write(f"Máximo do alvo: {data_quality['target_max']:.4f} t\n")
                f.write(f"p90 do alvo: {data_quality['target_p90']:.4f} t\n")
                f.write(f"p99 do alvo: {data_quality['target_p99']:.4f} t\n")
                f.write(
                    f"Emissão nos 10% maiores registros: "
                    f"{data_quality['share_em_top10_pct']:.2f}%\n"
                )
            f.write(
                f"Período: {data_quality['year_min']}–{data_quality['year_max']}\n\n"
            )

        nome = best_model_name or 'modelo'
        f.write(f"Modelo selecionado (menor MAE de CV): {nome}\n")
        if hasattr(model, 'best_params_'):
            f.write(f"Melhores parâmetros: {model.best_params_}\n")
            f.write(f"MAE de CV: {-model.best_score_:.4f}\n")
        f.write("\nDesempenho no teste:\n")
        f.write(_format_metrics_block(final_metrics))

        if model_metrics:
            f.write("\n=== MODELOS COMPARADOS NO TESTE ===\n")
            for name, metrics in model_metrics.items():
                f.write(f"\n{name}\n")
                f.write(_format_metrics_block(metrics))

        if baseline_results:
            f.write("\n=== COMPARAÇÃO COM BASELINES (TESTE) ===\n")
            for name, metrics in baseline_results.items():
                f.write(f"\n{name}\n")
                f.write(_format_metrics_block(metrics))

            persist = baseline_results.get(_primary_baseline_name(baseline_results))
            persist_multi = baseline_results.get('persistencia')
            if persist:
                mae_delta = persist['MAE'] - final_metrics['MAE']
                rmse_delta = persist['RMSE'] - final_metrics['RMSE']
                ref = _primary_baseline_name(baseline_results)
                f.write(f"\n=== GANHO VERSUS {ref.upper()} ===\n")
                f.write(
                    "Valores positivos = o modelo erra menos que a referência.\n"
                    "Com lags, a referência justa é persistencia_1passo (copiar t−1).\n"
                )
                f.write(f"Δ MAE  ({ref} − modelo): {mae_delta:.4f} t\n")
                f.write(f"Δ RMSE ({ref} − modelo): {rmse_delta:.4f} t\n")
                if persist_multi and ref != 'persistencia':
                    d2 = persist_multi['MAE'] - final_metrics['MAE']
                    f.write(
                        f"Δ MAE  (persistencia multi-ano − modelo): {d2:.4f} t\n"
                    )
                if mae_delta <= 0:
                    f.write(
                        "\nO modelo não supera a referência em MAE neste teste.\n"
                    )
                f.write("\n")

        _write_relevance_section(
            f, final_metrics, baseline_results, model_metrics, nome
        )
        if bias:
            _write_interpretation(f, final_metrics, baseline_results, bias)
            _write_bias_section(f, bias, data_quality)


def save_leitura_secao9(
    y_test,
    y_pred,
    baseline_results=None,
    data_quality=None,
    model_metrics=None,
    best_model_name=None,
    bias=None,
    model=None,
):
    """Página curta da seção 9.2: alvo, split, referência, tabela e limitação."""
    final_metrics = compute_metrics(y_test, y_pred)
    ref_name = _primary_baseline_name(baseline_results)
    persist = (baseline_results or {}).get(ref_name, {})
    corte = MODEL_CONFIG['split_year']
    path = os.path.join(PATHS['results_dir'], 'leitura-secao9.txt')
    nome = best_model_name or 'modelo'
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Previsão cronológica de N2O (t) — evidência da seção 9.2\n")
        f.write("Thiago Belagamba Bueno — mecanismo preditivo\n")
        f.write("Código: https://github.com/ThiagoBelagamba/n2o-forecast\n\n")
        f.write("1. Alvo e fonte\n")
        f.write("   Emissão de N2O em toneladas, SEEG Coleção 13 (1970–2024), agregado nacional.\n")
        f.write(
            "   Features: ano + hierarquia setorial + lags "
            "(emissao_lag1, emissao_lag2, delta_lag1).\n\n"
        )
        f.write("2. Split (anti-vazamento)\n")
        f.write(
            f"   Treino até {corte}, teste {corte + 1}–2024. "
            "TimeSeriesSplit (5 folds) no treino.\n"
        )
        f.write("   Lags usam só o passado; no teste lag1 de t usa a emissão observada de t−1.\n\n")
        f.write("3. Referência\n")
        f.write(
            f"   {ref_name}: com lags, copiar t−1 (1 passo). "
            "Também reportamos persistência multi-ano (último ano do treino).\n\n"
        )
        f.write(f"4. Métricas no teste > {corte}\n")
        if model_metrics:
            for name, metrics in model_metrics.items():
                tag = " (escolhido por MAE de CV)" if name == nome else ""
                f.write(
                    f"   {name}{tag}: MAE {metrics['MAE']:.2f} t | "
                    f"RMSE {metrics['RMSE']:.2f} t | "
                    f"MedAE {metrics['MedAE']:.4f} t | R² {metrics['R2']:.4f}\n"
                )
        f.write(
            f"   {ref_name}: MAE {persist.get('MAE', float('nan')):.2f} t | "
            f"RMSE {persist.get('RMSE', float('nan')):.2f} t\n"
        )
        if persist:
            delta = persist['MAE'] - final_metrics['MAE']
            f.write(f"   Δ MAE ({ref_name} − {nome}): {delta:.2f} t\n")
        if hasattr(model, 'best_params_'):
            f.write(f"   Hiperparâmetros: {model.best_params_}\n")
        f.write("\n5. Limitação\n")
        f.write(
            "   Previsão de um passo (inventário anual). "
            "Não é projeção multi-ano sem observar o intermediário.\n"
        )
        if persist:
            delta = persist['MAE'] - final_metrics['MAE']
            if delta <= 0:
                f.write(
                    "   Neste experimento o modelo não ganha da referência em MAE; "
                    "o método e o teto estão medidos.\n"
                )
            else:
                f.write("   Há ganho frente à referência de 1 passo; a cauda ainda concentra o erro.\n")
        if bias:
            f.write(
                f"\n6. Viés (evidência no teste)\n"
                f"   {bias['zeros_pct']:.1f}% zeros; média {bias['mean']:.1f} t vs "
                f"mediana {bias['median']:.4f} t.\n"
                f"   Top 10% das linhas: {bias['share_em_top10']:.1f}% da emissão e "
                f"{bias['share_err_top10']:.1f}% do erro absoluto.\n"
            )
        f.write("\nArquivos: relatorio.txt, graficos/, modelo_n2o.joblib, metricas_comparacao.csv\n")


def save_plots(model, X_test, y_test, y_pred, bias=None):
    """Salva gráficos de avaliação."""
    plt.style.use(PLOT_STYLE)
    y_true = np.asarray(y_test, dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_hat, alpha=0.5)
    y_min = min(np.min(y_true), np.min(y_hat))
    y_max = max(np.max(y_true), np.max(y_hat))
    plt.plot([y_min, y_max], [y_min, y_max], '--r')
    plt.xlabel("Valores Reais (t)")
    plt.ylabel("Valores Preditos (t)")
    plt.title("Valores Reais vs Preditos — conjunto de teste")
    path = os.path.join(PATHS['plots_dir'], 'reais_vs_preditos.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_hat, alpha=0.4)
    plt.plot([y_min, y_max], [y_min, y_max], '--r')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel("Valores reais (t, escala simlog)")
    plt.ylabel("Valores preditos (t, escala simlog)")
    plt.title("Reais vs preditos (escala simlog) — teste")
    path = os.path.join(PATHS['plots_dir'], 'reais_vs_preditos_simlog.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    residuos = y_hat - y_true
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, residuos, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valor real (t)")
    plt.ylabel("Resíduo (predito − real, t)")
    plt.title("Resíduos no teste")
    path = os.path.join(PATHS['plots_dir'], 'residuos_teste.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    if bias:
        _plot_bias(bias)

    save_feature_importance_plot(model)


def _plot_bias(bias):
    if bias['by_bin']:
        df = pd.DataFrame(bias['by_bin'])
        x = np.arange(len(df))
        width = 0.35
        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, df['mae'], width, label='Modelo')
        plt.bar(x + width / 2, df['mae_persistencia'], width, label='Persistência 1 passo')
        plt.xticks(x, [s.split(' (')[0] for s in df['faixa']], rotation=0)
        plt.ylabel("MAE (t)")
        plt.title("MAE por faixa de emissão — teste")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['plots_dir'], 'mae_por_faixa.png'), dpi=150, bbox_inches='tight')
        plt.close()

    if bias['by_nivel']:
        df = pd.DataFrame(bias['by_nivel']).sort_values('mae', ascending=True)
        y = np.arange(len(df))
        height = 0.35
        plt.figure(figsize=(10, max(4, 0.45 * len(df))))
        plt.barh(y - height / 2, df['mae'], height, label='Modelo')
        plt.barh(y + height / 2, df['mae_persistencia'], height, label='Persistência 1 passo')
        plt.yticks(y, df['nivel_1'])
        plt.xlabel("MAE (t)")
        plt.title("MAE por nivel_1 — teste")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['plots_dir'], 'mae_por_nivel1.png'), dpi=150, bbox_inches='tight')
        plt.close()

    if bias['by_year']:
        df = pd.DataFrame(bias['by_year'])
        plt.figure(figsize=(8, 5))
        plt.plot(df['ano'], df['mae'], marker='o', label='Modelo')
        plt.plot(df['ano'], df['mae_persistencia'], marker='s', linestyle='--', label='Persistência 1 passo')
        plt.xlabel("Ano")
        plt.ylabel("MAE (t)")
        plt.title("MAE anual no teste vs persistência 1 passo")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['plots_dir'], 'mae_por_ano_teste.png'), dpi=150, bbox_inches='tight')
        plt.close()


def save_feature_importance_plot(model):
    """Salva gráfico de importância das features via get_feature_names_out."""
    try:
        estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        preprocessor = estimator.named_steps['preprocessor']
        inner = estimator.named_steps['model']

        if not hasattr(inner, 'feature_importances_'):
            print("Modelo sem feature_importances_; gráfico de importância omitido.")
            return

        feature_names = preprocessor.get_feature_names_out()
        importances = inner.feature_importances_
        n = min(20, len(importances))
        indices = np.argsort(importances)[-n:]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top {n} — Importância das Features")
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importância")
        path = os.path.join(PATHS['plots_dir'], 'feature_importances.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Não foi possível plotar feature importances: {e}")


if __name__ == "__main__":
    print("Funções disponíveis neste módulo:")
    print("- save_results")
    print("- save_predictions")
    print("- save_report")
    print("- save_plots")
    print("- save_feature_importance_plot")
    print("- compute_bias_evidence")
