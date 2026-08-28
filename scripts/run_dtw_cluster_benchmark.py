# -*- coding: utf-8 -*-
# scripts/run_dtw_cluster_benchmark.py — Piloto DTW + modelos por cluster vs Ridge
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import DTW_CONFIG, MODEL_CONFIG, PATHS, RIDGE_MAE_BARRIER, ensure_dirs
from data_loading import load_and_analyze_data
from data_preprocessing import to_nixtla_panel
from forecasting import evaluate_on_test, fit_direct_models, load_lgbm_params, predict_horizon
from forecasting_dtw import (
    build_leaf_matrix,
    cluster_series_dtw,
    expand_and_reconcile_leaf_hat,
    fit_predict_per_cluster,
    merge_cluster_metadata,
    select_k_silhouette,
    transform_series_matrix,
)
from reconciliation import build_hierarchy, reconcile_forecasts


def _plot_clusters_pca(mat, uids, cluster_df, path):
    """Visualizacao 2D (PCA) das series normalizadas por cluster."""
    X = transform_series_matrix(mat)
    pca = PCA(n_components=2, random_state=DTW_CONFIG['random_state'])
    xy = pca.fit_transform(X)
    cdf = cluster_df.set_index('unique_id').loc[uids]['cluster'].values
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6))
    for cl in sorted(np.unique(cdf)):
        m = cdf == cl
        ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.6, label=f'cluster {cl}')
    ax.set_title('Folhas SEEG — PCA (series log1p+zscore) por cluster DTW')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Grafico clusters: {path}')


def main():
    ensure_dirs()
    corte = MODEL_CONFIG['split_year']
    horizon = MODEL_CONFIG['test_horizon']

    print('1. Painel + hierarquia...')
    df = load_and_analyze_data(PATHS['data'])
    panel, _ = to_nixtla_panel(df)
    Y_df, S_df, tags, _ = build_hierarchy(panel)
    leaf_key = list(tags.keys())[-1]
    leaf_ids = set(tags[leaf_key])

    print('\n2. Matriz folha + selecao K (silhouette DTW)...')
    mat, uids, _years = build_leaf_matrix(Y_df, leaf_ids, corte)
    best_k, sil_scores = select_k_silhouette(
        mat, uids,
        sample_n=DTW_CONFIG.get('silhouette_sample', 200),
    )
    print(f'   K selecionado: {best_k} | silhouettes: {sil_scores}')

    cluster_df = cluster_series_dtw(mat, uids, n_clusters=best_k)
    cluster_df = merge_cluster_metadata(cluster_df, panel)
    cluster_df.to_csv(PATHS['dtw_clusters'], index=False)
    print(f'   Clusters salvos: {PATHS["dtw_clusters"]}')
    print(cluster_df['cluster'].value_counts().sort_index())

    plot_path = Path(PATHS['plots_dir']) / 'dtw_clusters_pca.png'
    _plot_clusters_pca(mat, uids, cluster_df, plot_path)

    print('\n3. Treino por cluster + previsao h=5 (teste cego)...')
    Y_train = Y_df[Y_df['ds'].dt.year <= corte].copy()
    Y_test = Y_df[Y_df['ds'].dt.year > corte].copy()
    test_leaf = Y_test[Y_test['unique_id'].isin(leaf_ids)].copy()
    lgbm_params, _ = load_lgbm_params()

    leaf_hat = fit_predict_per_cluster(
        Y_train, leaf_ids, cluster_df, lgbm_params, horizon
    )

    print('\n4. Expansao hierarquica + MinT...')
    Y_rec = expand_and_reconcile_leaf_hat(
        leaf_hat, S_df, Y_df, tags, train_end_year=corte
    )

    eval_cols = []
    for m in DTW_CONFIG['models']:
        for suffix in ('', '/BottomUp', '/MinTrace_method-ols', '/MinTrace_method-mint_shrink'):
            c = f'{m}{suffix}' if suffix else m
            if c in Y_rec.columns:
                eval_cols.append(c)

    pred_leaf = Y_rec[Y_rec['unique_id'].isin(leaf_ids)].copy()
    mdict, _ = evaluate_on_test(pred_leaf, test_leaf, eval_cols)

    print('\n=== DTW cluster benchmark (folha reconciliada) ===')
    best_col, best_mae = None, float('inf')
    for name, m in sorted(mdict.items(), key=lambda x: x[1]['MAE']):
        print(f"  {name}: MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f}")
        if m['MAE'] < best_mae:
            best_mae, best_col = m['MAE'], name

    ridge_ref = RIDGE_MAE_BARRIER
    beat = best_mae < ridge_ref
    print(f'\nBarreira Ridge/MinTrace(ols): {ridge_ref:.2f} t')
    print(f'Melhor DTW-cluster: {best_col} MAE={best_mae:.2f} t')
    print(
        'DTW superou barreira Ridge.'
        if beat
        else f'DTW nao superou barreira (delta={ridge_ref - best_mae:+.2f} t).'
    )

    # Baseline Ridge global rapido
    print('\n5. Referencia Ridge global (mesmo protocolo)...')
    fcst = fit_direct_models(Y_train, lgbm_params, horizon)
    Y_hat_g = predict_horizon(fcst, h=horizon)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y_rec_g, _ = reconcile_forecasts(
            Y_hat_g, Y_df, S_df, tags, train_end_year=corte
        )
    ridge_col = 'Ridge/MinTrace_method-ols'
    pred_g = Y_rec_g[Y_rec_g['unique_id'].isin(leaf_ids)]
    m_g, _ = evaluate_on_test(pred_g, test_leaf, [ridge_col])
    ridge_mae = m_g[ridge_col]['MAE']
    print(f'   Ridge global {ridge_col}: MAE={ridge_mae:.2f}')

    out = {
        'k_clusters': int(best_k),
        'silhouette_scores': sil_scores,
        'best_dtw_model': best_col,
        'best_dtw_mae': best_mae,
        'ridge_global_mae': ridge_mae,
        'ridge_barrier': ridge_ref,
        'beat_barrier': beat,
        'delta_vs_barrier': ridge_ref - best_mae,
        'delta_vs_ridge_global': ridge_mae - best_mae,
    }
    with open(PATHS['dtw_benchmark'], 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    pd.DataFrame([{'metodo': k, **v} for k, v in mdict.items()]).to_csv(
        Path(PATHS['results_dir']) / 'metricas_dtw.csv', index=False
    )
    print(f'\nSalvo: {PATHS["dtw_benchmark"]}')
    return out


if __name__ == '__main__':
    main()
