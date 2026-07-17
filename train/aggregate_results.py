# -*- coding: utf-8 -*-
"""Aggregate 5-fold cross-validation results into a single summary CSV."""
import os, json, glob, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(os.path.dirname(_HERE), 'train_results')

TISSUES = ['Heart', 'Kidney', 'Muscle', 'Pancreas', 'Spleen']

all_rows = []

for tissue in TISSUES:
    pattern = os.path.join(_RESULTS_DIR, f'human_{tissue}_fold*', 'metrics.json')
    files = sorted(glob.glob(pattern))
    tissue_rows = []
    for fp in files:
        m = json.load(open(fp))
        fold = m.get('fold', '?')
        tissue_rows.append({
            'tissue': tissue,
            'fold': fold,
            'auc': m.get('auc', 0),
            'aupr': m.get('aupr', 0),
            'acc': m.get('acc', 0),
            'f1': m.get('f1', 0),
            'precision': m.get('precision', 0),
            'recall': m.get('recall', 0),
            'train_edges': m.get('train_edges', 0),
            'val_edges': m.get('val_edges', 0),
            'rna_nodes': m.get('rna_nodes', 0),
            'atac_nodes': m.get('atac_nodes', 0),
            'best_epoch': m.get('best_epoch', 0),
        })
    all_rows.extend(tissue_rows)

    # Per-tissue mean
    if tissue_rows:
        df_t = pd.DataFrame(tissue_rows)
        means = df_t[['auc', 'aupr', 'acc', 'f1']].mean().round(4)
        stds = df_t[['auc', 'aupr', 'acc', 'f1']].std().round(4)
        print(f'\n{tissue} (n={len(tissue_rows)} folds):')
        for col in ['auc', 'aupr', 'acc', 'f1']:
            print(f'  {col.upper()}={means[col]:.4f} ± {stds[col]:.4f}')

# Save full table
df_all = pd.DataFrame(all_rows)
out_path = os.path.join(_RESULTS_DIR, '5fold_summary.csv')
df_all.to_csv(out_path, index=False)
print(f'\nFull table saved: {out_path}')

# Save mean±std summary
summary_rows = []
for tissue in TISSUES:
    sub = df_all[df_all['tissue'] == tissue]
    if len(sub) == 0:
        continue
    row = {'tissue': tissue, 'n_folds': len(sub)}
    for col in ['auc', 'aupr', 'acc', 'f1', 'precision', 'recall']:
        row[f'{col}_mean'] = round(sub[col].mean(), 4)
        row[f'{col}_std'] = round(sub[col].std(), 4)
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(_RESULTS_DIR, '5fold_summary_means.csv')
df_summary.to_csv(summary_path, index=False)
print(f'Summary saved: {summary_path}')
print(df_summary[['tissue', 'auc_mean', 'auc_std', 'aupr_mean', 'aupr_std', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std']].to_string(index=False))
