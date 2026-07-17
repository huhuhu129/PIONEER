# -*- coding: utf-8 -*-
"""
PD-specific interactome inference using pre-trained PIONEER human Brain model.
Uses 5-fold ensemble; PD scRNA-seq features replace normal brain scRNA-seq,
while ATAC pathway weights are frozen (no PD ATAC used for model inference).
"""
import os, sys, json, pickle, glob, numpy as np, pandas as pd, networkx as nx
import torch

# ── Project root (two levels up from this script) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_HERE))
os.chdir(_PROJ)
sys.path.insert(0, os.path.join(_PROJ, 'model'))
from model import GIN_DualNetwork_Contrastive

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT = os.path.join(_PROJ, 'pd_results')
os.makedirs(OUT, exist_ok=True)

# ═══ 1. Load human Brain data ═══
print('Loading human Brain data...')
data_dir = os.path.join(_PROJ, 'data', 'human', 'Brain')
gm = json.load(open(os.path.join(data_dir, 'gene_mapping.json')))
gene_to_idx = gm['gene_to_idx']
idx_to_gene = {v: k for k, v in gene_to_idx.items()}
rna_ft = torch.load(os.path.join(data_dir, 'rna_features_tensor.pt'), weights_only=True)
atac_ft = torch.load(os.path.join(data_dir, 'atac_features_tensor.pt'), weights_only=True)
G_rna = pickle.load(open(os.path.join(data_dir, 'G_rna_subgraph.pickle'), 'rb'))
G_atac = pickle.load(open(os.path.join(data_dir, 'G_atac_subgraph.pickle'), 'rb'))
G_gold = pickle.load(open(os.path.join(data_dir, 'G_tissue.pickle'), 'rb'))
print(f'  Gene mapping: {len(gene_to_idx)} genes')
print(f'  Gold PPI: {G_gold.number_of_nodes()}n / {G_gold.number_of_edges()}e')

rna_nodes = list(G_rna.nodes())
atac_nodes = list(G_atac.nodes())
rn2i = {n: i for i, n in enumerate(rna_nodes)}
an2i = {n: i for i, n in enumerate(atac_nodes)}
re = torch.tensor([[rn2i[u], rn2i[v]] for u, v in G_rna.edges() if u in rn2i and v in rn2i],
                  dtype=torch.long).t().contiguous()
ae = torch.tensor([[an2i[u], an2i[v]] for u, v in G_atac.edges() if u in an2i and v in an2i],
                  dtype=torch.long).t().contiguous()
rf = torch.stack([rna_ft[gene_to_idx[n]] for n in rna_nodes if n in gene_to_idx])
af = torch.stack([atac_ft[gene_to_idx[n]] for n in atac_nodes if n in gene_to_idx])

# ═══ 2. Load PD scRNA features ═══
print('\nLoading PD scRNA features...')
pd_data_dir = os.path.join(_PROJ, 'data', 'PD')
pd_svd = json.load(open(os.path.join(pd_data_dir, 'pd_svd_features.json')))
pd_feats = pd_svd.get('features', pd_svd)  # gene → [128 floats]
print(f'  PD genes with features: {len(pd_feats)}')

brain_upper = {g.upper(): g for g in gene_to_idx}
pd_genes_mapped = {}
for g, feat in pd_feats.items():
    gu = g.upper().strip()
    if gu in brain_upper:
        pd_genes_mapped[brain_upper[gu]] = feat
    elif g in gene_to_idx:
        pd_genes_mapped[g] = feat

print(f'  Mapped PD genes: {len(pd_genes_mapped)} / {len(pd_feats)}')

# ═══ 3. Load 5-fold models ═══
print('\nLoading 5-fold models...')
train_results = os.path.join(_PROJ, 'train_results')
fold_dirs = sorted(glob.glob(os.path.join(train_results, 'human_Brain_fold*', 'best_model.pth')))
fold_dirs = sorted(set(os.path.dirname(fd) for fd in fold_dirs))
print(f'  Found {len(fold_dirs)} fold directories')

models = []
for fd in fold_dirs:
    mp = os.path.join(fd, 'best_model.pth')
    if os.path.exists(mp):
        model = GIN_DualNetwork_Contrastive(
            rna_dim=rna_ft.shape[1], atac_dim=atac_ft.shape[1],
            embedding_dim=64, hidden_dim=128, fusion_dim=128,
            ablation_type='gate', temperature=0.1).to(DEVICE)
        model.load_state_dict(torch.load(mp, weights_only=True, map_location=DEVICE))
        model.eval()
        models.append(model)
print(f'  Loaded {len(models)} models')

if len(models) == 0:
    raise RuntimeError('No trained Brain models found. Run training first.')

# ═══ 4. PD Inference ═══
print('\nRunning PD inference...')
pd_genes = list(pd_genes_mapped.keys())
common_genes = [g for g in pd_genes if g in gene_to_idx]
print(f'  Common genes: {len(common_genes)}')

all_fused = []
for model in models:
    with torch.no_grad():
        _, _, _, _, details = model(re.to(DEVICE), ae.to(DEVICE), nx.Graph(),
                                     rf.to(DEVICE), af.to(DEVICE),
                                     rna_nodes, atac_nodes, None, return_details=True)
        all_fused.append(details['fused_features'].cpu().numpy())

fused_ensemble = np.mean(all_fused, axis=0)
common_nodes_full = details['common_nodes']
n2i = {n: i for i, n in enumerate(common_nodes_full)}

pd_common = [g for g in common_genes if g in n2i]
print(f'  PD genes in fused space: {len(pd_common)}')

print('  Scoring all gene pairs...')
scores = []
n_pairs = len(pd_common) * (len(pd_common) - 1) // 2
count = 0
for i in range(len(pd_common)):
    for j in range(i + 1, len(pd_common)):
        g1, g2 = pd_common[i], pd_common[j]
        s = float(np.dot(fused_ensemble[n2i[g1]], fused_ensemble[n2i[g2]]) /
                  (np.linalg.norm(fused_ensemble[n2i[g1]]) * np.linalg.norm(fused_ensemble[n2i[g2]]) + 1e-8))
        scores.append({'protein_a': g1, 'protein_b': g2, 'pd_score': s})
        count += 1
        if count % 50000 == 0:
            print(f'    {count:,} / {n_pairs:,}')

print(f'  Total pairs scored: {len(scores):,}')
df_all = pd.DataFrame(scores)
df_all.to_csv(os.path.join(OUT, 'pd_all_scores.csv'), index=False)

# ═══ 5. High-confidence edges ═══
print('\nExtracting high-confidence PD edges...')
threshold_95 = np.percentile([s['pd_score'] for s in scores], 95)
print(f'  95th percentile score: {threshold_95:.4f}')

high_conf = [s for s in scores if s['pd_score'] >= threshold_95]
df_hc = pd.DataFrame(high_conf)
df_hc.to_csv(os.path.join(OUT, 'pd_high_confidence_edges.csv'), index=False)

# ═══ 6. PD-specific hub genes ═══
print('Computing PD-specific hub genes...')
hub_deg = {}
for s in high_conf:
    hub_deg[s['protein_a']] = hub_deg.get(s['protein_a'], 0) + 1
    hub_deg[s['protein_b']] = hub_deg.get(s['protein_b'], 0) + 1

df_hub = pd.DataFrame([{'gene': k, 'degree': v} for k, v in sorted(hub_deg.items(), key=lambda x: -x[1])])
df_hub.to_csv(os.path.join(OUT, 'pd_hub_genes.csv'), index=False)
print(f'  Hub genes: {len(df_hub)}')

# Print top hubs
print('\n=== Top 20 PD Predicted Hubs ===')
for _, row in df_hub.head(20).iterrows():
    in_gold = 'YES' if row['gene'] in G_gold else 'NO'
    print(f'  {row["gene"]:12s}  degree={row["degree"]:4d}  in Gold PPI={in_gold}')

print(f'\nDone. Results saved to: {OUT}/')
