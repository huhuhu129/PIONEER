# -*- coding: utf-8 -*-
"""PIONEER universal training script — mouse, human, and PD.
Metrics: AUC, AUPR, ACC, F1.  Saves model, predictions, and PPI probabilities."""
import os, sys, json, pickle, random, time, argparse, numpy as np, torch, networkx as nx
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, f1_score, precision_score, recall_score)
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from model import GIN_DualNetwork_Contrastive


def compute_all_metrics(labels, scores):
    """Compute ACC, AUC, AUPR, F1, Precision, Recall from scores & binary labels."""
    # Threshold at 0.5 for binary decisions
    preds_bin = [1 if s >= 0.5 else 0 for s in scores]
    return {
        'auc': roc_auc_score(labels, scores) if len(set(labels)) >= 2 else 0.0,
        'aupr': average_precision_score(labels, scores) if len(set(labels)) >= 2 else 0.0,
        'acc': accuracy_score(labels, preds_bin),
        'f1': f1_score(labels, preds_bin, zero_division=0),
        'precision': precision_score(labels, preds_bin, zero_division=0),
        'recall': recall_score(labels, preds_bin, zero_division=0),
    }


def evaluate_balanced(model, re, ae, rf, af, rna_nodes, atac_nodes,
                       pos_edges, neg_list, device, max_pairs=None):
    """Evaluate with 1:1 positive/negative ratio (balanced sampling).
    If max_pairs is None, uses all positives and matches with equal negatives."""
    model.eval()
    with torch.no_grad():
        _, _, _, _, details = model(re, ae, nx.Graph(), rf, af,
                                     rna_nodes, atac_nodes, None, return_details=True)
        fused = details['fused_features']
        common = details['common_nodes']
        n2i = {n: i for i, n in enumerate(common)}

    # Filter valid positives
    valid_pos = [(u, v) for u, v in pos_edges if u in n2i and v in n2i]
    
    # Shuffle and cap
    random.shuffle(valid_pos)
    if max_pairs is not None:
        valid_pos = valid_pos[:max_pairs // 2]

    # Match equal number of negatives (from the larger neg pool)
    random.shuffle(neg_list)
    valid_neg = [(u, v) for u, v in neg_list if u in n2i and v in n2i]
    valid_neg = valid_neg[:len(valid_pos)]  # 1:1

    pairs_scores = []
    labels = []

    for u, v in valid_pos:
        s = float(torch.cosine_similarity(fused[n2i[u]:n2i[u]+1], fused[n2i[v]:n2i[v]+1]).item())
        pairs_scores.append({'protein_a': u, 'protein_b': v, 'score': s, 'label': 1})
        labels.append(1)

    for u, v in valid_neg:
        s = float(torch.cosine_similarity(fused[n2i[u]:n2i[u]+1], fused[n2i[v]:n2i[v]+1]).item())
        pairs_scores.append({'protein_a': u, 'protein_b': v, 'score': s, 'label': 0})
        labels.append(0)

    scores = [d['score'] for d in pairs_scores]
    metrics = compute_all_metrics(labels, scores)
    return metrics, pairs_scores, labels, scores


def parse_args():
    p = argparse.ArgumentParser(description='PIONEER Training')
    p.add_argument('--species', type=str, default='mouse', choices=['mouse', 'human', 'PD'])
    p.add_argument('--tissue', type=str, default='Thymus')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--fusion', type=str, default='gate')
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--fusion_dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val_every', type=int, default=5, help='Validate every N epochs')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ── Paths ──
    data_dir = os.path.join(_PROJ, 'data', args.species, args.tissue)
    fold_dir = os.path.join(data_dir, f'fold_{args.fold}')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(_PROJ, 'train_results',
                            f'{args.species}_{args.tissue}_fold{args.fold}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    print(f'Tissue: {args.tissue}  Fold: {args.fold}  Species: {args.species}')
    print(f'Device: {args.device}  Save: {save_dir}')

    # ── Load data ──
    gm = json.load(open(os.path.join(data_dir, 'gene_mapping.json')))
    gene_to_idx = gm['gene_to_idx']
    rna_ft = torch.load(os.path.join(data_dir, 'rna_features_tensor.pt'), weights_only=True)
    atac_ft = torch.load(os.path.join(data_dir, 'atac_features_tensor.pt'), weights_only=True)
    G_rna = pickle.load(open(os.path.join(data_dir, 'G_rna_subgraph.pickle'), 'rb'))
    G_atac = pickle.load(open(os.path.join(data_dir, 'G_atac_subgraph.pickle'), 'rb'))
    G_train = pickle.load(open(os.path.join(fold_dir, 'G_tissue_train.pickle'), 'rb'))
    G_val = pickle.load(open(os.path.join(fold_dir, 'G_tissue_val.pickle'), 'rb'))
    neg_samples = json.load(open(os.path.join(data_dir, 'neg_samples.json')))

    # ── Build edge indices & features ──
    rna_nodes = list(G_rna.nodes()); atac_nodes = list(G_atac.nodes())
    rn2i = {n: i for i, n in enumerate(rna_nodes)}
    an2i = {n: i for i, n in enumerate(atac_nodes)}

    re = [[rn2i[u], rn2i[v]] for u, v in G_rna.edges() if u in rn2i and v in rn2i]
    ae = [[an2i[u], an2i[v]] for u, v in G_atac.edges() if u in an2i and v in an2i]

    rf = torch.stack([rna_ft[gene_to_idx[n]] for n in rna_nodes if n in gene_to_idx]) \
        if rna_nodes else torch.zeros(0, rna_ft.shape[1])
    af = torch.stack([atac_ft[gene_to_idx[n]] for n in atac_nodes if n in gene_to_idx]) \
        if atac_nodes else torch.zeros(0, atac_ft.shape[1])

    device = args.device
    re = torch.tensor(re, dtype=torch.long).t().contiguous().to(device)
    ae = torch.tensor(ae, dtype=torch.long).t().contiguous().to(device)
    rf = rf.to(device); af = af.to(device)

    print(f'  RNA nodes: {rf.shape[0]:,}  ATAC nodes: {af.shape[0]:,}')
    print(f'  Train edges: {G_train.number_of_edges():,}  Val edges: {G_val.number_of_edges():,}')
    print(f'  Neg samples: {len(neg_samples):,}')

    # ── Model ──
    model = GIN_DualNetwork_Contrastive(
        rna_dim=rna_ft.shape[1], atac_dim=atac_ft.shape[1],
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        fusion_dim=args.fusion_dim, ablation_type=args.fusion,
        temperature=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}  Fusion: {args.fusion}')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_edges = list(G_train.edges())
    val_edges = list(G_val.edges())
    neg_list = [[str(a), str(b)] for a, b in neg_samples]

    # ── Training state ──
    best_auc = 0.0; best_aupr = 0.0; best_acc = 0.0; best_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model.train(); opt.zero_grad()
        n_sample = min(len(pos_edges), 40000)
        sp = random.sample(pos_edges, n_sample)
        sn = random.sample(neg_list, min(n_sample, len(neg_list)))
        Gb = nx.Graph(); Gb.add_nodes_from(G_train.nodes())
        for u, v in sp: Gb.add_edge(u, v)

        _, loss, _, _ = model(re, ae, Gb, rf, af, rna_nodes, atac_nodes, sn)
        if torch.isnan(loss) or loss.item() == 0:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ── Validation (1:1 balanced) ──
        if epoch % args.val_every == 0 and len(val_edges) > 0:
            metrics, _, _, _ = evaluate_balanced(
                model, re, ae, rf, af, rna_nodes, atac_nodes,
                val_edges, neg_list, device, max_pairs=2000)

            auc = metrics['auc']; aupr = metrics['aupr']
            acc = metrics['acc']; f1v = metrics['f1']

            improved = auc > best_auc
            if improved:
                best_auc = auc; best_aupr = aupr
                best_acc = acc; best_f1 = f1v
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                save_best = True
            else:
                patience_counter += 1
                save_best = False

            row = {'epoch': epoch, 'loss': round(loss.item(), 6),
                   'auc': round(auc, 6), 'aupr': round(aupr, 6),
                   'acc': round(acc, 6), 'f1': round(f1v, 6),
                   'best': improved}
            history.append(row)
            print(f'  Epoch {epoch:4d}: loss={loss.item():.4f}  '
                  f'AUC={auc:.4f}  AUPR={aupr:.4f}  ACC={acc:.4f}  F1={f1v:.4f}  '
                  f'best={best_auc:.4f}{" *" if improved else ""}')

            if patience_counter >= args.patience:
                print(f'  Early stopping at epoch {epoch}')
                break

    # ── Final evaluation (1:1 balanced, all positives + matched negatives) ──
    print('\n── Final evaluation (best model, 1:1 balanced) ──')
    best_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True, map_location=device))

    final_metrics, all_pairs, all_labels, all_scores = evaluate_balanced(
        model, re, ae, rf, af, rna_nodes, atac_nodes,
        val_edges, neg_list, device, max_pairs=None)

    print(f'  AUC={final_metrics["auc"]:.4f}  AUPR={final_metrics["aupr"]:.4f}  '
          f'ACC={final_metrics["acc"]:.4f}  F1={final_metrics["f1"]:.4f}')

    # ── Save everything ──
    # 1. metrics.json
    result = {
        'tissue': args.tissue, 'species': args.species, 'fold': args.fold,
        'fusion': args.fusion, 'params': n_params,
        'train_edges': G_train.number_of_edges(), 'val_edges': G_val.number_of_edges(),
        'rna_nodes': int(rf.shape[0]), 'atac_nodes': int(af.shape[0]),
        'auc': round(final_metrics['auc'], 6),
        'aupr': round(final_metrics['aupr'], 6),
        'acc': round(final_metrics['acc'], 6),
        'f1': round(final_metrics['f1'], 6),
        'precision': round(final_metrics['precision'], 6),
        'recall': round(final_metrics['recall'], 6),
        'eval_pos_count': sum(1 for d in all_pairs if d['label'] == 1),
        'eval_neg_count': sum(1 for d in all_pairs if d['label'] == 0),
        'best_epoch': history[-1]['epoch'] if history else 0,
        'total_epochs': len(history) * args.val_every if history else args.epochs,
    }
    json.dump(result, open(os.path.join(save_dir, 'metrics.json'), 'w'), indent=2)

    # 2. training_history.csv
    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    # 3. predictions.csv — all protein pairs with scores & labels
    pred_df = pd.DataFrame(all_pairs)
    pred_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

    # 4. ppi_probabilities.csv — positive-label edges (exclude self-loops)
    ppi_df = pred_df[(pred_df['label'] == 1) & (pred_df['protein_a'] != pred_df['protein_b'])][['protein_a', 'protein_b', 'score']].copy()
    ppi_df.columns = ['protein_a', 'protein_b', 'probability']
    ppi_df = ppi_df.sort_values('probability', ascending=False)
    ppi_df.to_csv(os.path.join(save_dir, 'ppi_probabilities.csv'), index=False)

    # 5. args.json — record training config
    cfg = {k: str(v) for k, v in vars(args).items()}
    json.dump(cfg, open(os.path.join(save_dir, 'args.json'), 'w'), indent=2)

    print(f'\nSaved to: {save_dir}')
    print(f'  Files: metrics.json, training_history.csv, predictions.csv, '
          f'ppi_probabilities.csv, best_model.pth, args.json')


if __name__ == '__main__':
    main()
