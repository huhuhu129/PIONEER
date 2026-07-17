import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, MLP

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GINNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINNetwork, self).__init__()
        self.conv1 = GINConv(MLP([input_dim, hidden_dim, hidden_dim]), train_eps=True)
        self.conv2 = GINConv(MLP([hidden_dim, hidden_dim, output_dim]), train_eps=True)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.gate = nn.Linear(2 * feature_dim, 1)

    def forward(self, rna_features, atac_features):
        q = self.query_proj(rna_features)
        k = self.key_proj(atac_features)
        v = self.value_proj(atac_features)
        scale = torch.sqrt(torch.tensor(rna_features.size(1), dtype=torch.float32, device=rna_features.device))
        attention = torch.matmul(q, k.transpose(1, 0)) / scale
        attention = F.softmax(attention, dim=1)
        attended_features = torch.matmul(attention, v)
        gate_input = torch.cat([rna_features, attended_features], dim=1)
        gate_value = torch.sigmoid(self.gate(gate_input))
        fused_features = gate_value * rna_features + (1 - gate_value) * attended_features
        return fused_features

class ConcatFusion(nn.Module):
    def __init__(self, feature_dim):
        super(ConcatFusion, self).__init__()
        self.proj = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, rna_features, atac_features):
        concat = torch.cat([rna_features, atac_features], dim=1)
        return self.proj(concat)

class GateFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GateFusion, self).__init__()
        self.gate_proj = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, rna_features, atac_features):
        gate_input = torch.cat([rna_features, atac_features], dim=1)
        gate_values = torch.sigmoid(self.gate_proj(gate_input))
        fused_features = gate_values * rna_features + (1.0 - gate_values) * atac_features
        return fused_features

class AvgFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AvgFusion, self).__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, rna_features, atac_features):
        avg = (rna_features + atac_features) / 2.0
        return self.proj(avg)

def nt_xent_loss(z1, z2, temperature=0.1):
    device = z1.device
    batch_size = z1.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    positives = similarity_matrix[labels, torch.arange(2 * batch_size, device=device)]
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels_ce = torch.zeros(2 * batch_size, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels_ce, reduction='mean')
    return loss

class GIN_DualNetwork_Contrastive(nn.Module):
    def __init__(self, rna_dim, atac_dim, embedding_dim, hidden_dim, fusion_dim, ablation_type='cross_attn',
                 temperature=0.1):
        super(GIN_DualNetwork_Contrastive, self).__init__()

        self.fusion_dim = fusion_dim
        self.ablation_type = ablation_type
        self.temperature = temperature
        self.use_contrastive = '_cl' in ablation_type
        self.base_type = ablation_type.replace('_cl', '')

        self.single_rna = (ablation_type == 'rna_only')
        self.single_atac = (ablation_type == 'atac_only')

        if not self.single_atac:
            self.rna_encoder = FeatureEncoder(rna_dim, embedding_dim)
            self.rna_gnn = GINNetwork(embedding_dim, hidden_dim, fusion_dim)
        if not self.single_rna:
            self.atac_encoder = FeatureEncoder(atac_dim, embedding_dim)
            self.atac_gnn = GINNetwork(embedding_dim, hidden_dim, fusion_dim)

        if not self.single_rna and not self.single_atac:
            if self.base_type == 'cross_attn':
                self.fusion = CrossAttentionFusion(fusion_dim)
            elif self.base_type == 'concat':
                self.fusion = ConcatFusion(fusion_dim)
            elif self.base_type == 'gate':
                self.fusion = GateFusion(fusion_dim)
            elif self.base_type == 'avg':
                self.fusion = AvgFusion(fusion_dim)
            else:
                self.fusion = CrossAttentionFusion(fusion_dim)

    def encode_single(self, subgraph_feats, edge_index, subgraph_nodes, encoder, gnn):
        device = next(self.parameters()).device
        encoded = encoder(subgraph_feats.to(device).float())
        gnn_output = gnn(encoded, edge_index.to(device))
        return gnn_output

    def forward(self, rna_edge_index, atac_edge_index, tissue_ppi, rna_features, atac_features,
                rna_genes, atac_genes, neg_samples=None, fusion_nodes_only=False, return_details=False):
        device = next(self.parameters()).device

        rna_subgraph_nodes = list(rna_genes) if isinstance(rna_genes, set) else rna_genes
        atac_subgraph_nodes = list(atac_genes) if isinstance(atac_genes, set) else atac_genes

        gene_to_rna_idx = {gene: idx for idx, gene in enumerate(rna_subgraph_nodes)}
        gene_to_atac_idx = {gene: idx for idx, gene in enumerate(atac_subgraph_nodes)}

        rna_subgraph_feats = torch.stack([
            rna_features[gene_to_rna_idx[node]] for node in rna_subgraph_nodes if node in gene_to_rna_idx
        ]) if gene_to_rna_idx else torch.empty(0, rna_features.size(1))

        atac_subgraph_feats = torch.stack([
            atac_features[gene_to_atac_idx[node]] for node in atac_subgraph_nodes if node in gene_to_atac_idx
        ]) if gene_to_atac_idx else torch.empty(0, atac_features.size(1))

        encoded_rna = None
        encoded_atac = None
        rna_gnn_output = None
        atac_gnn_output = None

        if not self.single_atac and rna_subgraph_feats.numel() > 0:
            encoded_rna = self.rna_encoder(rna_subgraph_feats.to(device).float())
            rna_gnn_output = self.rna_gnn(encoded_rna, rna_edge_index.to(device))

        if not self.single_rna and atac_subgraph_feats.numel() > 0:
            encoded_atac = self.atac_encoder(atac_subgraph_feats.to(device).float())
            atac_gnn_output = self.atac_gnn(encoded_atac, atac_edge_index.to(device))

        contrastive_loss = torch.tensor(0.0, device=device)
        fused_features = None

        common_nodes = []
        node_to_idx = {}

        if self.single_rna and rna_gnn_output is not None:
            fused_features = rna_gnn_output
            common_nodes = list(rna_subgraph_nodes)
            node_to_idx = {node: idx for idx, node in enumerate(common_nodes)}
            n_common = len(common_nodes)

        elif self.single_atac and atac_gnn_output is not None:
            fused_features = atac_gnn_output
            common_nodes = list(atac_subgraph_nodes)
            node_to_idx = {node: idx for idx, node in enumerate(common_nodes)}
            n_common = len(common_nodes)

        elif rna_gnn_output is not None and atac_gnn_output is not None:
            rna_set = set(rna_subgraph_nodes)
            atac_set = set(atac_subgraph_nodes)
            common_set = rna_set.intersection(atac_set)
            rna_only_set = rna_set - atac_set
            atac_only_set = atac_set - rna_set

            # Common nodes: cross-modal fusion
            common_nodes_sorted = sorted(common_set, key=lambda n: rna_subgraph_nodes.index(n))
            rna_common_idxs = [rna_subgraph_nodes.index(n) for n in common_nodes_sorted]
            atac_common_idxs = [atac_subgraph_nodes.index(n) for n in common_nodes_sorted]

            # RNA-only: keep GNN output directly
            rna_only_sorted = sorted(rna_only_set, key=lambda n: rna_subgraph_nodes.index(n))
            rna_only_idxs = [rna_subgraph_nodes.index(n) for n in rna_only_sorted]

            # ATAC-only: keep GNN output directly
            atac_only_sorted = sorted(atac_only_set, key=lambda n: atac_subgraph_nodes.index(n))
            atac_only_idxs = [atac_subgraph_nodes.index(n) for n in atac_only_sorted]

            # Build full feature_nodes list
            feature_nodes = common_nodes_sorted + rna_only_sorted + atac_only_sorted
            node_to_idx = {node: idx for idx, node in enumerate(feature_nodes)}
            n_common = len(common_nodes_sorted)

            fused_parts = []

            # 1) Fused common
            if common_nodes_sorted:
                rna_stack = rna_gnn_output[torch.tensor(rna_common_idxs, device=device)]
                atac_stack = atac_gnn_output[torch.tensor(atac_common_idxs, device=device)]
                fused_common = self.fusion(rna_stack, atac_stack)
                fused_parts.append(fused_common)
                if self.use_contrastive:
                    contrastive_loss = nt_xent_loss(rna_stack, atac_stack, self.temperature)

            # 2) RNA-only features
            if rna_only_sorted:
                rna_only_feats = rna_gnn_output[torch.tensor(rna_only_idxs, device=device)]
                fused_parts.append(rna_only_feats)

            # 3) ATAC-only features
            if atac_only_sorted:
                atac_only_feats = atac_gnn_output[torch.tensor(atac_only_idxs, device=device)]
                fused_parts.append(atac_only_feats)

            if fused_parts:
                fused_features = torch.cat(fused_parts, dim=0)
            else:
                fused_features = torch.empty(0, self.fusion_dim, device=device)

        if fused_features is None:
            fused_features = torch.empty(0, self.fusion_dim, device=device)

        if self.single_rna:
            feature_nodes = rna_subgraph_nodes
        elif self.single_atac:
            feature_nodes = atac_subgraph_nodes
        else:
            # feature_nodes already set above in fusion branch
            pass

        pos_edges = []
        for u, v in tissue_ppi.edges():
            if u in node_to_idx and v in node_to_idx:
                i = node_to_idx[u]
                j = node_to_idx[v]
                if i < j:
                    pos_edges.append((i, j))

        neg_edges = []
        if neg_samples:
            for u, v in neg_samples:
                if u in node_to_idx and v in node_to_idx:
                    i = node_to_idx[u]
                    j = node_to_idx[v]
                    if i < j:
                        neg_edges.append((i, j))

        # ── Optional: BCE loss only on fusion (intersection) nodes ──
        if fusion_nodes_only and n_common > 0:
            pos_edges = [(i, j) for i, j in pos_edges if i < n_common and j < n_common]
            neg_edges = [(i, j) for i, j in neg_edges if i < n_common and j < n_common]

        all_edges = pos_edges + neg_edges
        if all_edges and fused_features.size(0) > 0:
            src_indices = torch.tensor([e[0] for e in all_edges], device=device)
            tgt_indices = torch.tensor([e[1] for e in all_edges], device=device)
            src_feats = fused_features[src_indices]
            tgt_feats = fused_features[tgt_indices]
            predictions = F.cosine_similarity(src_feats, tgt_feats)
            n_pos = len(pos_edges)
            pos_predictions = predictions[:n_pos]
            neg_predictions = predictions[n_pos:]
        else:
            predictions = torch.empty(0, device=device)
            pos_predictions = predictions
            neg_predictions = predictions

        labels = []
        labels.extend([1.0] * len(pos_edges))
        labels.extend([0.0] * len(neg_edges))

        if labels:
            labels_tensor = torch.tensor(labels, device=device)
            predictions_prob = torch.sigmoid(predictions)
            graph_loss = F.binary_cross_entropy(predictions_prob, labels_tensor)
        else:
            graph_loss = torch.tensor(0.0, device=device)

        total_loss = graph_loss + contrastive_loss

        if return_details:
            details = {
                'encoded_rna': encoded_rna.cpu() if encoded_rna is not None else torch.empty(0),
                'encoded_atac': encoded_atac.cpu() if encoded_atac is not None else torch.empty(0),
                'rna_gnn_output': rna_gnn_output.cpu() if rna_gnn_output is not None else torch.empty(0),
                'atac_gnn_output': atac_gnn_output.cpu() if atac_gnn_output is not None else torch.empty(0),
                'fused_features': fused_features.cpu(),
                'common_nodes': feature_nodes,
                'node_to_idx': node_to_idx,
                'pos_edges': [(feature_nodes[i], feature_nodes[j]) for i, j in pos_edges],
                'pos_scores': [s.item() for s in pos_predictions],
                'neg_scores': [s.item() for s in neg_predictions],
                'rna_subgraph_nodes': rna_subgraph_nodes,
                'atac_subgraph_nodes': atac_subgraph_nodes,
                'graph_loss': graph_loss.item(),
                'contrastive_loss': contrastive_loss.item(),
            }
            return predictions, total_loss, len(pos_edges), len(neg_edges), details

        return predictions, total_loss, len(pos_edges), len(neg_edges)
