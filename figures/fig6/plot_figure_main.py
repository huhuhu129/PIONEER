# -*- coding: utf-8 -*-
"""
PD 主图 7 张独立子图（每个图单面板，带标题，统一浅色系配色）
输出目录：PIONEER/PD/figure_main/
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr, mannwhitneyu

PD = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(PD, 'prediction_full')
PROT = os.path.join(PD, 'proteomics')
OUT = os.path.join(PD, 'figure_main')
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# ── 统一浅色系配色（低饱和、高级感）──
C_PD    = '#6E9BC5'   # PD 蓝
C_CTRL  = '#B9C2CC'   # 对照灰
C_SIG   = '#DFA184'   # 珊瑚（显著/上调）
C_NEG   = '#96B2A0'   # 灰绿（下调）
C_TEAL  = '#8FB8A8'   # teal（GO）
C_PUR   = '#A99BC4'   # 紫灰（KEGG）
C_BLUE  = '#9DB4D0'   # 蓝灰（Reactome）
C_BOX   = '#E8E2D8'   # 米白（箱体）

def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)

# ══════════════════════════════════════════════════════════════
# a. Workflow 示意图
# ══════════════════════════════════════════════════════════════
def fig_a():
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis('off')
    boxes = [
        (0.2, 'Control\nscRNA + snATAC', C_CTRL),
        (2.2, 'PIONEER\ntraining\n(reference PPI)', C_BLUE),
        (4.4, 'PD\nscRNA + snATAC', C_PD),
        (6.4, 'PD-specific\nPPI network', C_SIG),
        (8.4, 'Network\ndifference', C_PUR),
    ]
    for x, txt, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.2), 1.7, 1.4,
                     boxstyle='round,pad=0.06,rounding_size=0.12',
                     linewidth=1.0, edgecolor='#7a7a7a', facecolor=c))
        ax.text(x + 0.85, 1.9, txt, ha='center', va='center', fontsize=8.5, color='#333333')
    for x in [1.9, 4.1, 6.1, 8.1]:
        ax.add_patch(FancyArrowPatch((x, 1.9), (x + 0.3, 1.9),
                     arrowstyle='-|>', mutation_scale=14, color='#555555', lw=1.2))
    ax.text(5.0, 0.35, 'PD 5013 nodes / 88324 edges    vs    control 3113 nodes / 88324 edges',
            ha='center', va='center', fontsize=8.5, color='#555555')
    ax.set_title('PIONEER workflow for PD-specific PPI prediction', fontsize=11, pad=8)
    save(fig, 'fig_a_workflow')

# ══════════════════════════════════════════════════════════════
# b. 5 折 CV（箱式图）
# ══════════════════════════════════════════════════════════════
def fig_b():
    CV = json.load(open(os.path.join(PD, 'metrics', 'cv_metrics.json')))
    auc = [m['best_auc'] for m in CV]; aupr = [m['best_aupr'] for m in CV]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    bp = ax.boxplot([auc, aupr], positions=[0, 1], widths=0.42, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(edgecolor='#7a7a7a', linewidth=1.0),
                    medianprops=dict(color='#444444', linewidth=1.5),
                    whiskerprops=dict(color='#7a7a7a', linewidth=1.0),
                    capprops=dict(color='#7a7a7a', linewidth=1.0))
    bp['boxes'][0].set_facecolor(C_PD); bp['boxes'][0].set_alpha(0.75)
    bp['boxes'][1].set_facecolor(C_SIG); bp['boxes'][1].set_alpha(0.75)
    rng = np.random.default_rng(0)
    for i, v in enumerate([auc, aupr]):
        jx = np.full(len(v), i) + rng.uniform(-0.05, 0.05, len(v))
        ax.scatter(jx, v, color='#444444', s=22, zorder=3, edgecolor='white', linewidth=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['AUC', 'AUPR'])
    ax.set_ylim(min(auc) - 0.004, max(aupr) + 0.004)
    ax.set_ylabel('Score')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('5-fold cross-validation performance', fontsize=11, pad=8)
    save(fig, 'fig_b_cv_performance')

# ══════════════════════════════════════════════════════════════
# c. 网络重连（PD vs 对照 度分布）
# ══════════════════════════════════════════════════════════════
def fig_c():
    from scipy.stats import gaussian_kde
    G_pd = pickle.load(open(os.path.join(PF, 'pd_predicted_network_full.pickle'), 'rb'))
    G_ctrl = pickle.load(open(os.path.join(PF, 'control', 'control_predicted_network_full.pickle'), 'rb'))
    dpd = np.log10(np.array([d for _, d in G_pd.degree()]) + 1)
    dct = np.log10(np.array([d for _, d in G_ctrl.degree()]) + 1)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs = np.linspace(0.2, max(dpd.max(), dct.max()) + 0.15, 300)
    for vals, c, lab in [(dpd, C_PD, 'PD'), (dct, C_CTRL, 'Control')]:
        kde = gaussian_kde(vals, bw_method=0.25)
        y = kde(xs)
        ax.plot(xs, y, color=c, lw=2.4, label=lab)
        ax.fill_between(xs, y, color=c, alpha=0.20)
    ax.set_xlabel('log10(degree + 1)')
    ax.set_ylabel('Density')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('PD vs control network rewiring', fontsize=11, pad=8)
    save(fig, 'fig_c_network_rewiring')

# ══════════════════════════════════════════════════════════════
# d. 蛋白共丰度验证（预测边@PD vs 随机）
# ══════════════════════════════════════════════════════════════
def fig_d():
    mat = pd.read_csv(os.path.join(PROT, 'pxd024748_protein_matrix.csv'), index_col=0)
    mat.index = [str(g).upper() for g in mat.index]
    PD_COLS = ['N07', 'N08', 'N10']
    G_pred = pickle.load(open(os.path.join(PF, 'pd_predicted_network_full.pickle'), 'rb'))
    prot_genes = set(mat.index)

    def coabund(a, b, cols, min_valid=3):
        a, b = str(a).upper(), str(b).upper()
        if a not in mat.index or b not in mat.index or a == b:
            return None
        va = mat.loc[a, cols].values.astype(float); vb = mat.loc[b, cols].values.astype(float)
        m = ~(np.isnan(va) | np.isnan(vb))
        if m.sum() < min_valid:
            return None
        r, _ = spearmanr(va[m], vb[m])
        return r

    pred_edges = [(u, v) for u, v in G_pred.edges()
                  if str(u).upper() in prot_genes and str(v).upper() in prot_genes]
    pred_r = [r for r in (coabund(u, v, PD_COLS) for u, v in pred_edges) if r is not None]
    rng = np.random.default_rng(42)
    genes = list(mat.index)
    rand_r = []
    while len(rand_r) < len(pred_r):
        a, b = rng.choice(genes, 2, replace=False)
        r = coabund(a, b, PD_COLS)
        if r is not None:
            rand_r.append(r)
    u, p = mannwhitneyu(pred_r, rand_r, alternative='greater')

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    bp = ax.boxplot([pred_r, rand_r], positions=[0, 1], widths=0.5, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(edgecolor='#7a7a7a', linewidth=1.0),
                    medianprops=dict(color='#444444', linewidth=1.5),
                    whiskerprops=dict(color='#7a7a7a', linewidth=1.0),
                    capprops=dict(color='#7a7a7a', linewidth=1.0))
    bp['boxes'][0].set_facecolor(C_SIG); bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor(C_CTRL); bp['boxes'][1].set_alpha(0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Predicted\n@PD', 'Random\n@PD'])
    ax.set_ylabel('Co-abundance (Spearman r)')
    ax.axhline(0, color='#999999', ls='--', lw=0.8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(0.5, 0.95, f'p = {p:.1e}', transform=ax.transAxes, ha='center', va='top',
            fontsize=10, color='#333333')
    ax.set_title('Proteomic co-abundance validation of predicted PD edges', fontsize=11, pad=8)
    save(fig, 'fig_d_coabundance')

# ══════════════════════════════════════════════════════════════
# e. 已知基因重连（Δdegree 条形图）
# ══════════════════════════════════════════════════════════════
def fig_e():
    core = pd.read_csv(os.path.join(PF, 'core7_degree_compare.csv'))
    zac = pd.read_csv(os.path.join(PF, 'zaccaria33_degree_compare.csv'))
    core['src'] = 'core'; zac['src'] = 'zaccaria'
    d = pd.concat([core, zac], ignore_index=True)
    d = d[d['delta'].abs() >= 2].copy()   # 过滤无变化的
    d = d.sort_values('delta')
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    colors = [C_SIG if v > 0 else C_NEG for v in d['delta']]
    ax.barh(d['gene'], d['delta'], color=colors, edgecolor='#7a7a7a', linewidth=0.5)
    ax.axvline(0, color='#888888', lw=0.8)
    ax.set_xlabel('Δ degree (PD − control)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('Network rewiring of known PD-associated genes', fontsize=11, pad=8)
    save(fig, 'fig_e_known_genes')

# ══════════════════════════════════════════════════════════════
# f. 通路富集 dot plot
# ══════════════════════════════════════════════════════════════
def fig_f():
    import re
    libs = {
        'GO': (os.path.join(PF, 'enrichment_GO_Biological_Process_2021.csv'), C_TEAL, 'GO biological process'),
        'KEGG': (os.path.join(PF, 'enrichment_KEGG_2021_Human.csv'), C_PUR, 'KEGG pathway'),
        'Reactome': (os.path.join(PF, 'enrichment_Reactome_2022.csv'), C_BLUE, 'Reactome pathway'),
    }
    def clean_term(t):
        t = re.sub(r'\s*\(GO:\d+\)', '', t)
        t = re.sub(r'\s*R-HSA-\d+', '', t)
        t = re.sub(r'\s*\(hsa\d+\)', '', t)
        return t[:44]
    for lib, (path, c, title) in libs.items():
        df = pd.read_csv(path).head(10)
        terms = [clean_term(t) for t in df['Term']]
        neglogp = [-np.log10(p) for p in df['P-value']]
        y = np.arange(len(terms))[::-1]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.barh(y, neglogp, color=c, edgecolor='white', linewidth=0.5, alpha=0.92)
        ax.set_yticks(y)
        ax.set_yticklabels(terms, fontsize=10)
        ax.set_xlabel('-log10(p-value)', fontsize=11)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_title(f'{title} enrichment', fontsize=12, pad=8)
        save(fig, f'fig_f_{lib.lower()}_enrichment')

# ══════════════════════════════════════════════════════════════
# g. 机制图：蛋白稳态崩溃
# ══════════════════════════════════════════════════════════════
def fig_g():
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis('off')
    steps = [
        (0.1, 'α-synuclein\naggregation', C_PD),
        (2.0, 'ER stress', C_BLUE),
        (3.9, 'UPR activation\n(HSPA1A, HSPB1, HSPD1)', C_TEAL),
        (6.1, 'Chaperone overload\n(HSP90, DNAJ)', C_PUR),
        (8.3, 'CMA failure\n(GFAP, HSPA8)', C_SIG),
    ]
    for x, txt, c in steps:
        ax.add_patch(FancyBboxPatch((x, 1.1), 1.55, 1.3,
                     boxstyle='round,pad=0.05,rounding_size=0.1',
                     linewidth=1.0, edgecolor='#7a7a7a', facecolor=c))
        ax.text(x + 0.775, 1.75, txt, ha='center', va='center', fontsize=7.8, color='#333333')
    for x in [1.65, 3.55, 5.45, 7.55]:
        ax.add_patch(FancyArrowPatch((x, 1.75), (x + 0.4, 1.75),
                     arrowstyle='-|>', mutation_scale=13, color='#555555', lw=1.2))
    ax.set_title('Proteostasis collapse: a core PD pathogenic axis', fontsize=11, pad=8)
    save(fig, 'fig_g_mechanism')

if __name__ == '__main__':
    fig_a(); fig_b(); fig_c(); fig_d(); fig_e(); fig_f(); fig_g()
    print('全部完成 ->', OUT)
