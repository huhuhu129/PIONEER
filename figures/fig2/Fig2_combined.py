# -*- coding: utf-8 -*-
"""
Fig2 主图拼接（面板 A 完全照搬 figure2d_only.py 的 Fig2a 原始代码）
================================================================================
布局：
  A（基准点阵图，独占一行，全宽）  —— 沿用 figure2d_only.py 原始参数（4 方法、y_offset 下移、14.8 英寸宽）
  B（消融实验，左下）  C（5-fold CV vs LOO，右下）
每个面板：左上角 A/B/C 标签 + 顶部居中标题。

输出：Fig2_main.pdf / Fig2_main.svg / Fig2_main.png
"""

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns
from scipy.stats import ttest_ind

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- 面板标签 / 标题字号 ----------------
LABEL_FS = 18.0   # 面板标签 A/B/C/D/E（加粗）
TITLE_FS = 16.0   # 面板标题（加粗）

TITLE_A = 'Benchmark performance across tissues'
TITLE_B = 'Ablation study'
TITLE_C = '5-fold CV vs leave-one-tissue-out'
TITLE_D = 'Top-k high-confidence edge retrieval'
TITLE_E = 'Relative AUROC gain'

MAIN_TITLE = 'Fig2. PIONEER model robustness and stability'
MAIN_FS = 18.0   # 主标题字号（大于子图标题 16pt）

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def panel_label(ax, s):
    """左上角面板标签，与图标题垂直对齐（基线同高）。"""
    ax.annotate(s, xy=(0.0, 1.0), xycoords='axes fraction', xytext=(0, 10),
                textcoords='offset points', va='baseline', ha='left',
                fontsize=LABEL_FS, fontweight='bold', zorder=100)


# ============================================================
# 面板 A：多方法多组织基准点阵图（照搬 figure2d_only.py 原始代码）
# ============================================================
def draw_panel_a(ax):
    with open(os.path.join(SCRIPT_DIR, 'Fig2a_Radar_Grid.json'), encoding='utf-8') as f:
        data = json.load(f)

    metric_names = ['AUROC', 'AUPRC', 'Accuracy', 'F1']
    tissue_order = ['Brain', 'Heart', 'Kidney', 'Lung', 'Thymus']
    method_order = ['PIONEER', 'GraphSAGE', 'GAT', 'GCN']
    existing_methods = sorted({m for t in data for m in data[t]})
    method_order = [m for m in method_order if m in existing_methods]

    # long dataframe
    rows = []
    for tissue, method_dict in data.items():
        for method, values in method_dict.items():
            for metric, value in zip(metric_names, values):
                rows.append({'tissue': tissue, 'method': method, 'metric': metric, 'score': value})
    df_long = pd.DataFrame(rows)
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')

    # overall score
    overall = df_long.groupby('method', observed=False)['score'].mean().reindex(method_order)
    overall_min = np.nanmin(overall.values)
    overall_max = np.nanmax(overall.values)
    overall_norm = (overall - overall_min) / (overall_max - overall_min) if overall_max > overall_min else overall * 0 + 1

    # colors（注意：第一列标签是 "all"）
    group_colors = {'all': '#7B4FA1', 'AUROC': '#2F79B9', 'AUPRC': '#2E9A77',
                    'Accuracy': '#F07A24', 'F1': '#CC3C7A'}
    metric_colors = {k: v for k, v in group_colors.items() if k != 'all'}
    overall_cmap = LinearSegmentedColormap.from_list('overall_purple', ['#EFE2F5', '#7B4FA1'])

    def blend_with_white(hex_color, score):
        if pd.isna(score):
            return '#FFFFFF'
        rgb = np.array(mcolors.to_rgb(hex_color))
        white = np.array([1.0, 1.0, 1.0])
        score = np.clip(score, 0, 1)
        score_adj = score ** 1.8
        alpha = 0.10 + 0.90 * score_adj
        return tuple(white * (1 - alpha) + rgb * alpha)

    def score_to_size(score, min_size=15, max_size=320):
        if pd.isna(score):
            return 0
        score = np.clip(score, 0, 1)
        lo, hi = 0.5, 1.0
        score_norm = np.clip((score - lo) / (hi - lo), 0, 1)
        return min_size + (max_size - min_size) * score_norm

    def draw_round_header(x0, x1, y, h, color, text, fontsize=15.8):
        rect = patches.FancyBboxPatch(
            (x0, y), x1 - x0, h,
            boxstyle='round,pad=0.025,rounding_size=0.14',
            linewidth=0, facecolor=color, alpha=0.96, clip_on=False, zorder=5)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, y + h / 2, text, ha='center', va='center',
                color='white', fontsize=fontsize, zorder=6)

    n_methods = len(method_order)
    n_tissues = len(tissue_order)

    # 布局坐标（照搬原始）
    x_method_label = -1
    x_overall = 0.70
    overall_half_width = 0.90
    tissue_step = 1.22
    group_gap = 1.28

    metric_x_start = {}
    current_x = 2.55
    for metric in metric_names:
        metric_x_start[metric] = current_x
        current_x += (n_tissues - 1) * tissue_step + 1.0 + group_gap
    main_right_edge = current_x - group_gap + 0.30

    # Y 偏移：数据行 & 组织标签整体下移
    y_offset = 1.4
    y_offset_tissue = 1.2

    y_header = -1.7
    header_h = 0.58
    y_subheader_base = -1.25
    y_tissue_label = y_subheader_base + y_offset_tissue
    y_overall_score = y_header + header_h / 2

    y_row_min = y_offset - 0.5
    y_row_max = n_methods - 1 + y_offset + 0.5
    y_legend = n_methods + y_offset + 0.95

    x_min = -2.4
    x_max = main_right_edge + 0.45
    y_min = -1.8
    y_max = y_row_max + 0.4

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.axis('off')

    # 交替行背景（下移）
    for i, method in enumerate(method_order):
        row_color = '#F7F7F7' if i % 2 == 0 else '#FFFFFF'
        yc = i + y_offset
        ax.add_patch(patches.Rectangle((x_min, yc - 0.5), x_max - x_min, 1.0,
                                       facecolor=row_color, edgecolor='none', zorder=0))

    # 分组分隔线（下移）
    ax.plot([x_overall - overall_half_width, x_overall - overall_half_width],
            [y_row_min, y_row_max], color='#D0D0D0', lw=0.9)
    ax.plot([x_overall + overall_half_width, x_overall + overall_half_width],
            [y_row_min, y_row_max], color='#D0D0D0', lw=0.9)
    for metric in metric_names:
        xs = metric_x_start[metric]
        ax.plot([xs - 0.65, xs - 0.65], [y_row_min, y_row_max],
                color='#D9D9D9', lw=0.8, linestyle='--')

    # 分组头（不下移）
    draw_round_header(x_overall - overall_half_width + 0.06, x_overall + overall_half_width - 0.06,
                      y_header, header_h, group_colors['all'], 'all', fontsize=15)
    for metric in metric_names:
        xs = metric_x_start[metric]
        xe = xs + (n_tissues - 1) * tissue_step
        draw_round_header(xs - 0.58, xe + 0.58, y_header, header_h, group_colors[metric], metric)

    ax.text(x_overall, y_overall_score, 'all', ha='center', va='center', fontsize=14.2)

    # 组织标签（下移，幅度小于数据行）
    for metric in metric_names:
        xs = metric_x_start[metric]
        for j, tissue in enumerate(tissue_order):
            ax.text(xs + j * tissue_step, y_tissue_label, tissue, ha='center', va='center',
                    rotation=90, fontsize=15)

    # 方法行标签（下移）
    for i, method in enumerate(method_order):
        ax.text(x_method_label, i + y_offset, method, ha='right', va='center',
                fontsize=15, fontweight='bold')

    # Overall 方块（下移）
    for i, method in enumerate(method_order):
        val = overall_norm.loc[method]
        color = overall_cmap(val)
        size = 0.18 + 0.36 * (val ** 1.8)
        yc = i + y_offset
        ax.add_patch(patches.Rectangle((x_overall - size / 2, yc - size / 2), size, size,
                                       facecolor=color, edgecolor='#6B3D8B', linewidth=0.8, zorder=3))

    # 指标气泡（下移）
    for metric in metric_names:
        xs = metric_x_start[metric]
        base_color = metric_colors[metric]
        for i, method in enumerate(method_order):
            for j, tissue in enumerate(tissue_order):
                val_series = df_long.loc[(df_long['method'] == method) & (df_long['metric'] == metric) & (df_long['tissue'] == tissue), 'score']
                val = val_series.iloc[0] if len(val_series) else np.nan
                x = xs + j * tissue_step
                y = i + y_offset
                if pd.isna(val):
                    ax.text(x, y, '–', ha='center', va='center', fontsize=16.5,
                            color='#B8B8B8', zorder=3)
                    continue
                ax.scatter(x, y, s=score_to_size(val), c=[blend_with_white(base_color, val)],
                           edgecolors=base_color, linewidths=0.70, alpha=0.96, zorder=4)

    # 底部 Score 图例
    score_x = main_right_edge + 0.1
    score_y_top = y_offset + 0.8
    score_step_y = 0.72
    ax.text(score_x, score_y_top - 1.05, 'Score', ha='center', va='center',
            fontsize=15.8, rotation=90)
    for k, v in enumerate([1.00, 0.75, 0.50]):
        y = score_y_top + k * score_step_y
        ax.scatter(score_x, y, s=score_to_size(v), facecolors='white',
                   edgecolors='#666666', linewidths=0.8, zorder=5)
        ax.text(score_x + 0.45, y, f'{v:.2f}', ha='left', va='center', fontsize=13.2)

    ax.set_title(TITLE_A, fontsize=TITLE_FS, fontweight='bold', pad=10)
    panel_label(ax, 'A')


# ============================================================
# 面板 B：消融实验（照搬原始代码）
# ============================================================
def draw_panel_b(ax):
    with open(os.path.join(SCRIPT_DIR, 'Fig2b_ablation_data.json'), encoding='utf-8') as f:
        data = json.load(f)

    df_wide = pd.DataFrame.from_dict(data, orient='index')
    df_wide.columns = [f'fold_{i + 1}' for i in range(df_wide.shape[1])]
    df_wide = df_wide.reset_index().rename(columns={'index': 'method'})
    df_long = df_wide.melt(id_vars='method', var_name='fold', value_name='score')
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')
    df_long = df_long.dropna(subset=['score'])

    name_map = {'GIN+Gate (PIONEER)': 'PIONEER', 'GIN+Avg': 'Average fusion',
                'RNA-Only': 'w/o ATAC', 'ATAC-Only': 'w/o RNA',
                'GIN+Concat': 'Concat fusion', 'GIN+Gate+CL': 'w/ CL'}
    df_long['method_plot'] = df_long['method'].astype(str).map(name_map)

    method_order = ['w/o RNA', 'w/o ATAC', 'Average fusion', 'Concat fusion', 'w/ CL', 'PIONEER']
    palette = {'w/o RNA': '#AFC6E9', 'w/o ATAC': '#B9D7EA', 'Average fusion': '#C8DCC8',
               'Concat fusion': '#D6E8D6', 'w/ CL': '#E6E6E6', 'PIONEER': '#F2A7A7'}

    sns.violinplot(data=df_long, x='method_plot', y='score', order=method_order,
                   hue='method_plot', palette=palette, legend=False,
                   inner=None, cut=0, linewidth=1.2, saturation=1, ax=ax)
    sns.boxplot(data=df_long, x='method_plot', y='score', order=method_order, width=0.22,
                showcaps=True, showfliers=False,
                boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.75),
                whiskerprops=dict(color='black', linewidth=1.0),
                capprops=dict(color='black', linewidth=1.0),
                medianprops=dict(color='black', linewidth=1.2), ax=ax)

    rng = np.random.default_rng(123)
    for i, method in enumerate(method_order):
        vals = df_long.loc[df_long['method_plot'] == method, 'score'].values
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=20,
                   facecolors='none', edgecolors=palette[method], linewidths=1.4, zorder=4)
        ax.plot([i - 0.18, i + 0.18], [vals.mean(), vals.mean()], color='black', linewidth=1.6, zorder=5)

    # 显著性括号（PIONEER vs Average fusion）
    def add_sig_bracket(x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c='black')
        ax.text((x1 + x2) / 2, y + h + 0.003, text, ha='center', va='bottom', fontsize=16.5)

    g1 = df_long.loc[df_long['method_plot'] == 'Average fusion', 'score'].values
    g2 = df_long.loc[df_long['method_plot'] == 'PIONEER', 'score'].values
    _, pval = ttest_ind(g1, g2, equal_var=False)
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'NS'))
    y_max = df_long['score'].max()
    add_sig_bracket(2, 5, y_max + 0.01, 0.01, sig)

    ax.set_xlabel('')
    ax.set_ylabel('AUROC', fontsize=18)
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, rotation=30, ha='right', rotation_mode='anchor', fontsize=15)
    ax.set_ylim(0.60, 1.03)
    ax.tick_params(axis='y', labelsize=15)
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
        ax.spines[s].set_color('black')
    ax.tick_params(axis='both', width=1.0, length=4, direction='out')

    ax.set_title(TITLE_B, fontsize=TITLE_FS, fontweight='bold', pad=10)
    panel_label(ax, 'B')


# ============================================================
# 面板 C：5-fold CV vs Leave-One-Tissue-Out（照搬原始代码）
# ============================================================
def draw_panel_c(ax):
    with open(os.path.join(SCRIPT_DIR, 'Fig2c_violin_data.json'), encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for cv_type, tissue_dict in data.items():
        for tissue, values in tissue_dict.items():
            for value in values:
                rows.append({'cv_type': cv_type, 'tissue': tissue, 'score': value})
    df_long = pd.DataFrame(rows)
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')
    df_long['cv_plot'] = df_long['cv_type'].map({'5fold_CV': '5-fold CV', 'LOO': 'Leave-one-tissue-out'})
    df_long = df_long.dropna(subset=['score', 'cv_plot'])

    tissue_order = ['Brain', 'Heart', 'Kidney', 'Lung', 'Thymus']
    cv_order = ['5-fold CV', 'Leave-one-tissue-out']

    color_map = {'5-fold CV': '#3F6FA8', 'Leave-one-tissue-out': '#C95F58'}
    marker_map = {'5-fold CV': 'o', 'Leave-one-tissue-out': 's'}
    offset = {'5-fold CV': -0.15, 'Leave-one-tissue-out': 0.15}

    y = np.arange(len(tissue_order))
    for yi in y:
        ax.axhline(yi, color='#EAEAEA', lw=0.8, zorder=0)

    for cv in cv_order:
        means, stds = [], []
        for t in tissue_order:
            vals = df_long[(df_long['tissue'] == t) & (df_long['cv_plot'] == cv)]['score'].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        y_pos = y + offset[cv]
        ax.errorbar(means, y_pos, xerr=stds, fmt=marker_map[cv], markersize=7.2,
                    markerfacecolor=color_map[cv], markeredgecolor='white', markeredgewidth=0.9,
                    ecolor=color_map[cv], elinewidth=1.2, capsize=3.2, capthick=1.1,
                    linestyle='none', label=cv, zorder=5)

    rng = np.random.default_rng(123)
    for i, tissue in enumerate(tissue_order):
        for cv in cv_order:
            vals = df_long[(df_long['tissue'] == tissue) & (df_long['cv_plot'] == cv)]['score'].values
            if len(vals) == 0:
                continue
            y_jitter = rng.normal(0, 0.018, size=len(vals))
            x_jitter = rng.normal(0, 0.00045, size=len(vals))
            ax.scatter(vals + x_jitter, np.full(len(vals), i + offset[cv]) + y_jitter, s=14,
                       facecolors='white', edgecolors=color_map[cv], linewidths=0.6, alpha=0.42, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(tissue_order, fontsize=18)
    ax.set_xlabel('AUROC', fontsize=18)
    ax.invert_yaxis()

    xmin = float(df_long['score'].min())
    xmax = float(df_long['score'].max())
    pad = 0.012
    ax.set_xlim(max(0, xmin - pad), min(1.0, xmax + pad))
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelsize=15)

    ax.grid(axis='x', linestyle='-', linewidth=0.45, alpha=0.20, zorder=0)
    ax.grid(axis='y', visible=False)
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
        ax.spines[s].set_color('#4D4D4D')
    ax.tick_params(axis='both', width=1.0, length=4, direction='out')

    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=2, fontsize=15.8, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.0)

    ax.set_title(TITLE_C, fontsize=TITLE_FS, fontweight='bold', pad=10)
    panel_label(ax, 'C')


# ============================================================
# 面板 D：top-k 高置信边检索（precision@k + recall@100）
# ============================================================
def draw_panel_d(ax):
    from matplotlib.lines import Line2D

    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'Fig2e_metrics_summary_mouse.csv'))
    tissues = ['Brain', 'Heart', 'Kidney', 'Lung', 'Thymus']
    TISSUE_COLOR = {'Brain': '#0F4D92', 'Heart': '#C44E52', 'Kidney': '#2E9E44',
                    'Lung': '#E28E2C', 'Thymus': '#7C6CCF'}
    k_vals = [100, 500, 1000]

    for t in tissues:
        sub = df[(df['type'] == 'GIN+Gate') & (df['tissue'] == t)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        prec = [r['precision@100_mean'], r['precision@500_mean'], r['precision@1000_mean']]
        ax.plot(k_vals, prec, color=TISSUE_COLOR[t], lw=1.6, marker='o', markersize=4, label=t)
        ax.scatter([100], [r['recall@100_mean']], color=TISSUE_COLOR[t], marker='^', s=36,
                   edgecolor='white', linewidths=0.6, zorder=5)

    ax.set_xscale('log')
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals], fontsize=15)
    ax.set_xlabel('Top-k predicted edges (k)', fontsize=18)
    ax.set_ylabel('Precision / Recall @k', fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.tick_params(width=0.9, length=4)

    tissue_handles = [Line2D([0], [0], color=TISSUE_COLOR[t], lw=1.6, label=t) for t in tissues]
    metric_handles = [
        Line2D([0], [0], color='#333333', lw=1.6, marker='o', markersize=4, label='Precision@k'),
        Line2D([0], [0], color='#333333', linestyle='none', marker='^', markersize=5, label='Recall@100')]
    ax.legend(handles=tissue_handles + metric_handles, loc='lower left', fontsize=13,
              ncol=2, frameon=False, columnspacing=1.0)

    ax.set_title(TITLE_D, fontsize=TITLE_FS, fontweight='bold', pad=10)
    panel_label(ax, 'D')


# ============================================================
# 面板 E：delta-AUROC 相对增益热图
# ============================================================
def draw_panel_e(ax):
    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'Fig2e_metrics_summary_mouse.csv'))
    methods = ['GIN+Gate', 'GAT-NoRecon', 'SAGE-NoRecon', 'GCN-NoRecon']
    labels = ['PIONEER', 'GAT', 'GraphSAGE', 'GCN']
    tissues = ['Brain', 'Heart', 'Kidney', 'Lung', 'Thymus']

    auc = {}
    for _, r in df.iterrows():
        if r['type'] in methods and r['tissue'] in tissues:
            auc[(r['type'], r['tissue'])] = r['auroc_mean']

    mat = np.zeros((len(methods), len(tissues)))
    for j, t in enumerate(tissues):
        col = [auc[(m, t)] for m in methods if (m, t) in auc]
        if not col:
            continue
        worst = min(col)
        for i, m in enumerate(methods):
            if (m, t) in auc:
                mat[i, j] = auc[(m, t)] - worst

    cmap = plt.get_cmap('GnBu')
    im = ax.imshow(mat, cmap='GnBu', aspect='auto', vmin=0, vmax=0.26)
    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(tissues, rotation=45, ha='right', fontsize=15)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels, fontsize=15)

    for i in range(len(methods)):
        for j in range(len(tissues)):
            rgba = cmap(mat[i, j] / 0.26)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center', fontsize=13,
                    color='white' if lum < 0.5 else 'black')

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('delta-AUROC vs worst baseline', fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    ax.tick_params(width=0.9, length=4)
    ax.set_title(TITLE_E, fontsize=TITLE_FS, fontweight='bold', pad=10)
    panel_label(ax, 'E')


# ============================================================
# 主拼装
# ============================================================
def main():
    H = 16.2
    fig = plt.figure(figsize=(14.8, H))

    left, right, top, bottom = 0.05, 0.97, 0.928, 0.04
    content_w = right - left

    # 面板高度与纵向间距（英寸）
    hA_in, hBC_in, hDE_in = 5.0, 4.4, 3.0
    gap_ab_in = 0.57   # A 与 BC 轴间距（内容 GCN->标题 ≈ 0.54 英寸）
    gap_bc_in = 1.42   # BC 与 DE 轴间距（内容 AUROC->标题 ≈ 0.54 英寸）

    hA = hA_in / H
    hBC = hBC_in / H
    hDE = hDE_in / H
    gap_ab = gap_ab_in / H
    gap_bc = gap_bc_in / H

    # 横向间距（沿用 wspace=0.35 的效果）
    wspace = 0.35
    col_w_total = content_w / (1 + wspace / 2)
    wgap = wspace * col_w_total / 2
    col_w = col_w_total / 2

    axA = fig.add_axes([left, top - hA, content_w, hA])
    row1_top = top - hA - gap_ab
    axB = fig.add_axes([left, row1_top - hBC, col_w, hBC])
    axC = fig.add_axes([left + col_w + wgap, row1_top - hBC, col_w, hBC])
    row2_top = row1_top - hBC - gap_bc
    axD = fig.add_axes([left, row2_top - hDE, col_w, hDE])
    axE = fig.add_axes([left + col_w + wgap, row2_top - hDE, col_w, hDE])

    draw_panel_a(axA)
    draw_panel_b(axB)
    draw_panel_c(axC)
    draw_panel_d(axD)
    draw_panel_e(axE)

    # 最上方主标题（左侧与算法名称列对齐）
    main_title = fig.text(0.0, 0.995, MAIN_TITLE, ha='left', va='top',
                          fontsize=MAIN_FS, fontweight='bold')

    # 测量两处内容间距 + 主标题间距
    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    mt = main_title.get_window_extent(renderer=fig.canvas.get_renderer())
    mt_bottom = inv.transform((0.0, mt.y0))[1]
    at = axA.title.get_window_extent(renderer=fig.canvas.get_renderer())
    at_top = inv.transform((0.0, at.y1))[1]
    gap_main = (mt_bottom - at_top) * H
    print(f'[measure] 主标题->A标题 = {gap_main:.2f} inch')

    y_gcn_disp = axA.transData.transform((0.0, 4.9))[1]   # GCN 行底边（数据 y=4.9）
    y_gcn_frac = inv.transform((0.0, y_gcn_disp))[1]
    tb = axB.title.get_window_extent(renderer=fig.canvas.get_renderer())
    gap_ab_content = (y_gcn_frac - inv.transform((0.0, tb.y1))[1]) * H

    c_xlabel = axC.xaxis.label.get_window_extent(renderer=fig.canvas.get_renderer())
    c_xlabel_bottom = inv.transform((0.0, c_xlabel.y0))[1]
    td = axD.title.get_window_extent(renderer=fig.canvas.get_renderer())
    d_title_top = inv.transform((0.0, td.y1))[1]
    gap_bc_content = (c_xlabel_bottom - d_title_top) * H

    print(f'[measure] A-GCN->B标题 = {gap_ab_content:.2f} inch; C-AUROC->D标题 = {gap_bc_content:.2f} inch')

    for ext in ('pdf', 'svg', 'png'):
        out = os.path.join(SCRIPT_DIR, f'Fig2_main.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('saved:', out)

    print('DONE')


if __name__ == '__main__':
    main()
