# -*- coding: utf-8 -*-
"""
Fig4 主图（5 面板 a–e）
================================================================================
a  AUROC 消融箱线图（gate vs RNA-only vs ATAC-only，人类+小鼠 pooled，含显著性）
b  AUPRC 消融箱线图（同上）
c  边层面 Venn（金标准边：RNA-only / ATAC-only / 两者共同召回，总体 union）
d  GO 层面 Venn（RNA-only / ATAC-only 富集条目重叠）
e  GO 富集横置热图（固定组织顺序，RNA/ATAC 上下分组，横轴=GO 通路）

布局（遵循 作图标准.md，14.8 英寸画布）：
  行1：a | b（各半宽）
  行2：c | d（各半宽）
  行3：e（全宽热图）
输出：Fig4_main.pdf / .svg / .png
"""

import os
import json
import glob

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXPR5000_TR = os.path.join(HERE, 'metrics')

HUMAN_TISSUES = ['Heart', 'Kidney', 'Muscle', 'Pancreas', 'Spleen']
MOUSE_TISSUES = ['Brain', 'Heart', 'Kidney', 'Lung', 'Thymus']
FUSIONS = ['rna_only', 'atac_only', 'gate']
FUSION_LABEL = {'rna_only': 'RNA-only', 'atac_only': 'ATAC-only', 'gate': 'PIONEER'}
VAR_COLOR = {'rna_only': '#0F4D92', 'atac_only': '#C44E52', 'gate': '#2E9E44'}
METRIC_LABEL = {'auc': 'AUROC', 'aupr': 'AUPRC'}

LABEL_FS = 18.0   # 面板标签
TITLE_FS = 16.0   # 面板标题
AXIS_FS = 18.0    # 坐标轴标题
TICK_FS = 15.0    # 刻度
LEG_FS = 15.0     # 图例
ANNOT_FS = 13.0   # 注释/数值

MAIN_TITLE = 'Fig4. RNA and chromatin accessibility provide complementary evidence'

TITLE_A = 'AUROC'
TITLE_B = 'AUPRC'
TITLE_C = 'Reference edges'
TITLE_D = 'Enriched GO terms'
TITLE_E = 'Direct map recall'
TITLE_F = 'Functional divergence'
TITLE_G = 'Per-tissue edge complementarity'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
})


def panel_label(ax, s):
    ax.annotate(s.upper(), xy=(0.0, 1.0), xycoords='axes fraction', xytext=(0, 10),
                textcoords='offset points', va='baseline', ha='left',
                fontsize=LABEL_FS, fontweight='bold', zorder=100)


# ============================================================
# 数据收集：expr5000 统一新格式 metrics.json（人 + 鼠）
# ============================================================
def collect_fold_metrics():
    """{(species, tissue, fusion): {'auc': {fold: val}, 'aupr': {fold: val}}}"""
    data = {}
    for d in glob.glob(os.path.join(EXPR5000_TR, '*_*')):
        mp = os.path.join(d, 'metrics.json')
        if not os.path.exists(mp):
            continue
        try:
            m = json.load(open(mp, encoding='utf-8'))
        except Exception:
            continue
        sp, t, fu = m.get('species'), m.get('tissue'), m.get('fusion')
        fold, auc, aupr = m.get('fold'), m.get('best_auc'), m.get('best_aupr')
        if None in (sp, t, fu, fold, auc, aupr):
            continue
        if sp == 'human' and t not in HUMAN_TISSUES:
            continue
        if sp == 'mouse' and t not in MOUSE_TISSUES:
            continue
        if fu not in FUSIONS:
            continue
        key = (sp, t, fu)
        if key not in data:
            data[key] = {'auc': {}, 'aupr': {}}
        # 同一 fold 取最新（mtime 最大）
        for metric, val in (('auc', auc), ('aupr', aupr)):
            if fold not in data[key][metric] or os.path.getmtime(mp) > data[key].get('_mt', {}).get(fold, 0):
                data[key][metric][fold] = val
        data[key].setdefault('_mt', {})[fold] = os.path.getmtime(mp)
    return data


def collect_values(data, species, fusion, metric):
    vals = []
    ts = HUMAN_TISSUES if species == 'human' else MOUSE_TISSUES
    for t in ts:
        if (species, t, fusion) in data:
            vals.extend(data[(species, t, fusion)][metric].values())
    return vals


# ============================================================
# 面板 a / b：消融箱线图
# ============================================================
def draw_boxplot(ax, data, metric):
    species = ['human', 'mouse']
    x = np.arange(len(species))
    w = 0.26
    box_pos = {}
    all_vals = []
    for i, f in enumerate(FUSIONS):
        for j, sp in enumerate(species):
            vals = collect_values(data, sp, f, metric)
            pos = x[j] + (i - (len(FUSIONS) - 1) / 2) * w
            box_pos[(sp, f)] = pos
            all_vals.extend(vals)
            bp = ax.boxplot([vals], positions=[pos], widths=w * 0.8, patch_artist=True,
                            flierprops=dict(marker='o', markersize=3, markerfacecolor='#999999',
                                            markeredgecolor='none'),
                            medianprops=dict(color='black', lw=1.2),
                            boxprops=dict(facecolor=VAR_COLOR[f], alpha=0.55, edgecolor=VAR_COLOR[f], lw=1.0),
                            whiskerprops=dict(color=VAR_COLOR[f], lw=1.0),
                            capprops=dict(color=VAR_COLOR[f], lw=1.0))
            for p in bp['boxes']:
                p.set(facecolor=VAR_COLOR[f], alpha=0.55)

    y_max = max(all_vals)
    y_min = min(all_vals)
    y_range = y_max - y_min if y_max > y_min else 0.05

    # 显著性：gate vs 单模态（配对 t 检验，按 组织×折 配对）。
    # 每个 species 独立取最高点，星号横线紧贴各自箱体顶部、上下对齐。
    for sp in species:
        sp_top = max(max(collect_values(data, sp, f, metric)) for f in FUSIONS)
        for pi, (f_hi, f_lo) in enumerate([('gate', 'atac_only'), ('gate', 'rna_only')]):
            a, b = [], []
            ts = HUMAN_TISSUES if sp == 'human' else MOUSE_TISSUES
            for t in ts:
                k1, k2 = (sp, t, f_hi), (sp, t, f_lo)
                if k1 in data and k2 in data:
                    for fold in data[k1][metric]:
                        if fold in data[k2][metric]:
                            a.append(data[k1][metric][fold])
                            b.append(data[k2][metric][fold])
            if len(a) < 3:
                continue
            p = stats.ttest_rel(a, b).pvalue
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            x1, x2 = box_pos[(sp, f_lo)], box_pos[(sp, f_hi)]
            y = sp_top + y_range * (0.02 + pi * 0.08)
            ax.plot([x1, x2], [y, y], color='#333333', lw=0.9)
            ax.text((x1 + x2) / 2, y + y_range * 0.006, stars,
                    ha='center', va='bottom', fontsize=ANNOT_FS)

    ax.set_xticks(x)
    ax.set_xticklabels(['Human', 'Mouse'], fontsize=TICK_FS)
    ax.set_xlim(-0.5, 1.5)   # 固定 x 范围，箱体占 [6.8%, 93.2%]
    ax.set_ylabel(METRIC_LABEL[metric], fontsize=AXIS_FS)
    ax.set_ylim(top=y_max + y_range * 0.34)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.9, length=4)


# ============================================================
# 面板 e：直接 map 召回率对比（scRNA/scATAC/并集/PIONEER）
# ============================================================
MAP_METHODS = ['scRNA', 'scATAC', 'scRNA+scATAC', 'PIONEER']
MAP_COLOR = {'scRNA': '#0F4D92', 'scATAC': '#C44E52',
             'scRNA+scATAC': '#9467BD', 'PIONEER': '#DD8452'}
MAP_LABEL = {'scRNA': 'scRNA-map', 'scATAC': 'scATAC-map',
             'scRNA+scATAC': 'scRNA+scATAC', 'PIONEER': 'PIONEER'}


def draw_map_recall(ax):
    df = pd.read_csv(os.path.join(HERE, 'fig4_map_recall.csv'))
    species = ['人类', '小鼠']
    x = np.array([0, 0.8])
    w = 0.20
    box_pos = {}
    all_vals = []
    for i, m in enumerate(MAP_METHODS):
        for j, sp in enumerate(species):
            vals = df[(df['物种'] == sp) & (df['方法'] == m)]['召回率'].values * 100
            pos = x[j] + (i - (len(MAP_METHODS) - 1) / 2) * w
            box_pos[(sp, m)] = pos
            all_vals.extend(vals)
            bp = ax.boxplot([vals], positions=[pos], widths=w * 0.8, patch_artist=True,
                            flierprops=dict(marker='o', markersize=3, markerfacecolor='#999999',
                                            markeredgecolor='none'),
                            medianprops=dict(color='black', lw=1.2),
                            boxprops=dict(facecolor=MAP_COLOR[m], alpha=0.55, edgecolor=MAP_COLOR[m], lw=1.0),
                            whiskerprops=dict(color=MAP_COLOR[m], lw=1.0),
                            capprops=dict(color=MAP_COLOR[m], lw=1.0))
            for p in bp['boxes']:
                p.set(facecolor=MAP_COLOR[m], alpha=0.55)

    y_max = max(all_vals)
    y_min = min(all_vals)
    y_range = y_max - y_min if y_max > y_min else 0.05

    # 显著性：PIONEER vs 各 map 方法（配对 t 检验）
    for sp in species:
        pio = df[(df['物种'] == sp) & (df['方法'] == 'PIONEER')]['召回率'].values * 100
        for pi, m2 in enumerate(['scRNA+scATAC', 'scRNA', 'scATAC']):
            b = df[(df['物种'] == sp) & (df['方法'] == m2)]['召回率'].values * 100
            n = min(len(pio), len(b))
            if n < 3:
                continue
            p = stats.ttest_rel(pio[:n], b[:n]).pvalue
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            x1, x2 = box_pos[(sp, m2)], box_pos[(sp, 'PIONEER')]
            y = y_max + y_range * (0.06 + pi * 0.07)
            ax.plot([x1, x2], [y, y], color='#333333', lw=0.9)
            ax.text((x1 + x2) / 2, y + y_range * 0.006, stars,
                    ha='center', va='bottom', fontsize=ANNOT_FS)

    ax.set_xticks(x)
    ax.set_xticklabels(['Human', 'Mouse'], fontsize=TICK_FS)
    ax.set_ylabel('Recall (%)', fontsize=AXIS_FS)
    ax.set_ylim(top=y_max + y_range * 0.34)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.9, length=4)

    from matplotlib.patches import Patch
    handles = [Patch(color=MAP_COLOR[m], alpha=0.55, label=MAP_LABEL[m]) for m in MAP_METHODS]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4,
              fontsize=12, frameon=False, columnspacing=0.8, handletextpad=0.5)


# ============================================================
# 面板 f：功能分化（RNA-only 独有 vs ATAC-only 独有 GO 发散条形图）
# ============================================================
def draw_go_divergence(ax):
    df = load_go_data()
    df = df[df['species'] == 'human']
    piv = df.pivot_table(index='term', columns='variant',
                         values='Adjusted P-value', aggfunc='min')
    rna_p = piv['rna_only']
    atac_p = piv['atac_only']

    # 独有富集：该模态显著（adjP<0.05）、另一模态不显著/缺失
    rna_uniq = rna_p[(rna_p < 0.05) & (atac_p.fillna(1.0) >= 0.05)].sort_values()
    atac_uniq = atac_p[(atac_p < 0.05) & (rna_p.fillna(1.0) >= 0.05)].sort_values()

    rna_top = rna_uniq.head(7)
    atac_top = atac_uniq.head(7)

    rna_vals = -np.log10(rna_top.clip(lower=1e-300))
    atac_vals = -np.log10(atac_top.clip(lower=1e-300))

    n_rna = len(rna_top)
    n_atac = len(atac_top)
    gap = 1
    # 最显著（最长条）放每组顶部
    y_rna = np.arange(n_rna)[::-1]
    y_atac = np.arange(n_atac)[::-1] + n_rna + gap

    ax.barh(y_rna, rna_vals, height=0.62, color='#0F4D92', edgecolor='none', alpha=0.55)
    ax.barh(y_atac, atac_vals, height=0.62, color='#C44E52', edgecolor='none', alpha=0.55)

    labels = [short_term(t) for t in rna_top.index] + [short_term(t) for t in atac_top.index]
    ax.set_yticks(list(y_rna) + list(y_atac))
    ax.set_yticklabels(labels, fontsize=11)

    xmax = max(rna_vals.max(), atac_vals.max(), 4) * 1.4
    ax.set_xlim(0, xmax)
    ax.set_xlabel('-log10(adjusted p-value)', fontsize=AXIS_FS)

    # 分组标签：右端、各组垂直居中
    ax.text(xmax * 0.98, (n_rna - 1) / 2, 'RNA-only', ha='right', va='center',
            fontsize=12, color='#0F4D92', fontweight='bold')
    ax.text(xmax * 0.98, n_rna + gap + (n_atac - 1) / 2, 'ATAC-only', ha='right', va='center',
            fontsize=12, color='#C44E52', fontweight='bold')

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.9, length=4)


# ============================================================
# 面板 g：逐组织边互补（金标准边：RNA 独有 / 共享 / ATAC 独有）
# ============================================================
def draw_per_tissue_edges(ax):
    with open(os.path.join(HERE, 'fig4_edge_venn_data.json'), encoding='utf-8') as f:
        vd = json.load(f)
    pt = vd['per_tissue']
    tissues = ['Heart', 'Kidney', 'Muscle', 'Pancreas', 'Spleen']

    rna_uniq, shared, atac_uniq = [], [], []
    for t in tissues:
        d = pt[t]
        rna_uniq.append(d['gold_rna'] - d['gold_shared'])
        shared.append(d['gold_shared'])
        atac_uniq.append(d['gold_atac'] - d['gold_shared'])

    x = np.arange(len(tissues))
    w = 0.26
    ax.bar(x - w, rna_uniq, width=w, color='#0F4D92', edgecolor='none', alpha=0.55, label='RNA-only')
    ax.bar(x, shared, width=w, color='#2E9E44', edgecolor='none', alpha=0.55, label='Shared')
    ax.bar(x + w, atac_uniq, width=w, color='#C44E52', edgecolor='none', alpha=0.55, label='ATAC-only')

    ax.set_xticks(x)
    ax.set_xticklabels(tissues, fontsize=TICK_FS)
    ax.set_ylabel('Recovered reference edges', fontsize=AXIS_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.9, length=4)
    ax.legend(fontsize=13, frameon=False, loc='upper right')


# ============================================================
# 面板 c / d：二集 Venn（手动绘制，保证样式统一）
# ============================================================
def draw_venn2(ax, n_a, n_b, n_ab, label_a, label_b, color_a, color_b):
    """n_a: A 总数, n_b: B 总数, n_ab: 交集。固定坐标系，c/d 尺寸一致，画成圆。"""
    a_only = n_a - n_ab
    b_only = n_b - n_ab

    ra = np.sqrt(n_a)
    rb = np.sqrt(n_b)
    s = 0.62 / max(ra, rb)   # 最大圆半径固定，保证 c/d 尺寸一致
    ra *= s
    rb *= s
    d = (ra + rb) * 0.42    # 圆心距（重叠程度）

    xlim = (-1.15, 1.15)
    ylim = (-0.72, 0.95)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # 按轴实际长宽比补偿椭圆高度，使圆在屏幕上呈现为正圆
    bbox = ax.get_position()
    ax_w = bbox.width * ax.figure.get_figwidth()
    ax_h = bbox.height * ax.figure.get_figheight()
    aspect = (ax_w / (xlim[1] - xlim[0])) / (ax_h / (ylim[1] - ylim[0]))

    ea = Ellipse((-d / 2, 0), 2 * ra, 2 * ra * aspect, facecolor=color_a, alpha=0.32,
                 edgecolor=color_a, lw=1.6)
    eb = Ellipse((d / 2, 0), 2 * rb, 2 * rb * aspect, facecolor=color_b, alpha=0.32,
                 edgecolor=color_b, lw=1.6)
    ax.add_patch(ea)
    ax.add_patch(eb)

    top_y = max(ra, rb) * aspect + 0.24

    # 三个区域中心（数字居中，不贴边）：左独有 / 交集 / 右独有
    ax.text(-(ra + rb) / 2, 0, str(a_only), ha='center', va='center',
            fontsize=LABEL_FS, fontweight='bold')
    ax.text((ra + rb) / 2, 0, str(b_only), ha='center', va='center',
            fontsize=LABEL_FS, fontweight='bold')
    ax.text((ra - rb) / 2, 0, str(n_ab), ha='center', va='center',
            fontsize=LABEL_FS, fontweight='bold')

    # 集合名（圆上方）
    ax.text(-(d / 2 + ra * 0.45), top_y, label_a, ha='center', va='bottom',
            fontsize=TICK_FS, fontweight='bold', color=color_a)
    ax.text((d / 2 + rb * 0.45), top_y, label_b, ha='center', va='bottom',
            fontsize=TICK_FS, fontweight='bold', color=color_b)

    ax.axis('off')


def draw_edge_venn(ax):
    with open(os.path.join(HERE, 'fig4_edge_venn_data.json'), encoding='utf-8') as f:
        vd = json.load(f)
    o = vd['overall']
    draw_venn2(ax, o['gold_rna'], o['gold_atac'], o['gold_shared'],
               'RNA-only', 'ATAC-only', '#0F4D92', '#C44E52')


def draw_go_venn(ax):
    # go_enrichment_stats.csv：rna_only总数/atac_only总数/仅rna/仅atac/共享
    df = pd.read_csv(os.path.join(HERE, 'go_enrichment_stats.csv'))
    stat = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    rna_total = int(stat['rna_only总数'])
    atac_total = int(stat['atac_only总数'])
    shared = int(stat['共享(gate/共有)'])
    draw_venn2(ax, rna_total, atac_total, shared, 'RNA-only', 'ATAC-only', '#0F4D92', '#C44E52')


# ============================================================
# 面板 e / f / g：每个组织一个小热图
#   横轴 = RNA-only / ATAC-only，纵轴 = 该组织 top5 GO 通路
# ============================================================
GO_PREFIX = ['positive regulation of ', 'negative regulation of ',
             'regulation of ', 'cellular response to ', 'response to ']

# 常用 GO term 的简洁名（完整可读，不带省略号）
GO_SHORT = {
    'cell-cell adhesion via plasma-membrane adhesion molecules': 'cell-cell adhesion',
    'calcium-dependent cell-cell adhesion via plasma membrane cell adhesion molecules': 'Ca2+ cell adhesion',
    'potassium ion transport': 'K+ transport',
    'regulation of small GTPase mediated signal transduction': 'GTPase signaling',
    'cytokine-mediated signaling pathway': 'cytokine signaling',
    'B cell receptor signaling pathway': 'B-cell receptor',
    'antigen receptor-mediated signaling pathway': 'antigen receptor',
    'peptide biosynthetic process': 'peptide biosynthesis',
    'cellular protein metabolic process': 'protein metabolism',
    'nuclear-transcribed mRNA catabolic process, nonsense-mediated decay': 'mRNA catabolism',
    'SRP-dependent cotranslational protein targeting to membrane': 'SRP targeting',
    'cotranslational protein targeting to membrane': 'cotransl. targeting',
    'protein targeting to ER': 'ER targeting',
    'protein phosphorylation': 'protein phospho.',
    'cytoplasmic translation': 'cytoplasmic transl.',
}


def load_go_data():
    df_h = pd.read_csv(os.path.join(HERE, 'human_all_go_enrichment.csv'))
    df_h['species'] = 'human'
    df_m = pd.read_csv(os.path.join(HERE, 'mouse_all_go_enrichment.csv'))
    df_m['species'] = 'mouse'
    df = pd.concat([df_h, df_m], ignore_index=True)
    df['term'] = df['Term'].str.split('(').str[0].str.strip()
    df['neg_log_p'] = -np.log10(df['Adjusted P-value'].clip(lower=1e-300))
    return df


def short_term(t):
    t = t.strip()
    if t in GO_SHORT:
        return GO_SHORT[t]
    for p in GO_PREFIX:
        if t.startswith(p):
            t = t[len(p):]
            break
    if len(t) <= 18:
        return t
    t = t[:18]
    idx = t.rfind(' ')
    if idx > 6:
        t = t[:idx]
    return t


def tissue_top5_terms(df, species, tissue):
    sub = df[(df['species'] == species) & (df['tissue'] == tissue)]
    return (sub.groupby('term')['Adjusted P-value'].min()
            .nsmallest(5).index.tolist())


def tissue_heatmap_matrix(df, species, tissue, top5):
    sub = df[(df['species'] == species) & (df['tissue'] == tissue)]
    variants = ['rna_only', 'atac_only']
    mat = np.full((5, 2), np.nan)
    for i, term in enumerate(top5):
        for j, v in enumerate(variants):
            r = sub[(sub['term'] == term) & (sub['variant'] == v)]
            if len(r):
                mat[i, j] = -np.log10(r['Adjusted P-value'].clip(lower=1e-300).min())
    return mat


def draw_tissue_heatmap(ax, df, species, tissue, vmax):
    top5 = tissue_top5_terms(df, species, tissue)
    mat = tissue_heatmap_matrix(df, species, tissue, top5)

    cmap = plt.get_cmap('GnBu')
    cmap.set_bad('#f2f2f2')   # 空白单元格浅灰
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)

    # 数值标注（深色格子白字，浅色格子黑字，居中）
    for i in range(5):
        for j in range(2):
            if not np.isnan(mat[i, j]):
                c = 'white' if mat[i, j] > 0.55 * vmax else 'black'
                ax.text(j, i, f'{mat[i, j]:.1f}', ha='center', va='center',
                        fontsize=11, color=c)

    # x 轴（顶部）：RNA-only / ATAC-only
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['RNA-only', 'ATAC-only'], fontsize=10)
    ax.xaxis.set_ticks_position('top')

    # y 轴：top5 GO 通路（左对齐，从刻度向右绘制）
    ax.set_yticks([])
    for i, t in enumerate(top5):
        ax.text(-0.6, i, short_term(t), ha='left', va='center', fontsize=9)

    # 白色分隔线
    for x in [-0.5, 0.5, 1.5]:
        ax.axvline(x, color='white', lw=2)
    for y in [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]:
        ax.axhline(y, color='white', lw=2)

    # 隐藏边框，保留热图与文字
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-3.2, 1.5)   # 左侧留 GO 标签空间
    ax.set_ylim(-0.5, 4.5)

    return im


# ============================================================
# 主拼装
# ============================================================
def main():
    data = collect_fold_metrics()
    n_human = len([k for k in data if k[0] == 'human' and k[2] == 'gate'])
    print(f'[collect] 人类 gate 折数={n_human}, 总键数={len(data)}')

    H = 13.2
    fig = plt.figure(figsize=(14.8, H))
    left, right, top, bottom = 0.05, 0.97, 0.92, 0.04
    content_w = right - left

    # 行高（英寸）与间距
    ab_scale = 0.9   # A/B 缩小到 0.9 倍
    h_ab_in = 3.8 * ab_scale
    h_cd_in = 3.6
    h_e_in = 3.0   # e 高度缩小一点
    h_f_in = 2.8   # f 功能分化（发散条形图）
    gap_in = 0.9

    h_ab = h_ab_in / H
    h_cd = h_cd_in / H
    h_e = h_e_in / H
    h_f = h_f_in / H
    gap = gap_in / H

    # 行1：A 靠左、B 靠右（中间留共享图例，缩小 ab_scale 倍）
    ab_spread = 0.02   # A 左移 / B 右移的额外量
    wspace = 0.35
    col_w_total = content_w / (1 + wspace / 2)
    col_w = col_w_total / 2 * ab_scale

    # 行2：c/d 窄、e 宽（e 比 c/d 宽一点，等间距）
    gap3 = 0.03
    col_w_e = 0.34
    col_w_cd = (content_w - 2 * gap3 - col_w_e) / 2

    row1_top = top
    axA = fig.add_axes([left - ab_spread, row1_top - h_ab, col_w, h_ab])
    axB = fig.add_axes([right - col_w + ab_spread, row1_top - h_ab, col_w, h_ab])
    row2_top = row1_top - h_ab - gap
    axC = fig.add_axes([left - ab_spread, row2_top - h_cd, col_w_cd, h_cd])
    axD = fig.add_axes([left - ab_spread + col_w_cd + gap3, row2_top - h_cd, col_w_cd, h_cd])
    axE = fig.add_axes([left + 2 * (col_w_cd + gap3), row2_top - h_e, col_w_e, h_e])
    row3_top = row2_top - h_cd - gap
    gap_fg = 0.04
    col_w_fg = (content_w - gap_fg) / 2
    axF = fig.add_axes([left, row3_top - h_f, col_w_fg, h_f])
    axG = fig.add_axes([left + col_w_fg + gap_fg, row3_top - h_f, col_w_fg, h_f])

    draw_boxplot(axA, data, 'auc')
    draw_boxplot(axB, data, 'aupr')
    draw_go_venn(axC)
    draw_edge_venn(axD)
    draw_map_recall(axG)
    draw_go_divergence(axF)
    draw_per_tissue_edges(axE)

    # a/b 共享图例：放两图之间，上下排布（swatch-文字间距 > 条目间距）
    from matplotlib.patches import Patch
    handles = [Patch(color=VAR_COLOR[f], alpha=0.55, label=FUSION_LABEL[f]) for f in FUSIONS]
    leg_x = left + content_w / 2 - 0.02
    leg_y = row1_top - h_ab / 2
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(leg_x, leg_y),
               ncol=1, fontsize=LEG_FS, frameon=False,
               handletextpad=1.2, labelspacing=0.4)

    axA.set_title(TITLE_A, fontsize=TITLE_FS, fontweight='bold', pad=10)
    axB.set_title(TITLE_B, fontsize=TITLE_FS, fontweight='bold', pad=10)
    axC.set_title(TITLE_D, fontsize=TITLE_FS, fontweight='bold', pad=10)
    axD.set_title(TITLE_C, fontsize=TITLE_FS, fontweight='bold', pad=10)
    axE.set_title(TITLE_G, fontsize=TITLE_FS, fontweight='bold', pad=10)
    axF.set_title(TITLE_F, fontsize=TITLE_FS, fontweight='bold', pad=10)
    axG.set_title(TITLE_E, fontsize=TITLE_FS, fontweight='bold', pad=10)

    for ax, s in [(axA, 'a'), (axB, 'b'), (axC, 'c'), (axD, 'd'), (axE, 'e'), (axF, 'f'), (axG, 'g')]:
        panel_label(ax, s)

    fig.text(0.0, 0.995, MAIN_TITLE, ha='left', va='top',
             fontsize=LABEL_FS, fontweight='bold')

    for ext in ('pdf', 'svg', 'png'):
        out = os.path.join(HERE, f'Fig4_main.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('saved:', out)

    print('DONE')


if __name__ == '__main__':
    main()
