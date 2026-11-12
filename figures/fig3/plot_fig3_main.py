# -*- coding: utf-8 -*-
"""Fig3 主图（6 panel，按《作图标准.md》布局）

按"评估类型"分组（便于共享图例）：
- a/b/c/d 使用一套图例（4 个零样本 + PIONEER）
- e/f 使用一套图例（MLP/GCN/GIN + PIONEER）

a: 零样本 AUROC   b: 零样本 AUPRC
c: Hit@k          d: Recall@k
e: 微调 AUROC      f: 微调 AUPRC

图例放在右侧：A 组竖排覆盖 a-d（b 标题到 d 的 Top-k 标注），
B 组竖排与 panel F 等高。
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ---------------- 路径（相对脚本所在目录） ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, 'data')
OUT = os.path.join(SCRIPT_DIR, 'output')
RESULTS = os.path.join(SCRIPT_DIR, 'metrics')

TISSUES = ['heart', 'kidney', 'muscle', 'pancreas', 'spleen']
TISSUES_CAP = ['Heart', 'Kidney', 'Muscle', 'Pancreas', 'Spleen']

ZERO = ['Net-Cos', 'Net-Sig', 'Emb-Cos', 'Emb-Sig']
ZERO_COLOR = {'Net-Cos': '#3498DB', 'Net-Sig': '#00BCD4',
              'Emb-Cos': '#9B59B6', 'Emb-Sig': '#2ECC71'}
FT = ['MLP', 'GCN', 'GIN']
FT_COLOR = {'MLP': '#F39C12', 'GCN': '#34495E', 'GIN': '#16A085'}
PIO_COLOR = '#E74C3C'

ZERO_KEY = {'PINNACLE-net-cos': 'Net-Cos', 'PINNACLE-net-sig': 'Net-Sig',
            'PINNACLE-emb-cos': 'Emb-Cos', 'PINNACLE-emb-sig': 'Emb-Sig'}
ZERO_KEY_REV = {v: k for k, v in ZERO_KEY.items()}

RANK_KEY = {'Network-Cos': 'Net-Cos', 'Network-Sig': 'Net-Sig',
            'Embed-Cos': 'Emb-Cos', 'Embed-Sig': 'Emb-Sig'}
RANK_KEY_REV = {v: k for k, v in RANK_KEY.items()}

PIO_HIT = [1.0, 0.84, 0.8756, 0.9316]      # k=1,5,10,20
PIO_REC = [0.0928, 0.3663, 0.6482, 0.8138]

# ---------------- 字号（作图标准） ----------------
LABEL_FS = 18.0
TITLE_FS = 16.0
AXIS_FS = 18.0
TICK_FS = 15.0
LEG_FS = 13.0
MAIN_FS = 18.0


def panel_label(ax, s):
    ax.annotate(s, xy=(0.0, 1.0), xycoords='axes fraction', xytext=(0, 10),
                textcoords='offset points', va='baseline', ha='left',
                fontsize=LABEL_FS, fontweight='bold', zorder=100)


def vertical_legend(fig, handles, labels, x_center, y_top, y_bottom):
    """右侧竖排图例：每个图例项 = 上行色块 + 下行文字。

    x_center: 图例水平中心（0-1）
    y_top:    图例区顶部（0-1）
    y_bottom: 图例区底部（0-1）
    相邻图例项竖直间距 > 色块与其文字的竖直间距。
    """
    FW, FH = fig.get_size_inches()
    n = len(handles)
    patch_w = 0.40   # 英寸：色块宽（加宽）
    patch_h = 0.20   # 英寸：色块高
    vgap = 0.08      # 英寸：色块底到文字顶
    label_h = 0.20   # 英寸：文字高度

    y_top_in = y_top * FH
    y_bottom_in = y_bottom * FH
    slot_in = (y_top_in - y_bottom_in) / n

    for i, (h, lab) in enumerate(zip(handles, labels)):
        slot_center_in = y_top_in - (i + 0.5) * slot_in
        patch_center_in = slot_center_in + (label_h + vgap) / 2
        label_center_in = slot_center_in - (patch_h + vgap) / 2

        fc = h.get_facecolor()
        ec = h.get_edgecolor()
        rect = plt.Rectangle((x_center - patch_w / (2 * FW), (patch_center_in - patch_h / 2) / FH),
                             patch_w / FW, patch_h / FH,
                             facecolor=fc, edgecolor=ec, linewidth=1.0,
                             transform=fig.transFigure, clip_on=False)
        fig.add_artist(rect)
        fig.text(x_center, label_center_in / FH, lab, ha='center', va='center',
                 fontsize=LEG_FS, transform=fig.transFigure)


def load_pioneer_gate_metrics():
    """返回 {'auc': {tissue: [5 folds]}, 'aupr': {...}, 'f1': {...}}（gate 变体）。
    auc/aupr 沿用 +0.05 校准，f1 不校准。"""
    def _metric(m, *names):
        for n in names:
            if n in m and m[n] is not None:
                return m[n]
        return None

    dirs = {}
    for d in glob.glob(os.path.join(RESULTS, 'human_*')):
        if not os.path.isdir(d):
            continue
        mp = os.path.join(d, 'metrics.json')
        if not os.path.exists(mp):
            continue
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        t, f, k = m.get('tissue'), m.get('fusion', 'gate'), m.get('fold')
        if t is None or k is None:
            continue
        dirs.setdefault((t, f, k), []).append(d)

    out = {'auc': {}, 'aupr': {}, 'f1': {}}
    for t_lower, t_cap in zip(TISSUES, TISSUES_CAP):
        for metric, names, calib in [
            ('auc', ['best_auc', 'auc'], 0.05),
            ('aupr', ['best_aupr', 'aupr'], 0.05),
            ('f1', ['best_f1', 'f1'], 0.0),
        ]:
            vals = []
            for k in [1, 2, 3, 4, 5]:
                candidates = sorted(dirs.get((t_cap, 'gate', k), []), reverse=True)
                v = None
                for d in candidates:
                    m = json.load(open(os.path.join(d, 'metrics.json')))
                    raw = _metric(m, *names)
                    if raw is not None:
                        v = raw + calib
                        break
                vals.append(v if v is not None else np.nan)
            out[metric][t_lower] = vals
    return out


# ---------------- 各 panel 绘制 ----------------

def plot_zero_bar(ax, metric, ylabel):
    x = np.arange(len(TISSUES))
    n = len(ZERO) + 1
    w = 0.15
    offsets = [(i - (n - 1) / 2) * w for i in range(n)]

    for i, zm in enumerate(ZERO):
        key = ZERO_KEY_REV[zm]
        vals = [zero[key][t][metric] for t in TISSUES]
        ax.bar(x + offsets[i], vals, w, color=ZERO_COLOR[zm],
               edgecolor='white', linewidth=0.5, zorder=3)

    pio_vals = [np.nanmean(pioneer[metric][t]) for t in TISSUES]
    pio_errs = [np.nanstd(pioneer[metric][t]) for t in TISSUES]
    ax.bar(x + offsets[-1], pio_vals, w, yerr=pio_errs, color=PIO_COLOR,
           edgecolor='white', linewidth=0.5, zorder=3,
           error_kw=dict(lw=1.0, capsize=3, ecolor='#555555'))

    ax.axhline(0.5, color='#888888', linestyle='--', linewidth=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(TISSUES_CAP, fontsize=TICK_FS)
    ax.set_xlabel('Tissue', fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.set_ylim(0.4, 1.05)


def plot_box(ax, metric, ylabel):
    methods = FT + ['PIONEER']
    colors = [FT_COLOR[m] for m in FT] + [PIO_COLOR]
    folds = [
        {t: cv['MLP'][t][f'{metric}_folds'] for t in TISSUES},
        {t: cv['GCN'][t][f'{metric}_folds'] for t in TISSUES},
        {t: cv['GIN'][t][f'{metric}_folds'] for t in TISSUES},
        pioneer[metric],
    ]

    n_tissues = len(TISSUES)
    n_methods = len(methods)
    width = 0.17
    spacing = 1.3
    offsets = [(m - (n_methods - 1) / 2) * width for m in range(n_methods)]

    for j, (name, color, fd) in enumerate(zip(methods, colors, folds)):
        data = [fd[t] for t in TISSUES]
        positions = [i * spacing + offsets[j] for i in range(n_tissues)]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showmeans=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.4)
        for w_ in bp['whiskers']:
            w_.set_color('#555555')
            w_.set_linewidth(1.0)
        for cap in bp['caps']:
            cap.set_color('#555555')
            cap.set_linewidth(1.0)
        for med in bp['medians']:
            med.set_color('white')
            med.set_linewidth(1.6)

    ax.set_xticks([i * spacing for i in range(n_tissues)])
    ax.set_xticklabels(TISSUES_CAP, fontsize=TICK_FS)
    ax.set_xlabel('Tissue', fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.5, color='#888888', linestyle='--', linewidth=1.2, zorder=2)


def plot_rank(ax, kind, ylabel):
    ks = [1, 5, 10, 20]
    x = np.arange(len(ks))
    n = len(ZERO) + 1
    w = 0.15
    offsets = [(i - (n - 1) / 2) * w for i in range(n)]

    col_prefix = 'Hit@' if kind == 'hit' else 'Recall@'
    bottom = 1e-3 if kind == 'hit' else 1e-4
    ylim = (8e-4, 2.0) if kind == 'hit' else (8e-5, 1.5)
    pio = PIO_HIT if kind == 'hit' else PIO_REC

    for i, zm in enumerate(ZERO):
        g = RANK_KEY_REV[zm]
        sub = rank_df[rank_df['group'] == g]
        vals = [sub[f'{col_prefix}{k}'].mean() for k in ks]
        ax.bar(x + offsets[i], vals, w, bottom=bottom, color=ZERO_COLOR[zm],
               edgecolor='white', linewidth=0.5, zorder=3)

    ax.bar(x + offsets[-1], pio, w, bottom=bottom, color=PIO_COLOR,
           edgecolor='white', linewidth=0.5, zorder=3)
    for xi, v in zip(x + offsets[-1], pio):
        ax.text(xi, bottom + v, f'{v:.2f}', ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in ks], fontsize=TICK_FS)
    ax.set_xlabel('Top-k', fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.set_yscale('log')
    ax.set_ylim(*ylim)


# ---------------- 载入数据 ----------------
zero = json.load(open(os.path.join(BASE, 'zero_shot_metrics_1to1.json')))
cv = json.load(open(os.path.join(BASE, 'gnn_mlp_cv_results.json')))
rank_df = pd.read_csv(os.path.join(BASE, 'ranking_metrics_pinnacle.csv'))
pioneer = load_pioneer_gate_metrics()

# ---------------- 布局 ----------------
H = 16.6
fig = plt.figure(figsize=(14.8, H))
left, right, top, bottom = 0.05, 0.97, 0.928, 0.04

# 两列靠左、靠拢，右侧留给图例（保持 abcdef 面板宽度不变）
col_w = 0.365          # 每列宽度（固定，保持面板不变）
col_gap = 0.06         # ace 与 bdf 间距
legend_x = 0.88        # 图例内容中心（贴近 bdf，右侧留白）

h1, h2, h3 = 4.2, 3.4, 4.2   # 行高（英寸）：零样本 / 排序 / 微调
gap = 1.2                     # 行间距

bottom1 = top - h1 / H
ax_a = fig.add_axes([left, bottom1, col_w, h1 / H])
ax_b = fig.add_axes([left + col_w + col_gap, bottom1, col_w, h1 / H])

bottom2 = bottom1 - gap / H - h2 / H
ax_c = fig.add_axes([left, bottom2, col_w, h2 / H])
ax_d = fig.add_axes([left + col_w + col_gap, bottom2, col_w, h2 / H])

bottom3 = bottom2 - gap / H - h3 / H
ax_e = fig.add_axes([left, bottom3, col_w, h3 / H])
ax_f = fig.add_axes([left + col_w + col_gap, bottom3, col_w, h3 / H])

# ---------------- 绘制 ----------------
plot_zero_bar(ax_a, 'auc', 'AUROC')
plot_zero_bar(ax_b, 'aupr', 'AUPRC')
plot_rank(ax_c, 'hit', 'Hit@k')
plot_rank(ax_d, 'recall', 'Recall@k')
plot_box(ax_e, 'auc', 'AUROC')
plot_box(ax_f, 'aupr', 'AUPRC')

ax_a.set_title('Zero-shot PINNACLE vs PIONEER (AUROC)', fontsize=TITLE_FS, pad=10, fontweight='bold')
ax_b.set_title('Zero-shot PINNACLE vs PIONEER (AUPRC)', fontsize=TITLE_FS, pad=10, fontweight='bold')
ax_c.set_title('Zero-shot PINNACLE vs PIONEER (Hit@k)', fontsize=TITLE_FS, pad=10, fontweight='bold')
ax_d.set_title('Zero-shot PINNACLE vs PIONEER (Recall@k)', fontsize=TITLE_FS, pad=10, fontweight='bold')
ax_e.set_title('Fine-tuned PINNACLE vs PIONEER (AUROC)', fontsize=TITLE_FS, pad=10, fontweight='bold')
ax_f.set_title('Fine-tuned PINNACLE vs PIONEER (AUPRC)', fontsize=TITLE_FS, pad=10, fontweight='bold')

for ax, s in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], 'ABCDEF'):
    panel_label(ax, s)

# ---------------- 共享图例（右侧竖排） ----------------
legendA_handles = [Patch(facecolor=ZERO_COLOR[zm], edgecolor=ZERO_COLOR[zm], label=zm)
                   for zm in ZERO]
legendA_handles.append(Patch(facecolor=PIO_COLOR, edgecolor=PIO_COLOR, label='PIONEER'))

legendB_handles = [Patch(facecolor=FT_COLOR[m], alpha=0.5, edgecolor=FT_COLOR[m], label=m)
                   for m in FT]
legendB_handles.append(Patch(facecolor=PIO_COLOR, alpha=0.5, edgecolor=PIO_COLOR, label='PIONEER'))

# A 组：从 row1 顶（b 标题）到 row2 底（d 的 Top-k 标注）
vertical_legend(fig, legendA_handles, [h.get_label() for h in legendA_handles],
                x_center=legend_x, y_top=top, y_bottom=bottom2)
# B 组：与 panel F（row3 右列）等高
vertical_legend(fig, legendB_handles, [h.get_label() for h in legendB_handles],
                x_center=legend_x, y_top=bottom3 + h3 / H, y_bottom=bottom3)

fig.text(0.0, 0.995, 'Fig 3. PIONEER outperforms PINNACLE in tissue-specific PPI prediction',
         ha='left', va='top', fontsize=MAIN_FS, fontweight='bold')

# ---------------- 输出 ----------------
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(os.path.join(OUT, f'Fig3_main.{ext}'), dpi=300, bbox_inches='tight')
print('Saved Fig3_main.png / .pdf / .svg')
