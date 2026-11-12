# -*- coding: utf-8 -*-
"""Draw Fig. 5 and its supplementary figure directly from local data.

Each output is ONE complete, multipanel figure on ONE canvas. No panel images
are pasted together and no separate panel files are written.

Run: python draw_fig5_nature.py
Requires: numpy, pandas, matplotlib, networkx (for reading trusted local pickles).
Outputs: output/pdf/{Fig5_main,FigS5_supplement,Fig5_main_nature,
                      FigS5_supplement_nature}.{pdf,svg,png}
PDF/SVG are editable vector artwork; PNG is a 600-dpi preview.
The default writes the requested A-H/title version AND a lowercase version with
the whole-figure title placed in a separate legend, for Nature production.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

MM = 1 / 25.4
WIDTH, HEIGHT = 180.0, 170.0
BLUE, GREY = "#2878A5", "#8B8F94"
ORANGE, TEAL, LIGHT = "#BA6734", "#4B9E8C", "#E8EDF1"
BLUE_LIGHT, ORANGE_LIGHT = "#A9C7DC", "#E0C5AF"
SPECIES = {
    "mouse": ("networks", "fig5_data", ["Brain", "Heart", "Kidney", "Lung", "Thymus"]),
    "human": ("fig5_human", "fig5_human", ["Heart", "Kidney", "Muscle", "Pancreas", "Spleen"]),
}
TISSUE_ABBR = {"Brain": "Br", "Heart": "He", "Kidney": "Ki", "Lung": "Lu",
               "Thymus": "Th", "Muscle": "Mu", "Pancreas": "Pa", "Spleen": "Sp"}
# Readable labels: the exact, unabridged GO term and ID are exported in source data.
SHORT_GO = {
    "GO:0002181": "Cytoplasmic translation",
    "GO:0043043": "Peptide biosynthesis",
    "GO:0045047": "Protein targeting to ER",
    "GO:0019646": "Aerobic electron transport chain",
    "GO:0042775": "ATP synthesis coupled electron transport",
    "GO:0006397": "mRNA processing",
    "GO:0006974": "Response to DNA damage",
    "GO:2000113": "Macromolecule biosynthesis (negative regulation)",
    "GO:0090630": "GTPase activation",
    "GO:0047497": "Mitochondrion transport along microtubule",
    "GO:0070125": "Mitochondrial translation elongation",
    "GO:0097435": "Supramolecular fiber organization",
    "GO:0045893": "Transcription (positive regulation)",
    "GO:0019941": "Modification-dependent protein catabolism",
    "GO:1905477": "Protein localization to membrane (positive regulation)",
    "GO:0010467": "Gene expression",
    "GO:0000398": "mRNA splicing via spliceosome",
}


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": .55, "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "text.color": "#20252A", "axes.labelcolor": "#20252A",
        "axes.edgecolor": "#50565D", "xtick.color": "#50565D", "ytick.color": "#50565D",
    })


def fig_text(fig, x, y, s, **kwargs):
    return fig.text(x / WIDTH, y / HEIGHT, s, **kwargs)


def axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])


def heading(fig, letter, text, x, y, lowercase):
    fig_text(fig, x, y, letter.lower() if lowercase else letter,
             fontsize=8, fontweight="bold", va="baseline")
    fig_text(fig, x + 7, y, text, fontsize=7, fontweight="bold", va="baseline")


def new_figure(title, lowercase):
    fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM))
    if not lowercase:
        fig_text(fig, 4, 166, title, fontsize=7, fontweight="bold", va="baseline")
    return fig


def norm_edges(graph):
    return {tuple(sorted((str(u).upper(), str(v).upper()))) for u, v in graph.edges()}


def incident_nodes(edges):
    return {n for e in edges for n in e}


def jac(a, b):
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def validate_matrix(df, tissues, symmetric=False):
    df = df.loc[tissues, tissues].astype(float)
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Non-finite matrix values")
    if symmetric and not np.allclose(df, df.T, atol=1e-10):
        raise ValueError("Expected a symmetric matrix")
    return df


def load_data(root):
    datasets, files, audit, tables = {}, [], [], []
    for species, (netdir, datadir, tissues) in SPECIES.items():
        data = {"species": species, "tissues": tissues}
        scale_path = root / datadir / "scale.csv"
        files.append(scale_path)
        data["scale"] = pd.read_csv(scale_path, index_col=0).loc[tissues]
        for name in ["jaccard", "shared_edges", "shared_pct", "gold_shared_pct"]:
            path = root / datadir / f"{name}.csv"
            files.append(path)
            data[name] = validate_matrix(pd.read_csv(path, index_col=0), tissues,
                                         symmetric=name in ["jaccard", "shared_edges"])
        graphs, edges = {}, {}
        for t in tissues:
            path = root / netdir / f"{t}_PIONEER.pickle"
            files.append(path)
            # Only load the trusted, user-provided local graph files.
            with path.open("rb") as handle:
                graphs[t] = pickle.load(handle)
            edges[t] = norm_edges(graphs[t])
            sc = data["scale"].loc[t]
            if len(edges[t]) != int(sc.pred_edges) or len(incident_nodes(edges[t])) != int(sc.pred_nodes):
                raise ValueError(f"Network/scale mismatch: {species} {t}")
        counts = Counter(edge for tissue_edges in edges.values() for edge in tissue_edges)
        common_nodes = set.intersection(*(incident_nodes(edges[t]) for t in tissues))
        common_edges = {t: {e for e in edges[t] if set(e) <= common_nodes} for t in tissues}
        degree = {t: {str(n).upper(): v for n, v in graphs[t].degree()} for t in tissues}
        all_genes = sorted(set().union(*(set(d) for d in degree.values())))
        ds = np.array([[degree[t].get(g, 0) for t in tissues] for g in all_genes], dtype=float)
        z = (ds - ds.mean(axis=1, keepdims=True)) / (ds.std(axis=1, keepdims=True) + 1e-8)
        data.update(edges=edges, common_nodes=common_nodes, common_edges=common_edges,
                    counts=counts, hub_counts={})
        composition, pairs, summaries = [], [], []
        for i, t in enumerate(tissues):
            total = len(edges[t])
            composition.append([sum(counts[e] == 1 for e in edges[t]),
                                sum(2 <= counts[e] <= 4 for e in edges[t]),
                                sum(counts[e] == 5 for e in edges[t])])
            data["hub_counts"][t] = int(((ds.argmax(1) == i) & (z[:, i] > 1.5)
                                        & ((ds > 0).sum(1) >= 2)).sum())
            summaries.append({
                "species": species, "tissue": t, "pred_nodes": len(incident_nodes(edges[t])),
                "pred_edges": total, "reference_nodes": int(data["scale"].loc[t, "gold_nodes"]),
                "reference_edges": int(data["scale"].loc[t, "gold_edges"]),
                "unique_edges": composition[-1][0], "shared_2_to_4_edges": composition[-1][1],
                "shared_all_5_edges": composition[-1][2],
                "shared_at_least_2_pct": 100 * (1 - composition[-1][0] / total),
                "pred_mean_pairwise_shared_pct": data["shared_pct"].loc[t].drop(t).mean(),
                "reference_mean_pairwise_shared_pct": data["gold_shared_pct"].loc[t].drop(t).mean(),
                "tissue_biased_hubs": data["hub_counts"][t],
            })
            for j, other in enumerate(tissues):
                expected = jac(edges[t], edges[other])
                if not np.isclose(expected, data["jaccard"].loc[t, other], atol=1e-10):
                    raise ValueError(f"Jaccard mismatch: {species} {t}/{other}")
                shared = len(edges[t] & edges[other])
                if shared != data["shared_edges"].loc[t, other]:
                    raise ValueError(f"Shared-edge count mismatch: {species} {t}/{other}")
                if not np.isclose(shared / total * 100, data["shared_pct"].loc[t, other]):
                    raise ValueError(f"Shared percentage mismatch: {species} {t}/{other}")
                # The reference graphs are absent; check that directed percentages
                # imply symmetric, integer intersections using reference edge counts.
                ginter = data["gold_shared_pct"].loc[t, other] * data["scale"].loc[t, "gold_edges"] / 100
                ginter_other = data["gold_shared_pct"].loc[other, t] * data["scale"].loc[other, "gold_edges"] / 100
                if not np.isclose(ginter, ginter_other) or not np.isclose(ginter, round(ginter)):
                    raise ValueError("Reference percentages are internally inconsistent")
                if j > i:
                    pairs.append({"species": species, "tissue_1": t, "tissue_2": other,
                                  "jaccard_all": expected,
                                  "jaccard_common_nodes": jac(common_edges[t], common_edges[other]),
                                  "common_node_count": len(common_nodes), "shared_edges": shared})
        data["composition"] = np.asarray(composition, dtype=float)
        data["pairs"] = pd.DataFrame(pairs)
        data["summary"] = pd.DataFrame(summaries)
        conserved = set.intersection(*edges.values())
        audit.append({"species": species, "common_nodes": len(common_nodes),
                      "common_universe_edges_by_tissue": {t: len(common_edges[t]) for t in tissues},
                      "conserved_edges": len(conserved), "conserved_endpoint_genes": len(incident_nodes(conserved)),
                      "median_jaccard_all": float(data["pairs"].jaccard_all.median()),
                      "median_jaccard_common_nodes": float(data["pairs"].jaccard_common_nodes.median()),
                      "unique_edge_multiplicity_counts": {str(k): v for k, v in sorted(Counter(counts.values()).items())},
                      "all_predicted_edges_exactly_2x_reference": bool((data["scale"].pred_edges == 2 * data["scale"].gold_edges).all()),
                      "reference_graphs_available": False,
                      "edge_budget_matched": False})
        datasets[species] = data
        tables.extend(summaries)
    return datasets, files, audit, pd.DataFrame(tables)


def short_go(term):
    match = re.search(r"GO:\d+", term)
    if match and match.group() in SHORT_GO:
        return SHORT_GO[match.group()]
    return re.sub(r"\s*\(GO:\d+\)", "", term).capitalize()


def load_go(root, datasets):
    selected, files = [], []
    for species, data in datasets.items():
        for selection, filename, tissues, number in [
            ("conserved", f"{species}_conserved_go.csv", [None], 3),
            ("biased_hub", f"{species}_tissue_specific_hub_go_{{tissue}}.csv", data["tissues"], 1),
            ("unique_edge", f"{species}_specific_go_{{tissue}}.csv", data["tissues"], 1),
        ]:
            for tissue in tissues:
                path = root / "fig5_conserved_specific" / filename.format(tissue=tissue)
                files.append(path)
                df = pd.read_csv(path)
                df = df.loc[df["Adjusted P-value"] < .05].sort_values("Adjusted P-value", kind="stable").head(number)
                for _, row in df.iterrows():
                    selected.append({"species": species, "tissue": tissue or "All five tissues",
                                     "gene_set_definition": selection, "source_file": str(path.relative_to(root)),
                                     "full_term": row["Term"], "display_term": short_go(row["Term"]),
                                     "adjusted_p": float(row["Adjusted P-value"]),
                                     "raw_p": float(row["P-value"]), "overlap": row["Overlap"],
                                     "genes": row["Genes"],
                                     "hub_input_count": data["hub_counts"].get(tissue) if selection == "biased_hub" else None})
    return pd.DataFrame(selected), files


def prediction_legend(fig, x, y):
    handles = [Patch(facecolor=BLUE, label="Prediction"), Patch(facecolor=GREY, label="Reference")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x / WIDTH, y / HEIGHT),
               ncol=2, fontsize=5.5, frameon=False, handlelength=1.1,
               columnspacing=1.3, borderaxespad=0)


def count_bars(fig, data, x, y, column="edges"):
    ax = axes_mm(fig, x + 12, y, 67, 20)
    tissues, sc = data["tissues"], data["scale"]
    positions = np.arange(len(tissues))
    divisor = 1000 if column == "edges" else 1
    ax.bar(positions - .19, sc[f"pred_{column}"] / divisor, .36, color=BLUE, linewidth=0)
    ax.bar(positions + .19, sc[f"gold_{column}"] / divisor, .36, color=GREY, linewidth=0)
    ax.set_xticks(positions, tissues, rotation=25, ha="right")
    ax.set_ylabel("Interactions ($\\times 10^3$)" if column == "edges" else "Proteins")
    ax.set_ylim(0, 80 if column == "edges" else 4200)
    ax.set_yticks([0, 20, 40, 60, 80] if column == "edges" else [0, 1000, 2000, 3000, 4000])
    ax.tick_params(axis="x", length=0, pad=2)
    prediction_legend(fig, x + 12, y + 21)
    return ax


def jaccard_heatmap(fig, data, x):
    arr = data["jaccard"].to_numpy()
    mask = np.triu(np.ones(arr.shape, dtype=bool))
    cmap = LinearSegmentedColormap.from_list("overlap", ["#F3F7F9", "#98C6DC", "#225C84"])
    cmap.set_bad("#F3F3F3")
    ax = axes_mm(fig, x + 23, 88, 27, 27)
    # Vector cells: imshow would silently rasterize the heatmap in PDF/SVG.
    ax.set_facecolor("#F3F3F3")
    im = ax.pcolormesh(np.arange(6) - .5, np.arange(6) - .5,
                       np.ma.array(arr, mask=mask), vmin=0, vmax=.20, cmap=cmap,
                       shading="flat", edgecolors="none", rasterized=False)
    ax.set_xlim(-.5, 4.5)
    ax.set_ylim(4.5, -.5)
    ax.set_aspect("equal")
    tissues = data["tissues"]
    ax.set_xticks(range(5), tissues, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(5), tissues)
    ax.tick_params(length=0, pad=2)
    for i in range(5):
        for j in range(i):
            value = arr[i, j]
            ax.text(j, i, f"{value:.3f}", fontsize=5.5, ha="center", va="center",
                    color="white" if value > .155 else "#20252A")
    for spine in ax.spines.values():
        spine.set_visible(False)
    med = data["pairs"].jaccard_all.median()
    fig_text(fig, x + 12, 117, f"10 tissue pairs; median = {med:.3f}", fontsize=5.5)
    return im


def pairwise_sharing(fig, data, x):
    ax = axes_mm(fig, x + 12, 50, 67, 19)
    tissues = data["tissues"]
    for k, tissue in enumerate(tissues):
        for name, offset, color, marker in [("shared_pct", -.16, BLUE, "o"),
                                            ("gold_shared_pct", .16, GREY, "s")]:
            values = data[name].loc[tissue].drop(tissue).to_numpy()
            jitter = np.linspace(-.08, .08, 4)
            ax.scatter(k + offset + jitter, values, s=5, marker=marker, alpha=.6,
                       color=color, edgecolor="none", zorder=3)
            # The horizontal segment is the descriptive arithmetic mean of four
            # directed partner comparisons, not an uncertainty interval.
            ax.plot([k + offset - .115, k + offset + .115], [values.mean()] * 2,
                    color=color, linewidth=1, zorder=4)
    ax.set_xticks(range(5), tissues, rotation=25, ha="right")
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 20, 40])
    ax.set_ylabel("Pairwise shared edges (%)")
    ax.tick_params(axis="x", length=0, pad=2)
    prediction_legend(fig, x + 12, 70)
    return ax


def go_chart(fig, rows, x, y=8, height=28, mode="conserved", xmax=20, light=False):
    # Reserve a label column as part of the same figure, instead of creating
    # separate panel exports. Full terms remain in Fig5_source_data_go.csv.
    label_width = 44 if x < 90 else 45
    ax = axes_mm(fig, x + label_width + 2, y, 79 - label_width - 2, height)
    rows = rows.reset_index(drop=True)
    n = len(rows)
    ax.set_ylim(-.5, max(n - .5, .5))
    ax.invert_yaxis()
    blue = BLUE_LIGHT if light else BLUE
    orange = ORANGE_LIGHT if light else ORANGE
    for i, row in rows.iterrows():
        p = row.adjusted_p
        value = -np.log10(max(p, 1e-300))
        color = blue if row.species == "mouse" else orange
        ax.hlines(i, 0, value, color=color, linewidth=.6)
        ax.scatter(value, i, s=13, color=color, zorder=3, linewidth=0)
        if mode == "conserved":
            label = ("m: " if row.species == "mouse" else "h: ") + row.display_term
        else:
            prefix = ("m" if row.species == "mouse" else "h") + row.tissue
            label = prefix + ": " + row.display_term
        label = textwrap.fill(label, width=39 if mode == "conserved" else 42,
                              break_long_words=False, break_on_hyphens=False)
        fy = y + height * (1 - (i + .5) / max(n, 1))
        fig_text(fig, x + label_width + 1, fy, label, ha="right", va="center",
                 fontsize=5.2, linespacing=1.0)
    ax.axvline(-np.log10(.05), color="#707070", linestyle=(0, (2, 2)), linewidth=.5, zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("$-\\log_{10}$ (adjusted $P$)", labelpad=1, fontsize=6)
    ticksets = {12: [0, 4, 8, 12], 20: [0, 5, 10, 15, 20], 30: [0, 10, 20, 30]}
    ax.set_xticks(ticksets[xmax])
    ax.tick_params(axis="x", labelsize=5.5)
    return ax


def main_figure(datasets, go, lowercase=False):
    fig = new_figure("Fig. 5 | Shared and tissue-restricted organization of predicted interactomes", lowercase)
    mouse, human = datasets["mouse"], datasets["human"]
    for data, x, letter in [(mouse, 4, "A"), (human, 96, "B")]:
        heading(fig, letter, data["species"].capitalize() + " network size", x, 157, lowercase)
        count_bars(fig, data, x, 132)
    heading(fig, "C", "Mouse tissue similarity", 4, 121, lowercase)
    heading(fig, "D", "Human tissue similarity", 96, 121, lowercase)
    jaccard_heatmap(fig, mouse, 4)
    im = jaccard_heatmap(fig, human, 96)
    cax = axes_mm(fig, 156, 88, 2.2, 27)
    cb = fig.colorbar(im, cax=cax, ticks=[0, .05, .10, .15, .20])
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
        # Close subpixel seams between adjacent vector colourbar cells.
        cb.solids.set_edgecolor("face")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=.4)
    cb.set_label("Edge-set Jaccard index", fontsize=6, labelpad=4)
    heading(fig, "E", "Mouse shared-edge comparison", 4, 77, lowercase)
    heading(fig, "F", "Human shared-edge comparison", 96, 77, lowercase)
    pairwise_sharing(fig, mouse, 4)
    pairwise_sharing(fig, human, 96)
    heading(fig, "G", "Functions of the shared core", 4, 38, lowercase)
    heading(fig, "H", "Functions of tissue-biased hubs", 96, 38, lowercase)
    go_chart(fig, go.loc[go.gene_set_definition == "conserved"], 4, xmax=20, light=True)
    go_chart(fig, go.loc[go.gene_set_definition == "biased_hub"], 96, mode="hub", xmax=12, light=True)
    fig_text(fig, 4, 1.2, "m, mouse; h, human; dashed line, adjusted P = 0.05", fontsize=5)
    fig_text(fig, 96, 1.2, "No retained terms: mBrain, mLung, hPancreas", fontsize=5)
    return fig


def edge_composition(fig, data, x):
    ax = axes_mm(fig, x + 12, 91, 67, 23)
    vals = data["composition"] / data["composition"].sum(1, keepdims=True) * 100
    bottom = np.zeros(5)
    colors = [BLUE, TEAL, ORANGE]
    for j in range(3):
        ax.bar(range(5), vals[:, j], bottom=bottom, width=.64, color=colors[j], linewidth=0)
        bottom += vals[:, j]
    for i in range(5):
        ax.text(i, vals[i, 0] / 2, f"{vals[i, 0]:.1f}", ha="center", va="center",
                fontsize=5.5, color="white")
    ax.set_xticks(range(5), data["tissues"], rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Edges within each tissue (%)")
    ax.tick_params(axis="x", length=0, pad=2)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, ["One tissue", "2-4 tissues", "All five"])]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=((x + 12) / WIDTH, 115.5 / HEIGHT),
               ncol=3, fontsize=5.2, frameon=False, handlelength=.9, columnspacing=1,
               borderaxespad=0)
    return ax


def common_universe_chart(fig, data, x):
    ax = axes_mm(fig, x + 22, 48, 57, 27)
    pairs = data["pairs"]
    for i, row in pairs.iterrows():
        a, b = row.jaccard_all, row.jaccard_common_nodes
        ax.plot([a, b], [i, i], color="#BBC0C5", linewidth=.6)
        ax.scatter(a, i, s=8, color=GREY, marker="o", linewidth=0, zorder=3)
        ax.scatter(b, i, s=11, color=TEAL, marker="D", linewidth=0, zorder=3)
    labels = [TISSUE_ABBR[r.tissue_1] + "-" + TISSUE_ABBR[r.tissue_2] for _, r in pairs.iterrows()]
    ax.set_yticks(range(10), labels, fontsize=5.5)
    ax.set_ylim(9.6, -.6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .50, .75, 1])
    ax.set_xlabel("Edge-set Jaccard index", labelpad=1)
    ax.tick_params(axis="y", length=0, pad=2)
    fig_text(fig, x + 12, 78, f"Common proteins: {len(data['common_nodes']):,}", fontsize=5.5)
    return ax


def supp_figure(datasets, go, lowercase=False):
    fig = new_figure("Fig. S5. Coverage, tissue restriction and functional sensitivity", lowercase)
    for data, x, letter in [(datasets["mouse"], 4, "A"), (datasets["human"], 96, "B")]:
        heading(fig, letter, data["species"].capitalize() + " protein coverage", x, 157, lowercase)
        count_bars(fig, data, x, 132, column="nodes")
    for data, x, letter in [(datasets["mouse"], 4, "C"), (datasets["human"], 96, "D")]:
        heading(fig, letter, data["species"].capitalize() + " edge composition", x, 121, lowercase)
        edge_composition(fig, data, x)
    for data, x, letter in [(datasets["mouse"], 4, "E"), (datasets["human"], 96, "F")]:
        heading(fig, letter, data["species"].capitalize() + " common-protein sensitivity", x, 82, lowercase)
        common_universe_chart(fig, data, x)
    fig.legend(handles=[Line2D([], [], color=GREY, marker="o", linestyle="none", markersize=2.5, label="All proteins"),
                        Line2D([], [], color=TEAL, marker="D", linestyle="none", markersize=2.5, label="Common proteins")],
               loc="lower center", bbox_to_anchor=(.5, 40 / HEIGHT), frameon=False, ncol=2,
               fontsize=5.5, handletextpad=.4, columnspacing=1.5, borderaxespad=0)
    for species, x, letter in [("mouse", 4, "G"), ("human", 96, "H")]:
        heading(fig, letter, species.capitalize() + " unique-edge endpoint functions", x, 36, lowercase)
        rows = go.loc[(go.gene_set_definition == "unique_edge") & (go.species == species)]
        go_chart(fig, rows, x, height=23.5, mode="unique", xmax=30)
    fig_text(fig, 4, 1.2, "No retained unique-edge endpoint terms: mouse Heart", fontsize=5)
    fig_text(fig, 96, 1.2, "Common-protein sensitivity does not match edge counts", fontsize=5)
    return fig


MAIN_LEGEND = """**Fig. 5 | Shared and tissue-restricted organization of predicted interactomes.**
**A, B,** Numbers of interactions in the supplied mouse and human predictions and
tissue reference networks. **C, D,** Pairwise predicted edge-set Jaccard indices;
ten unordered pairs per species. Upper triangles and self-comparisons are masked;
both panels use the same colour scale. **E, F,** Directed pairwise shared-edge
percentages, 100 × |E_i ∩ E_j| / |E_i|. Each point compares a focal tissue with one
of the other four tissues; horizontal segments are their arithmetic means. These
four comparisons are not independent biological replicates. **G,** The three
lowest-adjusted-P GO terms, per species, for genes incident to edges found in all
five tissues. **H,** The lowest-adjusted-P retained term for each tissue-biased hub
gene set. Hubs follow the supplied definition: maximum within-gene degree z-score
>1.5 across five tissues and nonzero degree in at least two tissues. No significant
terms were retained for mouse brain/lung or human pancreas. GO values are taken
from the supplied Enrichr GO Biological Process 2021 tables; dashed lines indicate
adjusted P = 0.05. m, mouse; h, human; ER, endoplasmic reticulum. Analyses use the
supplied network operating point; no edge-budget matching is implied.
"""

SUPP_LEGEND = """**Supplementary Fig. S5 | Coverage, tissue restriction and functional sensitivity.**
**A, B,** Protein counts for predictions and tissue reference networks. **C, D,**
Within-tissue predicted edges classified by occurrence in exactly one, two to four,
or all five tissues. Numerals label the percentage unique to one tissue; this is a
network-set definition and does not establish biological absence in other tissues.
**E, F,** Jaccard indices for each of ten unordered tissue pairs before and after
restricting both endpoints to proteins incident to predicted edges in all five
tissues of the same species. Grey circles, all proteins; green diamonds, common
proteins. Lines connect the same tissue pair. This sensitivity analysis is newly
computed from the supplied networks and does not match interaction counts.
Pair abbreviations: Br, brain; He, heart; Ki, kidney; Lu, lung; Th, thymus; Mu,
muscle; Pa, pancreas; Sp, spleen. **G, H,** The lowest-adjusted-P retained GO term
for genes incident to single-tissue-unique predicted edges in each tissue. No term
was retained for mouse heart. These endpoint gene sets can include genes present
in multiple tissues and are not tissue-exclusive genes. GO values come from the
supplied Enrichr GO Biological Process 2021 results. Dashed lines, adjusted
P = 0.05; m, mouse; h, human. Neither tissue pairs nor individual edges are
biological replicates.
"""


def save_complete(fig, out, stem, dpi):
    # Do not use bbox_inches='tight': it changes physical submission dimensions.
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--letter-case", choices=["both", "upper", "lower"], default="both")
    args = parser.parse_args()
    root = args.data_root.resolve()
    out = args.output.resolve() if args.output else root / "output" / "pdf"
    out.mkdir(parents=True, exist_ok=True)
    source = out.parent / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    configure_style()
    datasets, files, audit, summaries = load_data(root)
    go, go_files = load_go(root, datasets)
    summaries.to_csv(source / "Fig5_source_data_networks.csv", index=False)
    pd.concat([d["pairs"] for d in datasets.values()], ignore_index=True).to_csv(
        source / "FigS5_source_data_pairs.csv", index=False)
    go.to_csv(source / "Fig5_source_data_go.csv", index=False)
    directed_rows = []
    for sp, d in datasets.items():
        for a in d["tissues"]:
            for b in d["tissues"]:
                if a != b:
                    directed_rows.append({"species": sp, "focal_tissue": a, "partner_tissue": b,
                                          "prediction_shared_pct": d["shared_pct"].loc[a, b],
                                          "reference_shared_pct": d["gold_shared_pct"].loc[a, b]})
    pd.DataFrame(directed_rows).to_csv(source / "Fig5_source_data_sharing.csv", index=False)
    manifest = [{"file": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(set(files + go_files))]
    (source / "Fig5_audit.json").write_text(json.dumps({"audit": audit, "input_manifest": manifest}, indent=2), encoding="utf-8")
    (out.parent / "Fig5_legends.md").write_text(MAIN_LEGEND + "\n" + SUPP_LEGEND +
        "\nFor the *_nature files, change panel references to lowercase a-h. "
        "Supplementary numbering S5 is provisional: the manuscript already uses S5 for local hubs. "
        "Reconcile the number before submission.\n", encoding="utf-8")
    if args.letter_case in ["both", "upper"]:
        save_complete(main_figure(datasets, go), out, "Fig5_main", args.dpi)
        save_complete(supp_figure(datasets, go), out, "FigS5_supplement", args.dpi)
    if args.letter_case in ["both", "lower"]:
        save_complete(main_figure(datasets, go, True), out, "Fig5_main_nature", args.dpi)
        save_complete(supp_figure(datasets, go, True), out, "FigS5_supplement_nature", args.dpi)
    print("Saved complete figures to", out)
    print("Physical dimensions: 180 x 170 mm; vector PDF/SVG,", args.dpi, "dpi PNG")
    print(summaries.drop(columns=["pred_nodes", "reference_nodes"]).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
