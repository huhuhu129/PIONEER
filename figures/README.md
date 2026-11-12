# Main figure plotting scripts

This directory contains the plotting scripts that generate the manuscript's
main figures (Fig. 2–6), together with the input data each script reads.
All paths are relative to each script's own directory.

| Figure | Script | Topic |
|--------|--------|-------|
| Fig. 2 | `fig2/Fig2_combined.py` | Model robustness and stability (mouse benchmark, ablation, CV vs LOO, top-k retrieval) |
| Fig. 3 | `fig3/plot_fig3_main.py` | PIONEER vs PINNACLE tissue-specific PPI prediction benchmark |
| Fig. 4 | `fig4/fig4_main.py` | RNA / chromatin-accessibility ablation (complementary evidence) |
| Fig. 5 | `fig5/draw_fig5_nature.py` | Tissue specificity of predicted PPI networks (mouse & human) |
| Fig. 6 | `fig6/plot_figure_main.py` | Parkinson's disease transfer inference |

## Data layout

- `fig2/` — input JSON/CSV sit next to the script.
- `fig3/` — `data/` holds the comparison tables; `metrics/` holds the PIONEER
  gate fold metrics (`metrics.json`, one per training run directory).
- `fig4/` — ablation/GO/map CSV+JSON sit next to the script; `metrics/` holds
  the human+mouse ablation fold metrics (`metrics.json`, one per run directory).
- `fig5/` — `networks/` (mouse PIONEER pickles), `fig5_human/` (human pickles
  and tables), `fig5_data/` (mouse tables), `fig5_conserved_specific/` (GO).
- `fig6/` — `metrics/cv_metrics.json`, `prediction_full/` (predicted networks
  and enrichment tables), `proteomics/` (PXD024748 matrix).

## Run

Each script is run from its own directory, e.g.:

```bash
python fig5/draw_fig5_nature.py --data-root fig5
```

The scripts require `numpy`, `pandas`, `matplotlib`, `scipy`, and (for Fig. 5)
`networkx`. Fig. 3/4 also read `scikit-learn`-produced metrics but do not
import it directly.
