# PIONEER reconstructs protein interactomes across tissues and disease states using dual-view graph learning of single-cell multi-omics

A dual-view graph neural network for reconstructing tissue- and disease-specific protein interactomes from matched tissue-level scRNA-seq and scATAC-seq datasets.

## Directory Structure

```
PIONEER/
├── model/
│   └── model.py               # GIN_DualNetwork_Contrastive model definition
├── train/
│   ├── train.py               # Single-tissue single-fold training
│   ├── train_all_folds.py     # 5-fold serial training
│   └── aggregate_results.py   # Aggregate 5-fold results to CSV
├── inference/
│   └── pd_predict.py          # PD-specific interactome inference
├── data/
│   ├── human/                 # Human tissue preprocessed data (Brain, Heart, Kidney, Muscle, Pancreas, Spleen)
│   ├── mouse/                 # Mouse tissue preprocessed data (Brain, Heart, Kidney, Liver, Lung, Muscle, Thymus)
│   └── PD/                    # PD midbrain data for transfer learning
├── figures/
│   ├── README.md              # Overview of the main figure plotting scripts
│   ├── fig2/                  # Fig. 2 robustness/stability script + data
│   ├── fig3/                  # Fig. 3 PINNACLE benchmark script + data
│   ├── fig4/                  # Fig. 4 RNA/ATAC ablation script + data
│   ├── fig5/                  # Fig. 5 tissue-specificity script + data
│   └── fig6/                  # Fig. 6 PD transfer script + data
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place preprocessed data in the `data/` directory with the following structure per tissue:

```
data/{species}/{Tissue}/
├── gene_mapping.json          # Gene name to index mapping
├── rna_features_tensor.pt     # scRNA-seq features (PyTorch tensor)
├── atac_features_tensor.pt    # scATAC-seq features (PyTorch tensor)
├── G_rna_subgraph.pickle      # RNA-guided PPI subgraph (NetworkX)
├── G_atac_subgraph.pickle     # ATAC-guided PPI subgraph (NetworkX)
├── G_tissue.pickle            # Gold-standard tissue PPI (NetworkX)
├── neg_samples.json           # Negative edge samples
└── fold_{1..5}/
    ├── G_tissue_train.pickle
    └── G_tissue_val.pickle
```

> **Example dataset included:** a complete mouse **Brain** dataset is provided
> at `data/mouse/Brain/` (with all 5 folds) so reviewers can run the code
> immediately, e.g. `python train.py --species mouse --tissue Brain --fold 1`.
> Other tissues must be downloaded and preprocessed following the Methods.

### 3. Train

Single tissue, single fold:
```bash
cd train
python train.py --species human --tissue Brain --fold 1 --fusion gate
```

5-fold cross-validation for one tissue:
```bash
cd train
python train_all_folds.py --species human --tissue Brain --fusion gate
```

### 4. Aggregate results

```bash
cd train
python aggregate_results.py
```

### 5. PD-specific inference

```bash
cd inference
python pd_predict.py
```

## Model Architecture

PIONEER uses dual GIN encoders to independently model:
- **ATAC-guided chromatin regulatory subgraph** (upstream: chromatin accessibility)
- **RNA-guided transcriptional expression subgraph** (downstream: gene transcription)

Gated cross-modal fusion integrates both regulatory layers. For disease inference, the model is trained on control (non-PD) midbrain RNA and ATAC features supervised by the human-brain PPI gold standard, then applied to PD RNA and ATAC features to predict the PD-specific interactome.

## Data Sources

- **Mouse (7 tissues)**: Brain, Heart, Kidney, Liver, Lung, Muscle, Thymus — Tabula Muris scRNA-seq; ENCODE mouse scATAC-seq; PCP-SILAM gold-standard PPIs
- **Human (6 tissues)**: Brain, Heart, Kidney, Muscle, Pancreas, Spleen — Tabula Sapiens scRNA-seq; Human scATAC-Corpus; BICCN STRATOME (brain); IID gold-standard PPIs
- **PD**: GSE157783 (snRNA-seq, model input); GSE148434 (snATAC-seq, model input)
- **Global PPI scaffold**: STRING v12.0

## Reference

PIONEER reconstructs protein interactomes across tissues and disease states using dual-view graph learning of single-cell multi-omics.

## License

[License to be added]
