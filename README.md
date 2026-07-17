# PIONEER: Protein InteractOme NEtwork rEconstruction fRamework

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
│   ├── human/                 # Human tissue preprocessed data (9 organs)
│   ├── mouse/                 # Mouse tissue preprocessed data (7 organs)
│   └── PD/                    # PD midbrain data for transfer learning
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

Gated cross-modal fusion and NT-Xent contrastive alignment integrate both regulatory layers. For disease inference, pre-trained normal brain ATAC weights are frozen while PD-derived scRNA-seq features replace normal RNA features.

## Data Sources

- **Mouse**: Tabula Muris scRNA-seq; ENCODE mouse scATAC-seq; PCP-SILAM gold-standard PPIs
- **Human**: Tabula Sapiens scRNA-seq; Human scATAC-Corpus; BICCN STRATOME (brain); IID gold-standard PPIs
- **PD**: GSE157783 (snRNA-seq); GSE147672 (snATAC-seq, post-hoc annotation only)
- **Global PPI scaffold**: STRING v12.0

## Reference

PIONEER: tissue- and disease-resolved protein interactome reconstruction via dual-view graph neural networks with single-cell multi-omics.

## License

[License to be added]
