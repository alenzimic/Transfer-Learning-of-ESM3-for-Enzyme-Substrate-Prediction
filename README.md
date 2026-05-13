# Transfer Learning of ESM3 for Enzyme Substrate Prediction

Nate Smith (nss97), Alen Zimic (amz63), Chase Holdener (ch2228), Darke Hull (drh257)

## Overview

This project evaluates whether frozen ESM3 representations can be reused for enzyme substrate prediction in BAHD acyltransferases. ESM3's native function tokens are too coarse to separate substrate-level BAHD activity, so this work treats ESM3 as a feature extractor and trains downstream multi-label classifiers for acceptor and donor substrate classes.

The main task is multi-label classification under strong class imbalance and limited labeled data. BAHD enzymes are used for training and cross-validation, while UGT glycosyltransferases are held out for cross-family generalization on shared acceptor classes.

## Repository Layout

```text
.
├── Datasets/
│   ├── BAHD_dataset.xlsx
│   └── UGT_dataset.tsv
├── modeling_pipeline/
│   ├── *.py                         # Feature extraction, model screening, baselines
│   ├── *.ipynb                      # Colab / analysis notebooks
│   ├── reference/                   # Reference ESM3 notebooks
│   ├── figures/                     # Final comparison figures
│   └── results/                     # Saved model metrics, predictions, and model artifacts
├── interpretability_pipeline/
│   ├── *.py                         # Residue-level and feature-position interpretability scripts
│   ├── analysis_outputs/            # Generated interpretability figures/tables
│   ├── final_analysis_plots/        # Final supplemental plots
│   └── interpretability_outputs/    # Per-protein attribution and visualization outputs
├── requirements.txt
└── README.md
```

The two timestamped Google Drive export folders were renamed to `modeling_pipeline/` and `interpretability_pipeline/` for readability. No source code was edited.

## Data and Feature Inputs

The curated source tables are included in `Datasets/`:

- `BAHD_dataset.xlsx`: curated BAHD enzyme-substrate metadata used for training and validation.
- `UGT_dataset.tsv`: curated UGT enzyme-substrate metadata used as a held-out cross-family test set.

Several scripts expect compact tensor caches generated during preprocessing:

- `modeling_pipeline/BAHD_dataset/labels.pt`
- `modeling_pipeline/UGT_dataset/labels.pt`
- `modeling_pipeline/BAHD_lastLayer_embeddings/*_hidden_layer_steps10.pt`
- `modeling_pipeline/UGT_lastLayer_embeddings/*_hidden_layer_steps10.pt`
- function-logit tensors referenced by the `function_logit_path` fields

Those tensor caches are not the same as the raw source tables. They are generated from the dataset tables, PDB structures, and ESM3 inference. Saved results and figures are included so the project can be reviewed without rerunning ESM3.

## Setup

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ESM3 feature extraction requires ESM model access and an `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=<your_huggingface_token>
```

## Main Workflows

Extract ESM3 last hidden layer features from PDB structures:

```bash
python modeling_pipeline/extract_esm3_hidden_layer.py \
  --input_dir path/to/pdb_structures \
  --output_dir modeling_pipeline/BAHD_lastLayer_embeddings \
  --num_steps 10 \
  --device cuda
```

Screen baseline classifiers on mean-pooled ESM3 features:

```bash
python modeling_pipeline/classifier_screen.py \
  --feature_type last_layer \
  --dataset BAHD \
  --task both \
  --cv_mode kfold \
  --n_splits 5
```

Train the final logistic-regression baseline on mean-pooled ESM3 last-layer embeddings:

```bash
python modeling_pipeline/train_lr_embedding_baseline.py \
  --dataset BAHD \
  --cv_mode kfold \
  --n_splits 5 \
  --random_seed 45 \
  --C 0.025787
```

Run the interpretability pipeline after placing the required tensor caches and final model pickles in the expected locations:

```bash
python interpretability_pipeline/run_full_analysis.py \
  --root_dir interpretability_pipeline \
  --metrics_dir modeling_pipeline/results/lr_embedding_baseline/lr_ll_final_5fold_run00_20260509_200740 \
  --predictions_dir modeling_pipeline/results/lr_embedding_baseline/lr_ll_final_5fold_run00_20260509_200740
```

## Key Results

Final BAHD 5-fold cross-validation results from the last-hidden-layer logistic-regression model:

| Task | Samples | Labels | Micro-AUPR | Macro-AUPR | Micro-AUROC | Macro-AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BAHD acceptor superclass | 366 | 23 | 0.736 | 0.608 | 0.924 | 0.915 |
| BAHD donor type | 366 | 2 | 0.964 | 0.964 | 0.962 | 0.963 |

Relevant saved outputs:

- Final BAHD CV metrics: `modeling_pipeline/results/lr_embedding_baseline/lr_ll_final_5fold_run00_20260509_200740/`
- Final full BAHD model artifacts: `modeling_pipeline/results/lr_embedding_baseline/lr_ll_final_full_run00_20260509_201316/`
- UGT comparison figures: `modeling_pipeline/figures/`
- Interpretability figures: `interpretability_pipeline/analysis_outputs/writeup_figures/`
- Discriminative feature-position tables: `interpretability_pipeline/analysis_outputs/discriminative_features_full/`

## Method Summary

BAHD PDB structures were passed through ESM3 inference to produce two representation types:

- Function logits (FL): per-residue function-track outputs.
- Last hidden layer embeddings (LL): final trunk hidden states before track-specific logits.

Per-residue tensors were mean-pooled across sequence length to produce fixed-length protein representations. Linear, tree-based, MLP, and transformer-based classifiers were screened, then refined with random hyperparameter search. The final model was a regularized logistic regression over mean-pooled last hidden layer embeddings.

For interpretability, the final linear model was decomposed into per-feature and per-residue contributions. A second probing analysis evaluated unpooled ESM3 hidden-state features across relative residue-position bins, revealing localized substrate-specific signals that mean pooling partially obscured.

## AI Disclosure

Generative AI was used in this assignment as a tool for analysis, coding, syntax, grammar, and debugging.

## Glossary

- Enzyme: A protein that catalyzes a specific chemical reaction.
- Substrate: The input molecule an enzyme acts on. BAHD reactions include acceptor and donor substrates.
- BAHD acyltransferases: A plant enzyme family that transfers acyl groups from donor molecules to diverse acceptor scaffolds.
- UGT glycosyltransferases: An enzyme family that attaches sugar groups to acceptor molecules; used here as a held-out cross-family test set.
- CoA: Coenzyme A, a carrier molecule that activates acyl groups for transfer.
- SMILES: A text representation of molecular structure used to encode substrate identity.

## References

- Hayes, T. et al. (2025). Simulating 500 million years of evolution with a language model. Science, 387(6736), 850-858. https://doi.org/10.1126/science.ads0018
- Kim, H. W. et al. (2021). NPClassifier: A deep neural network-based structural classification tool for natural products. Journal of Natural Products, 84(11), 2795-2807. https://doi.org/10.1021/acs.jnatprod.1c00399
- Kruse, L. H. et al. (2022). Orthology-based analysis helps map evolutionary diversification and predict substrate class use of BAHD acyltransferases. The Plant Journal, 111(5), 1453-1468. https://doi.org/10.1111/tpj.15902
- Pandala, S. R. (2019). LazyPredict (Version 0.3.0). https://github.com/shankarpandala/lazypredict
- Smith, N., Yuan, X., Melissinos, C., & Moghe, G. D. (2025). FuncFetch: An LLM-assisted workflow enables mining thousands of enzyme-substrate interactions from published manuscripts. Bioinformatics, 41(1), btae756. https://doi.org/10.1093/bioinformatics/btae756
