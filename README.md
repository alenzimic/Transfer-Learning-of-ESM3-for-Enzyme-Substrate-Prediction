# Transfer Learning of ESM3 for Enzyme-Substrate Prediction

Nate Smith (nss97), Alen Zimic (amz63), Chase Holdener (ch2228), Darke Hull (drh257)
CS 4782 — Spring 2026, Cornell University

## 1. Introduction

This repository contains the final project for CS 4782 (Deep Learning). Rather than reproducing a specific quantitative result from a single paper, we extend the **ESM3** protein foundation model (Hayes et al., 2025) to a more granular biochemical prediction task and analyze whether its intermediate representations encode information beyond what its native output tracks expose.

ESM3 is a 98B-parameter masked generative transformer pretrained on 2.78 billion proteins. Its main contribution is the joint, multi-track tokenization of protein sequence, 3D structure, and functional annotation, processed through a shared bidirectional encoder with rotary embeddings, SwiGLU, and SE(3)-invariant geometric attention. Its function track outputs InterPro-level annotations.

We use BAHD acyltransferases — a promiscuous plant enzyme family — as the prediction target. Predicting which substrate class(es) a given BAHD acts on is a multi-label problem with high class imbalance, limited labeled data (n = 366), and long input sequences (>400 residues).

## 2. Chosen Result (Extension Framing)

ESM3's native function-token vocabulary is coarse enough that it assigns essentially the same labels to all BAHDs (see Fig. 2 in `report/`), collapsing substrate-level diversity. **In lieu of replicating this result, we take it as our starting point** and instead ask:

1. Do ESM3's intermediate representations (function logits and last hidden layer embeddings) encode finer-grained substrate information that downstream classifiers can recover?
2. Can the resulting models yield interpretable, biochemically meaningful signals about which protein regions drive substrate specificity?

The reference point in the source paper is the function-track output (Hayes et al., 2025, Fig. 1 — multi-track decoding). Our project replaces that decoder with a task-specific transfer-learning pipeline (Fig. 3 in `report/`).

## 3. GitHub Contents

```text
.
├── README.md
├── code/
│   ├── requirements.txt
│   ├── modeling_pipeline/           # Feature extraction, model screening, baselines
│   └── interpretability_pipeline/   # Residue-level and feature-position interpretability scripts
├── data/
│   ├── BAHD_dataset.xlsx
│   ├── UGT_dataset.tsv
│   └── README.md
├── results/
│   ├── modeling_pipeline/           # Metrics, predictions, model artifacts, figures
│   ├── interpretability_pipeline/   # Generated interpretability figures and tables
│   └── README.md
├── poster/
│   ├── DL_poster.pdf
│   └── README.md
├── report/
│   ├── 117_ESM3Hayes2025_2page_report.pdf
│   ├── final_report.pdf
│   └── final_report.md
├── LICENSE
└── .gitignore
```

The original modeling and interpretability folders from our working repo were reorganized under `code/` and `results/` to match the required submission structure. Source code contents were not modified during reorganization.

## 4. Re-implementation Details

**Approach.** ESM3 is frozen and used as a feature extractor. Two intervention points are evaluated:

- **Function logits (FL)** — per-residue function-track outputs used by ESM3 to predict functional annotations.
- **Last hidden layer (LL) embeddings** — final hidden states from which all track-specific logits are computed.

Per-residue tensors are mean-pooled across sequence length to produce fixed-length protein representations. These are used as input features for multi-label classification of acceptor and donor substrate classes.

**Datasets.** Curated via the FuncFetch pipeline (Smith et al., 2025):

- BAHD acyltransferases: n = 366 (train/val, 5-fold CV)
- UGT glycosyltransferases: n = 482 (held-out cross-family test on 20 shared acceptor classes)

Each enzyme's experimentally verified substrates were represented as SMILES strings and clustered into categorical superclass labels using NPClassifier (Kim et al., 2021). Labels with fewer than 2 examples were dropped.

**Models screened.** A LazyPredict-inspired (Pandala, 2019) script screened linear, tree-based, and other classifiers with 5-fold CV. The top linear performer, a shallow MLP, and a transformer-based head were then compared via a 15× random hyperparameter search, with the best FL and LL configurations refined over 30 additional runs.

**Final model.** Regularized logistic regression on mean-pooled last-hidden-layer embeddings, evaluated by 5-fold and leave-one-out CV, then retrained on the full BAHD dataset and applied to the UGT held-out set.

**Primary metric.** BAHD-acceptor-micro-AUPR (BAμPR) — micro-averaging gives rare classes proportional weight, and AUPR is more discriminating than AUROC under severe imbalance.

**Interpretability.** Two complementary analyses:

1. Linear decomposition of each one-vs-rest score into per-residue and per-feature contributions (exact because pooling, standardization, and LR are all linear).
2. Direct probing of unpooled ESM3 hidden states across 50 relative-position bins, treating each feature-by-bin activation as a univariate classifier (AUROC + permutation tests). Cross-protein heatmaps and protein-length controls validate that signals are not explained by length confounding.

**Key modifications vs. naïve transfer learning.** No fine-tuning of the ESM3 encoder — dataset size made adapter or full fine-tuning likely to overfit. Mean pooling was retained for the primary classifier but explicitly bypassed during interpretability probing to recover localized signals it obscures.

## 5. Reproduction Steps

**Environment.**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

ESM3 feature extraction requires HuggingFace access and a token:

```bash
export HF_TOKEN=<your_huggingface_token>
```

**Compute requirements.**

- **ESM3 feature extraction** (`extract_esm3_hidden_layer.py`): requires a CUDA-capable GPU. VRAM scales with protein length; BAHD/UGT sequences (~400–600 residues) ran on a single consumer GPU.
- **Downstream classifier training, screening, and interpretability**: CPU-only, runs on a laptop in minutes to a few hours depending on the search budget.

**Workflows.**

Extract ESM3 last-hidden-layer features from PDB structures:

```bash
python code/modeling_pipeline/extract_esm3_hidden_layer.py \
  --input_dir path/to/pdb_structures \
  --output_dir code/modeling_pipeline/BAHD_lastLayer_embeddings \
  --num_steps 10 \
  --device cuda
```

Screen baseline classifiers on mean-pooled ESM3 features:

```bash
python code/modeling_pipeline/classifier_screen.py \
  --feature_type last_layer \
  --dataset BAHD \
  --task both \
  --cv_mode kfold \
  --n_splits 5
```

Train the final logistic-regression baseline on mean-pooled ESM3 last-layer embeddings:

```bash
python code/modeling_pipeline/train_lr_embedding_baseline.py \
  --dataset BAHD \
  --cv_mode kfold \
  --n_splits 5 \
  --random_seed 45 \
  --C 0.025787
```

Run the full interpretability pipeline:

```bash
python code/interpretability_pipeline/run_full_analysis.py \
  --root_dir code/interpretability_pipeline \
  --metrics_dir results/modeling_pipeline/lr_embedding_baseline/lr_ll_final_5fold_run00_20260509_200740 \
  --predictions_dir results/modeling_pipeline/lr_embedding_baseline/lr_ll_final_5fold_run00_20260509_200740
```

Saved tensor caches (`BAHD_dataset/labels.pt`, `BAHD_lastLayer_embeddings/*_hidden_layer_steps10.pt`, etc.) are generated from the dataset tables, PDB structures, and ESM3 inference. Final saved metrics and figures are included so the repo can be reviewed without rerunning ESM3.

## 6. Results / Insights

Final BAHD 5-fold cross-validation results from the last-hidden-layer logistic-regression model:

| Task | Samples | Labels | Micro-AUPR | Macro-AUPR | Micro-AUROC | Macro-AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BAHD acceptor superclass | 366 | 23 | 0.736 | 0.608 | 0.924 | 0.915 |
| BAHD donor type         | 366 |  2 | 0.964 | 0.964 | 0.962 | 0.963 |

**Comparison to the source paper.** ESM3's published function track predicts broad InterPro annotations (e.g. *Chloramphenicol acetyltransferase-like domain superfamily*) for BAHDs and UGTs — it does not separate them at the substrate-class level. Our transfer-learning approach recovers meaningful, more granular substrate predictions from the same model's hidden states, particularly for donor type (near-ceiling AUROC) and with modest cross-family generalizability on shared UGT acceptor classes (Fig. 5).

**Interpretability finding.** Direct probing of unpooled ESM3 hidden states surfaced localized residue regions with substrate-specific signal that the mean-pooled, classifier-level explanations did not expose. Several single feature-by-residue-bin activations achieved high AUROC for specific acceptor classes (e.g. phenolic acids, naphthalenes), and these signals mapped to consistent spatial regions across folded structures (Fig. 6). Protein-length distributions did not explain the separation, reducing the likelihood of length confounding.

**Saved outputs of interest.**

- Final BAHD CV metrics: `results/modeling_pipeline/lr_embedding_baseline/lr_ll_final_5fold_run00_20260509_200740/`
- Final full-data BAHD model artifacts: `results/modeling_pipeline/lr_embedding_baseline/lr_ll_final_full_run00_20260509_201316/`
- UGT comparison figures: `results/modeling_pipeline/figures/`
- Interpretability figures: `results/interpretability_pipeline/analysis_outputs/writeup_figures/`
- Discriminative feature-position tables: `results/interpretability_pipeline/analysis_outputs/discriminative_features_full/`

## 7. Conclusion

- ESM3 can be repurposed for a more granular biochemical task than the original objective of its function track. Although ESM3 assigns overly broad functional labels to BAHDs, its hidden states contain substrate-specific information that simple linear classifiers can recover.
- The dominant limitation was dataset size and imbalance: several labels had only a few positive examples, making 5-fold CV estimates unstable and constraining model complexity (MLPs, transformer heads, and full fine-tuning all overfit relative to logistic regression).
- Mean pooling is a useful default for classifier input but obscures localized biological signal; bypassing it during interpretability probing recovered residue-region-level explanations.
- Future directions: expand the dataset across more enzyme families; use structure- or alignment-aware residue binning; evaluate position-aware architectures (attention pooling, regional probes, adapter-based fine-tuning); map high-signal residue positions onto known catalytic motifs to test biochemical plausibility.

## 8. References

- Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., Verkuil, R., Tran, V. Q., Deaton, J., Wiggert, M., Badkundri, R., Shafkat, I., Gong, J., Derry, A., Molina, R. S., Thomas, N., Khan, Y. A., Mishra, C., Kim, C., … Rives, A. (2025). Simulating 500 million years of evolution with a language model. *Science*, 387(6736), 850–858. https://doi.org/10.1126/science.ads0018
- Kim, H. W., Wang, M., Leber, C. A., Nothias, L.-F., Reher, R., Kang, K. B., van der Hooft, J. J. J., Dorrestein, P. C., Gerwick, W. H., & Cottrell, G. W. (2021). NPClassifier: A deep neural network-based structural classification tool for natural products. *Journal of Natural Products*, 84(11), 2795–2807. https://doi.org/10.1021/acs.jnatprod.1c00399
- Kruse, L. H., Weigle, A. T., Irfan, M., Martínez-Gómez, J., Chobirko, J. D., Schaffer, J. E., Bennett, A. A., Specht, C. D., Jez, J. M., Shukla, D., & Moghe, G. D. (2022). Orthology-based analysis helps map evolutionary diversification and predict substrate class use of BAHD acyltransferases. *The Plant Journal*, 111(5), 1453–1468. https://doi.org/10.1111/tpj.15902
- Pandala, S. R. (2019). LazyPredict (Version 0.3.0). https://github.com/shankarpandala/lazypredict
- Smith, N., Yuan, X., Melissinos, C., & Moghe, G. D. (2025). FuncFetch: An LLM-assisted workflow enables mining thousands of enzyme–substrate interactions from published manuscripts. *Bioinformatics*, 41(1), btae756. https://doi.org/10.1093/bioinformatics/btae756

## 9. Acknowledgements

This project was completed as the final assignment for **CS 4782: Introduction to Deep Learning** at Cornell University, Spring 2026. We thank the course instructors and teaching staff for their feedback throughout the semester. We also thank the **Moghe Lab** (Cornell SIPS) for biochemical context, dataset curation through the FuncFetch pipeline, and access to BAHD and UGT enzyme metadata. ESM3 model access was provided through the EvolutionaryScale / HuggingFace public release of Hayes et al. (2025).

## AI Disclosure

Generative AI was used in this assignment as a tool for analysis, coding, syntax, grammar, debugging, and for paraphrasing and reworking writing.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Glossary

- **Enzyme** — A protein that catalyzes a specific chemical reaction.
- **Substrate** — The input molecule(s) an enzyme acts on. BAHD reactions involve an acceptor (modified molecule) and a donor (group-supplying molecule).
- **BAHD acyltransferases** — A plant enzyme family that transfers acyl groups from donor molecules to diverse acceptor scaffolds. Named after the initials of the first four characterized members.
- **UGT glycosyltransferases** — A family that attaches sugar groups to acceptor molecules. Used here as a held-out cross-family test set.
- **CoA (Coenzyme A)** — A carrier molecule that activates acyl groups for transfer; the BAHD donor is typically a CoA thioester.
- **SMILES** — A text representation of molecular structure, used to encode substrate identity before clustering into categorical labels.
