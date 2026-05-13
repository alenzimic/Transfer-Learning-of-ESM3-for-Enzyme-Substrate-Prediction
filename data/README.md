# Data

This directory contains the curated datasets used for training and evaluation.

- `BAHD_dataset.xlsx`: curated BAHD acyltransferase enzyme-substrate metadata used for training and cross-validation.
- `UGT_dataset.tsv`: curated UGT glycosyltransferase enzyme-substrate metadata used as a held-out cross-family test set.

Some scripts also expect preprocessed ESM3 tensor caches (`labels.pt`, function logits, and last-layer embeddings). Those caches are generated from these source tables, PDB structures, and ESM3 inference. They are described in the root `README.md`.
