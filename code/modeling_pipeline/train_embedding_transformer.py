#!/usr/bin/env python3
"""
Transformer classifier for BAHD and UGT substrate-label prediction from ESM3
last-layer embeddings.

This is the last-layer embedding analogue of soft_func_logit_decoder.py.

Input feature for each protein:
  last-layer embedding tensor: [residues, hidden_dim]

The model keeps the full variable-length residue sequence and applies:
  [residues, hidden_dim]
    -> linear projection to d_model
    -> sinusoidal positional encoding
    -> TransformerEncoder
    -> mean or attention pooling over residues
    -> multilabel output head

Supported tasks:
  - BAHD donor type
  - UGT donor compound
  - BAHD/UGT acceptor superclass

Label source is unchanged from the function-logit workflow:
  BAHD_dataset/labels.pt
  UGT_dataset/labels.pt

Embedding files are expected to live outside the dataset folders by default:
  BAHD_lastLayer_embeddings/*_hidden_layer_steps10.pt
  UGT_lastLayer_embeddings/*_hidden_layer_steps10.pt

The script matches embedding files to label rows primarily by cache["fasta_ids"].
It also supports optional path lists in labels.pt, such as "hidden_layer_paths",
"last_layer_embedding_paths", "embedding_paths", or "feature_paths".
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import (
    build_cv_splits,
    dataset_dir_for_family,
    filter_labels_by_support,
    get_device,
    get_task_definition,
    label_indices_to_names,
    load_dataset_cache,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_GLOB = "*_hidden_layer_steps10.pt"
FEATURE_TYPE = "esm3_last_layer_embedding_transformer"

EMBEDDING_PATH_CACHE_KEYS: Tuple[str, ...] = (
    "hidden_layer_paths",
    "last_layer_paths",
    "last_layer_embedding_paths",
    "embedding_paths",
    "feature_paths",
)

# These are never treated as direct embedding paths, but their file stems can
# help map old function-logit cache rows to new last-layer embedding files.
ID_HINT_CACHE_KEYS: Tuple[str, ...] = (
    "function_logit_paths",
)


# -----------------------------------------------------------------------------
# CLI and validation
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a transformer classifier using ESM3 last-layer embeddings."
    )
    parser.add_argument("--dataset", choices=("BAHD", "UGT", "both"), default="both")
    parser.add_argument(
        "--bahd_dataset_dir",
        default=str(dataset_dir_for_family("BAHD")),
        help="Path to compact BAHD dataset directory containing labels.pt.",
    )
    parser.add_argument(
        "--ugt_dataset_dir",
        default=str(dataset_dir_for_family("UGT")),
        help="Path to compact UGT dataset directory containing labels.pt.",
    )
    parser.add_argument(
        "--bahd_embedding_dir",
        default=str(SCRIPT_DIR / "BAHD_lastLayer_embeddings"),
        help="Directory containing BAHD last-layer embedding .pt files.",
    )
    parser.add_argument(
        "--ugt_embedding_dir",
        default=str(SCRIPT_DIR / "UGT_lastLayer_embeddings"),
        help="Directory containing UGT last-layer embedding .pt files.",
    )
    parser.add_argument(
        "--embedding_glob",
        default=DEFAULT_EMBEDDING_GLOB,
        help="Glob pattern used to discover embedding files in each embedding directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "results" / "embedding_transformer"),
        help="Directory for metrics, predictions, and optional models.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--cv_mode", choices=("kfold", "loocv"), default="kfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min_positive_count", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=("mean", "attention"), default="attention")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--write_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_models", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def families_from_arg(dataset_arg: str) -> List[str]:
    value = str(dataset_arg).upper()
    if value == "BOTH":
        return ["BAHD", "UGT"]
    return [value]


def validate_dataset_dir(path: Path, family: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{family} dataset directory not found: {path}")
    labels_path = path / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"{family} labels cache not found: {labels_path}")


def validate_embedding_dir(path: Path, family: str, embedding_glob: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{family} embedding directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{family} embedding path is not a directory: {path}")
    if not any(path.glob(embedding_glob)):
        raise FileNotFoundError(
            f"No {family} embedding files matching {embedding_glob!r} found under {path}"
        )


# -----------------------------------------------------------------------------
# Last-layer embedding loading and matching
# -----------------------------------------------------------------------------

def strip_embedding_suffix(value: str | Path) -> str:
    """
    Convert a file name or ID to the protein ID used before the hidden-layer suffix.

    Examples:
        Aagr_HCT6_QPI70575.1_hidden_layer_steps10.pt -> Aagr_HCT6_QPI70575.1
        Aagr_HCT6_QPI70575.1_hidden_layer_steps10    -> Aagr_HCT6_QPI70575.1
    """
    text = Path(str(value)).name
    if text.endswith(".pt"):
        text = text[:-3]
    return re.sub(r"_hidden_layer_steps\d+$", "", text)


def strip_known_feature_suffix(value: str | Path) -> str:
    """Strip known old/new feature suffixes from a path-like cache entry."""
    text = Path(str(value)).name
    for suffix in (".pt", ".npy"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = re.sub(r"_hidden_layer_steps\d+$", "", text)
    text = re.sub(r"_function_logits_(sequence|structure)_steps\d+$", "", text)
    text = re.sub(r"_residue_annotation_logits_(sequence|structure)_steps\d+$", "", text)
    text = re.sub(r"_esm3_embedding_structure_steps\d+$", "", text)
    text = re.sub(r"_esmc_embedding_sequence_steps\d+$", "", text)
    return text


def get_cache_sequence(cache: Mapping[str, Any], key: str, expected_len: int) -> List[str]:
    value = cache.get(key)
    if value is None:
        return [""] * expected_len
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected cache[{key!r}] to be a sequence, found {type(value)}")
    if len(value) != expected_len:
        raise ValueError(
            f"Expected cache[{key!r}] to have length {expected_len}, found {len(value)}"
        )
    return [str(item) for item in value]


def get_optional_path_list(cache: Mapping[str, Any], expected_len: int) -> Tuple[str | None, List[str] | None]:
    for key in EMBEDDING_PATH_CACHE_KEYS:
        value = cache.get(key)
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"Expected cache[{key!r}] to be a sequence, found {type(value)}")
        if len(value) != expected_len:
            raise ValueError(
                f"Expected cache[{key!r}] to have length {expected_len}, found {len(value)}"
            )
        return key, [str(item) for item in value]
    return None, None


def get_optional_id_hint_list(cache: Mapping[str, Any], expected_len: int) -> Tuple[str | None, List[str] | None]:
    for key in ID_HINT_CACHE_KEYS:
        value = cache.get(key)
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"Expected cache[{key!r}] to be a sequence, found {type(value)}")
        if len(value) != expected_len:
            raise ValueError(
                f"Expected cache[{key!r}] to have length {expected_len}, found {len(value)}"
            )
        return key, [str(item) for item in value]
    return None, None


def build_embedding_index(embedding_dir: Path, embedding_glob: str) -> Dict[str, List[Path]]:
    """
    Build a permissive lookup table for embedding files.

    Keys include:
      - filename stem without _hidden_layer_stepsN
      - full filename stem
      - full filename
    Values are lists so duplicate key collisions can be detected later.
    """
    index: Dict[str, List[Path]] = {}
    for path in sorted(embedding_dir.glob(embedding_glob)):
        keys = {
            strip_embedding_suffix(path),
            path.stem,
            path.name,
        }
        for key in keys:
            if key:
                index.setdefault(key, []).append(path)
    return index


def resolve_cache_path(raw_path: str, dataset_dir: Path, embedding_dir: Path) -> Path | None:
    """Resolve a path stored in labels.pt, if that path points to an existing file."""
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            embedding_dir / path,
            dataset_dir / path,
            dataset_dir.parent / path,
            Path.cwd() / path,
        ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def resolve_embedding_path_for_sample(
    *,
    sample_idx: int,
    dataset_dir: Path,
    embedding_dir: Path,
    embedding_index: Mapping[str, List[Path]],
    fasta_id: str,
    enzyme_id: str,
    optional_path: str | None,
    id_hint: str | None = None,
) -> Path:
    """Find the last-layer embedding file corresponding to one label row."""
    # First honor an explicit path if labels.pt provides one.
    if optional_path:
        explicit = resolve_cache_path(optional_path, dataset_dir, embedding_dir)
        if explicit is not None:
            return explicit

    candidate_keys: List[str] = []
    base_candidates = [fasta_id, enzyme_id, optional_path or "", id_hint or ""]

    # Common filename convention in this project:
    #   <enzyme_id>_<fasta_id>_hidden_layer_steps10.pt
    # Example:
    #   Aagr_HCT6_QPI70575.1_hidden_layer_steps10.pt
    # where enzyme_id may be Aagr_HCT6 and fasta_id may be QPI70575.1.
    if fasta_id and enzyme_id:
        base_candidates.extend(
            [
                f"{enzyme_id}_{fasta_id}",
                f"{fasta_id}_{enzyme_id}",
            ]
        )

    for raw in base_candidates:
        if not raw:
            continue
        raw_str = str(raw).strip()
        if not raw_str:
            continue
        candidate_keys.extend(
            [
                raw_str,
                raw_str.split()[0],
                Path(raw_str).name,
                Path(raw_str).stem,
                strip_known_feature_suffix(raw_str),
                strip_embedding_suffix(raw_str),
            ]
        )

    # Preserve order while removing duplicates.
    deduped_keys = list(dict.fromkeys(key for key in candidate_keys if key))
    for key in deduped_keys:
        matches = embedding_index.get(key, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous embedding match for sample index {sample_idx}, key={key!r}: "
                + ", ".join(str(path) for path in matches[:5])
            )

    raise FileNotFoundError(
        "Could not find last-layer embedding for "
        f"sample_idx={sample_idx}, fasta_id={fasta_id!r}, enzyme_id={enzyme_id!r}. "
        f"Tried keys: {deduped_keys[:12]}"
    )


def load_last_layer_embedding(path: str | Path, device: str | torch.device = "cpu") -> np.ndarray:
    """Load one [residues, hidden_dim] last-layer embedding tensor."""
    torch_device = get_device(str(device))
    tensor = torch.load(path, map_location=torch_device)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, found {type(tensor)}")

    tensor = tensor.detach().float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected last-layer embedding with shape [residues, hidden_dim], got {tuple(tensor.shape)} in {path}"
        )
    if tensor.shape[0] < 1:
        raise ValueError(f"Embedding has zero residues: {path}")
    if tensor.shape[1] < 1:
        raise ValueError(f"Embedding has zero hidden dimensions: {path}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Embedding contains NaN or Inf values: {path}")

    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def build_embedding_sequence_list(
    *,
    dataset_dir: str | Path,
    embedding_dir: str | Path,
    cache: Mapping[str, Any],
    embedding_glob: str,
    device: str | torch.device = "cpu",
) -> Tuple[List[np.ndarray], List[str], List[List[int]], str | None]:
    """
    Build variable-length embedding sequences in the same row order as labels.pt.

    Returns:
        sequences: list of arrays with shape [residues, hidden_dim]
        embedding_paths: resolved embedding file path for each sample
        embedding_shapes: original tensor shape for each sample
        path_cache_key: optional labels.pt key used for direct path resolution or ID hints
    """
    dataset_dir = Path(dataset_dir)
    embedding_dir = Path(embedding_dir)

    label_tensor = cache.get("labels")
    if not isinstance(label_tensor, torch.Tensor):
        raise TypeError("labels.pt is missing tensor 'labels'")
    num_samples = int(label_tensor.shape[0])

    fasta_ids = get_cache_sequence(cache, "fasta_ids", num_samples)
    enzyme_ids = get_cache_sequence(cache, "enzyme_ids", num_samples)
    path_cache_key, optional_paths = get_optional_path_list(cache, num_samples)
    id_hint_cache_key, id_hints = get_optional_id_hint_list(cache, num_samples)

    embedding_index = build_embedding_index(embedding_dir, embedding_glob)
    if not embedding_index:
        raise FileNotFoundError(
            f"No embedding files matching {embedding_glob!r} found under {embedding_dir}"
        )

    sequences: List[np.ndarray] = []
    resolved_paths: List[str] = []
    original_shapes: List[List[int]] = []

    for idx in range(num_samples):
        path = resolve_embedding_path_for_sample(
            sample_idx=idx,
            dataset_dir=dataset_dir,
            embedding_dir=embedding_dir,
            embedding_index=embedding_index,
            fasta_id=fasta_ids[idx],
            enzyme_id=enzyme_ids[idx],
            optional_path=optional_paths[idx] if optional_paths is not None else None,
            id_hint=id_hints[idx] if id_hints is not None else None,
        )
        embedding = load_last_layer_embedding(path, device=device)
        sequences.append(embedding.astype(np.float32, copy=False))
        resolved_paths.append(str(path))
        original_shapes.append([int(dim) for dim in embedding.shape])

    if not sequences:
        raise ValueError(f"No last-layer embeddings loaded from {embedding_dir}")

    hidden_dims = {int(seq.shape[1]) for seq in sequences}
    if len(hidden_dims) != 1:
        raise ValueError(f"Inconsistent hidden dimensions across embeddings: {sorted(hidden_dims)}")

    return sequences, resolved_paths, original_shapes, path_cache_key or id_hint_cache_key


# -----------------------------------------------------------------------------
# Dataset, model, and training helpers
# -----------------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[np.ndarray], labels: np.ndarray):
        self.sequences = list(sequences)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.sequences[idx], self.labels[idx]


def collate_embeddings(batch: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    max_len = max(seq.shape[0] for seq in sequences)
    feat_dim = sequences[0].shape[1]
    x = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    for idx, seq in enumerate(sequences):
        length = seq.shape[0]
        if seq.shape[1] != feat_dim:
            raise ValueError(f"Inconsistent feature dimension in batch: {seq.shape[1]} vs {feat_dim}")
        x[idx, :length] = seq
        mask[idx, :length] = True
    y = np.stack(labels, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(y)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds positional encoding max length {self.pe.shape[1]}"
            )
        return x + self.pe[:, : x.shape[1]]


class EmbeddingTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        pooling: str,
        num_labels: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.position = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        if pooling == "attention":
            self.attention = nn.Linear(d_model, 1)
        else:
            self.attention = None
        self.output = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.input_norm(hidden)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)

        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "mean":
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        else:
            if self.attention is None:
                raise RuntimeError("Attention pooling requested but attention layer is missing.")
            scores = self.attention(hidden).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (hidden * weights * mask_f).sum(dim=1)
        return self.output(pooled)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.sum() == 0:
        return None
    score = float(average_precision_score(y_true, y_score))
    return score if np.isfinite(score) else None


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    score = float(roc_auc_score(y_true, y_score))
    return score if np.isfinite(score) else None


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor | None:
    positives = y_train.sum(axis=0).astype(np.float32)
    negatives = float(len(y_train)) - positives
    valid = positives > 0
    if not np.any(valid):
        return None
    weights = np.ones_like(positives, dtype=np.float32)
    weights[valid] = negatives[valid] / positives[valid]
    return torch.from_numpy(weights)


def train_model(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    set_random_seed(seed)
    dataset = SequenceDataset(sequences, labels)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_embeddings,
    )

    model = EmbeddingTransformerClassifier(
        input_dim=sequences[0].shape[1],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        pooling=pooling,
        num_labels=labels.shape[1],
    ).to(device)
    pos_weight = compute_pos_weight(labels)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_mask, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}


def score_model(
    trained: Dict[str, Any],
    sequences: Sequence[np.ndarray],
    device: torch.device,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    pooling: str,
    num_labels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = SequenceDataset(sequences, np.zeros((len(sequences), num_labels), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_embeddings)
    model = EmbeddingTransformerClassifier(
        input_dim=sequences[0].shape[1],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        pooling=pooling,
        num_labels=num_labels,
    ).to(device)
    model.load_state_dict(trained["model_state"])
    model.eval()
    with torch.no_grad():
        batch_x, batch_mask, _ = next(iter(loader))
        logits = model(batch_x.to(device), batch_mask.to(device)).detach().cpu().numpy().astype(np.float32, copy=False)
    preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(np.uint8, copy=False)
    return preds, logits


def run_cv_multilabel(
    sequences: Sequence[np.ndarray],
    y: np.ndarray,
    label_names: Sequence[str],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cv_mode: str,
    device: torch.device,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    n_samples, n_labels = y.shape
    pred_matrix = np.zeros((n_samples, n_labels), dtype=np.uint8)
    score_matrix = np.zeros((n_samples, n_labels), dtype=np.float32)
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        trained = train_model(
            sequences=[sequences[i] for i in train_idx],
            labels=y[train_idx],
            device=device,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_idx,
        )
        preds, scores = score_model(
            trained=trained,
            sequences=[sequences[i] for i in test_idx],
            device=device,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            num_labels=n_labels,
        )
        pred_matrix[test_idx] = preds
        score_matrix[test_idx] = scores

    per_label_metrics: List[Dict[str, Any]] = []
    aupr_values: List[float] = []
    auroc_values: List[float] = []
    positive_counts = y.sum(axis=0).astype(int)
    for label_idx, label_name in enumerate(label_names):
        y_true = y[:, label_idx]
        y_score = score_matrix[:, label_idx]
        label_aupr = safe_average_precision(y_true, y_score)
        label_auroc = safe_roc_auc(y_true, y_score)
        if label_aupr is not None:
            aupr_values.append(label_aupr)
        if label_auroc is not None:
            auroc_values.append(label_auroc)
        per_label_metrics.append(
            {
                "label": str(label_name),
                "positive_count": int(positive_counts[label_idx]),
                "aupr": label_aupr,
                "auroc": label_auroc,
            }
        )

    return {
        "pred_matrix": pred_matrix,
        "score_matrix": score_matrix,
        "metrics": {
            "num_samples": int(n_samples),
            "num_labels": int(n_labels),
            "cv_mode": str(cv_mode),
            "num_folds": int(len(cv_splits)),
            "micro_aupr": safe_average_precision(y.reshape(-1), score_matrix.reshape(-1)),
            "micro_auroc": safe_roc_auc(y.reshape(-1), score_matrix.reshape(-1)),
            "macro_aupr": float(np.mean(aupr_values)) if aupr_values else None,
            "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
            "per_label": per_label_metrics,
        },
    }


def fit_full_model(
    sequences: Sequence[np.ndarray],
    y: np.ndarray,
    device: torch.device,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    trained = train_model(
        sequences=sequences,
        labels=y,
        device=device,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        pooling=pooling,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    trained["model_type"] = "embedding_transformer"
    trained["feature_type"] = FEATURE_TYPE
    trained["model_params"] = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "pooling": pooling,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "random_seed": seed,
    }
    return trained


# -----------------------------------------------------------------------------
# Output writers and family/task runner
# -----------------------------------------------------------------------------

def write_predictions_tsv(
    path: str | Path,
    fasta_ids: Sequence[str],
    enzyme_ids: Sequence[str],
    embedding_paths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        header = [
            "fasta_id",
            "enzyme_id",
            "embedding_path",
            "true_labels",
            "predicted_labels",
            "true_vector",
            "pred_vector",
            "score_vector",
        ]
        handle.write("\t".join(header) + "\n")
        for idx, fasta_id in enumerate(fasta_ids):
            true_labels = ";".join(label_indices_to_names(y_true[idx], label_names))
            pred_labels = ";".join(label_indices_to_names(y_pred[idx], label_names))
            true_vector = ",".join(str(int(v)) for v in y_true[idx].tolist())
            pred_vector = ",".join(str(int(v)) for v in y_pred[idx].tolist())
            score_vector = ",".join(f"{float(v):.6f}" for v in y_score[idx].tolist())
            row = [
                str(fasta_id),
                str(enzyme_ids[idx]),
                str(embedding_paths[idx]),
                true_labels,
                pred_labels,
                true_vector,
                pred_vector,
                score_vector,
            ]
            handle.write("\t".join(row) + "\n")


def run_family_task(
    family: str,
    task_name: str,
    dataset_dir: str | Path,
    embedding_dir: str | Path,
    embedding_glob: str,
    run_dir: Path,
    device: str,
    cv_mode: str,
    n_splits: int,
    min_positive_count: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    random_seed: int,
    write_predictions: bool,
    save_models: bool,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    embedding_dir = Path(embedding_dir)
    cache = load_dataset_cache(dataset_dir)
    resolved_device = get_device(device)

    sequences, embedding_paths, embedding_shapes, path_cache_key = build_embedding_sequence_list(
        dataset_dir=dataset_dir,
        embedding_dir=embedding_dir,
        cache=cache,
        embedding_glob=embedding_glob,
        device=resolved_device,
    )

    y_raw, label_names_raw = get_task_definition(cache, task_name)
    y, label_names, kept_counts = filter_labels_by_support(
        y_raw,
        label_names_raw,
        min_positive_count=min_positive_count,
    )
    if y.shape[1] == 0:
        raise ValueError(
            f"No labels remain for family={family} task={task_name} after min_positive_count={min_positive_count}"
        )

    if len(sequences) != y.shape[0]:
        raise ValueError(
            f"Feature/label row mismatch for family={family} task={task_name}: "
            f"loaded {len(sequences)} sequences, y has {y.shape[0]} rows"
        )

    cv_splits = build_cv_splits(
        num_samples=len(sequences),
        cv_mode=cv_mode,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    result = run_cv_multilabel(
        sequences=sequences,
        y=y,
        label_names=label_names,
        cv_splits=cv_splits,
        cv_mode=cv_mode,
        device=resolved_device,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        pooling=pooling,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        seed=random_seed,
    )

    hidden_dims = [int(shape[1]) for shape in embedding_shapes if len(shape) == 2]
    residue_counts = [int(shape[0]) for shape in embedding_shapes if len(shape) == 2]

    metrics_payload = {
        "family": family,
        "task": task_name,
        "dataset_dir": str(dataset_dir),
        "embedding_dir": str(embedding_dir),
        "embedding_glob": str(embedding_glob),
        "embedding_path_cache_key": path_cache_key,
        "device": str(resolved_device),
        "cv_mode": cv_mode,
        "n_splits": int(n_splits) if cv_mode == "kfold" else None,
        "random_seed": int(random_seed),
        "feature_type": FEATURE_TYPE,
        "pooling": pooling,
        "input_shape_example": embedding_shapes[0] if embedding_shapes else None,
        "hidden_feature_dim": int(hidden_dims[0]) if hidden_dims else None,
        "min_residues": int(min(residue_counts)) if residue_counts else None,
        "max_residues": int(max(residue_counts)) if residue_counts else None,
        "mean_residues": float(np.mean(residue_counts)) if residue_counts else None,
        "min_positive_count": int(min_positive_count),
        "transformer_params": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "pooling": pooling,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "random_seed": random_seed,
        },
        "kept_labels": list(label_names),
        "kept_label_positive_counts": kept_counts,
        "metrics": result["metrics"],
    }

    stem = f"{family.lower()}_{task_name}"
    write_json(run_dir / f"{stem}_metrics.json", metrics_payload)

    fasta_ids = get_cache_sequence(cache, "fasta_ids", len(sequences))
    enzyme_ids = get_cache_sequence(cache, "enzyme_ids", len(sequences))

    if write_predictions:
        write_predictions_tsv(
            run_dir / f"{stem}_predictions.tsv",
            fasta_ids,
            enzyme_ids,
            embedding_paths,
            y,
            result["pred_matrix"],
            result["score_matrix"],
            label_names,
        )

    if save_models:
        full_model = fit_full_model(
            sequences=sequences,
            y=y,
            device=resolved_device,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            seed=random_seed,
        )
        full_model["label_names"] = list(label_names)
        full_model["embedding_paths"] = embedding_paths
        full_model["embedding_shapes"] = embedding_shapes
        full_model["embedding_glob"] = str(embedding_glob)
        torch.save(full_model, run_dir / f"{stem}_models.pt")

    return metrics_payload


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = {
        "BAHD": Path(args.bahd_dataset_dir),
        "UGT": Path(args.ugt_dataset_dir),
    }
    embedding_dirs = {
        "BAHD": Path(args.bahd_embedding_dir),
        "UGT": Path(args.ugt_embedding_dir),
    }

    selected_families = families_from_arg(args.dataset)
    for family in selected_families:
        validate_dataset_dir(dataset_dirs[family], family)
        validate_embedding_dir(embedding_dirs[family], family, args.embedding_glob)

    all_results: List[Dict[str, Any]] = []
    for family in selected_families:
        for task_name in ("donor", "acceptor"):
            all_results.append(
                run_family_task(
                    family=family,
                    task_name=task_name,
                    dataset_dir=dataset_dirs[family],
                    embedding_dir=embedding_dirs[family],
                    embedding_glob=args.embedding_glob,
                    run_dir=run_dir,
                    device=args.device,
                    cv_mode=args.cv_mode,
                    n_splits=args.n_splits,
                    min_positive_count=args.min_positive_count,
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    pooling=args.pooling,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    random_seed=args.random_seed,
                    write_predictions=args.write_predictions,
                    save_models=args.save_models,
                )
            )

    write_json(
        run_dir / "run_config.json",
        {
            "dataset": args.dataset,
            "bahd_dataset_dir": args.bahd_dataset_dir,
            "ugt_dataset_dir": args.ugt_dataset_dir,
            "bahd_embedding_dir": args.bahd_embedding_dir,
            "ugt_embedding_dir": args.ugt_embedding_dir,
            "embedding_glob": args.embedding_glob,
            "output_dir": str(Path(args.output_dir)),
            "run_name": run_name,
            "device": str(get_device(args.device)),
            "cv_mode": args.cv_mode,
            "n_splits": args.n_splits if args.cv_mode == "kfold" else None,
            "random_seed": args.random_seed,
            "min_positive_count": args.min_positive_count,
            "feature_type": FEATURE_TYPE,
            "pooling": args.pooling,
            "transformer_params": {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "pooling": args.pooling,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "random_seed": args.random_seed,
            },
            "write_predictions": args.write_predictions,
            "save_models": args.save_models,
            "results_written": [
                {
                    "family": item["family"],
                    "task": item["task"],
                    "metrics_file": f"{item['family'].lower()}_{item['task']}_metrics.json",
                    "predictions_file": (
                        f"{item['family'].lower()}_{item['task']}_predictions.tsv"
                        if args.write_predictions
                        else None
                    ),
                    "model_file": (
                        f"{item['family'].lower()}_{item['task']}_models.pt"
                        if args.save_models
                        else None
                    ),
                }
                for item in all_results
            ],
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR running {Path(__file__).name}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
