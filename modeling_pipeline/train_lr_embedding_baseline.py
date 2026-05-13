#!/usr/bin/env python3
"""
Simple logistic-regression baseline for BAHD and UGT datasets using ESM3
last-layer embeddings.

This is intentionally structured to be comparable to train_lr_baseline.py.
The only major feature change is:

    old function-logit workflow:
        [1, residues, steps, channels] -> mean over residues -> flattened vector

    new last-layer workflow:
        [residues, hidden_dim] -> mean over residues -> [hidden_dim]

Supported tasks:
  - BAHD donor type
  - UGT donor compound
  - BAHD/UGT acceptor superclass

Labels are loaded from the same labels.pt files used by the function-logit
workflow:
  - BAHD_dataset/labels.pt
  - UGT_dataset/labels.pt

Embedding files are expected by default at the code_and_data level:
  - BAHD_lastLayer_embeddings/*_hidden_layer_steps10.pt
  - UGT_lastLayer_embeddings/*_hidden_layer_steps10.pt

Important: this script does NOT use sklearn.multiclass.OneVsRestClassifier.
It keeps the same manual one-label-at-a-time loop used by train_lr_baseline.py,
including the constant-model fallback for single-class training folds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
import re
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


DEFAULT_LOGISTIC_PARAMS: Dict[str, Any] = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "liblinear",
    "tol": 1e-06,
    "class_weight": "balanced",
    "max_iter": 5000,
    "fit_intercept": True,
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_GLOB = "*_hidden_layer_steps10.pt"

# If a future labels.pt explicitly stores embedding paths, these keys are checked first.
EMBEDDING_PATH_CACHE_KEYS: Tuple[str, ...] = (
    "hidden_layer_paths",
    "last_layer_paths",
    "last_layer_embedding_paths",
    "embedding_paths",
    "feature_paths",
)

# Old function-logit path lists are not direct paths to the new files, but their
# filename stems are useful ID hints for matching rows to embedding files.
ID_HINT_CACHE_KEYS: Tuple[str, ...] = (
    "function_logit_paths",
)


# -----------------------------------------------------------------------------
# CLI and validation
# -----------------------------------------------------------------------------

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic-regression baseline from mean-pooled ESM3 last-layer embeddings."
    )
    parser.add_argument(
        "--dataset",
        choices=("BAHD", "UGT", "both"),
        default="both",
        help="Which dataset family to run.",
    )
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
        default=str(SCRIPT_DIR / "results" / "lr_embedding_baseline"),
        help="Directory for metrics, predictions, and optional models.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional run folder name under output_dir. Defaults to timestamp.",
    )
    parser.add_argument(
        "--cv_mode",
        choices=("kfold", "loocv"),
        default="kfold",
        help="Cross-validation mode.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds when --cv_mode=kfold.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Tensor preprocessing device: auto, cpu, cuda, or cuda:N. Sklearn training remains CPU-based.",
    )
    parser.add_argument(
        "--min_positive_count",
        type=int,
        default=2,
        help="Drop labels with fewer than this many positives before cross-validation.",
    )
    parser.add_argument(
        "--write_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sequence cross-validated predictions as TSV.",
    )
    parser.add_argument(
        "--pickle_models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit models on the full dataset after cross-validation and pickle them.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed used for k-fold splitting.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_LOGISTIC_PARAMS["C"],
        help="Inverse regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--penalty",
        choices=("l1", "l2"),
        default=DEFAULT_LOGISTIC_PARAMS["penalty"],
        help="Regularization penalty for LogisticRegression.",
    )
    return parser.parse_args()


def families_from_arg(dataset_arg: str) -> List[str]:
    value = str(dataset_arg).upper()
    if value == "BOTH":
        return ["BAHD", "UGT"]
    return [value]


# -----------------------------------------------------------------------------
# Last-layer embedding loading and row matching
# -----------------------------------------------------------------------------

def _clean_id(value: str | Path) -> str:
    """Normalize path-like IDs enough for matching file names to labels.pt rows."""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith(">"):
        text = text[1:]
    text = text.split()[0]
    return text


def strip_embedding_suffix(value: str | Path) -> str:
    """
    Convert an embedding file name or ID to the protein ID before the hidden-layer suffix.

    Example:
        Aagr_HCT6_QPI70575.1_hidden_layer_steps10.pt -> Aagr_HCT6_QPI70575.1
    """
    text = Path(_clean_id(value)).name
    if text.endswith(".pt"):
        text = text[:-3]
    return re.sub(r"_hidden_layer_steps\d+$", "", text)


def strip_known_feature_suffix(value: str | Path) -> str:
    """Strip old/new feature suffixes from a path-like cache entry."""
    text = Path(_clean_id(value)).name
    for suffix in (".pt", ".npy"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = re.sub(r"_hidden_layer_steps\d+$", "", text)
    text = re.sub(r"_function_logits_(sequence|structure)_steps\d+$", "", text)
    text = re.sub(r"_residue_annotation_logits_(sequence|structure)_steps\d+$", "", text)
    text = re.sub(r"_esm3_embedding_structure_steps\d+$", "", text)
    text = re.sub(r"_esmc_embedding_sequence_steps\d+$", "", text)
    return text


def split_species_prefix(value: str | Path) -> Tuple[str, str]:
    """
    Split a cleaned ID at the first underscore.

    Examples:
        Aann_UGT25_PWA63700.1 -> ("Aann", "UGT25_PWA63700.1")
        Aann_AaUGT25_PWA63700.1 -> ("Aann", "AaUGT25_PWA63700.1")
    """
    text = strip_known_feature_suffix(value)
    if "_" not in text:
        return "", text
    species, rest = text.split("_", 1)
    return species, rest


def remove_terminal_underscore_token(value: str | Path) -> str:
    """
    Remove the final underscore-delimited token from an ID.

    This is useful for UGT labels where the enzyme_id includes a terminal
    accession that is not repeated in the embedding filename prefix.

    Examples:
        Aann_AaUGT25_PWA63700.1 -> Aann_AaUGT25
        Abel_AbUGT1_QGI57841.1 -> Abel_AbUGT1
    """
    text = strip_known_feature_suffix(value)
    parts = text.split("_")
    if len(parts) <= 1:
        return text
    return "_".join(parts[:-1])


def add_ugt_style_candidate_keys(keys: List[str], fasta_id: str, enzyme_id: str) -> None:
    """
    Add candidate keys for UGT last-layer embedding filenames.

    Observed UGT embedding filenames can combine the enzyme short name and
    fasta ID while avoiding duplicate accession tokens, for example:

        labels.pt fasta_id:  Abel_UGT1_QGI57841.1
        labels.pt enzyme_id: Abel_AbUGT1_QGI57841.1
        embedding stem:      Abel_AbUGT1_UGT1_QGI57841.1

    and:

        labels.pt fasta_id:  Aann_UGT25_PWA63700.1
        labels.pt enzyme_id: Aann_AaUGT25_PWA63700.1
        embedding stem:      Aann_AaUGT25_UGT25_PWA63700.1
    """
    fasta_clean = strip_known_feature_suffix(fasta_id)
    enzyme_clean = strip_known_feature_suffix(enzyme_id)
    if not fasta_clean or not enzyme_clean:
        return

    _fasta_species, fasta_without_species = split_species_prefix(fasta_clean)
    enzyme_base = remove_terminal_underscore_token(enzyme_clean)

    # Primary observed UGT pattern:
    #   <species>_<enzyme_short_name>_<fasta_without_species>
    if enzyme_base and fasta_without_species:
        _add_candidate_key(keys, f"{enzyme_base}_{fasta_without_species}")
        _add_candidate_key(keys, f"{enzyme_base}_{Path(fasta_without_species).stem}")

    # Defensive variant rebuilt from species + enzyme short name.
    enzyme_species, enzyme_without_species = split_species_prefix(enzyme_clean)
    enzyme_short = remove_terminal_underscore_token(enzyme_without_species)
    if enzyme_species and enzyme_short and fasta_without_species:
        _add_candidate_key(keys, f"{enzyme_species}_{enzyme_short}_{fasta_without_species}")
        _add_candidate_key(keys, f"{enzyme_species}_{enzyme_short}_{Path(fasta_without_species).stem}")


def _add_candidate_key(keys: List[str], value: str | Path | None) -> None:
    if value is None:
        return
    raw = _clean_id(value)
    if not raw:
        return
    variants = [
        raw,
        Path(raw).name,
        Path(raw).stem,
        strip_known_feature_suffix(raw),
        strip_embedding_suffix(raw),
    ]
    for item in variants:
        item = _clean_id(item)
        if item:
            keys.append(item)


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

    Keys include the full filename, filename stem, and stem before the
    _hidden_layer_stepsN suffix. Values are lists so duplicate key collisions
    are detected instead of silently choosing the wrong file.
    """
    index: Dict[str, List[Path]] = {}
    for path in sorted(embedding_dir.glob(embedding_glob)):
        keys = {
            _clean_id(path.name),
            _clean_id(path.stem),
            strip_embedding_suffix(path),
        }
        for key in keys:
            if key:
                index.setdefault(key, []).append(path)
    return index


def resolve_cache_path(raw_path: str, dataset_dir: Path, embedding_dir: Path) -> Path | None:
    """Resolve an explicit path stored in labels.pt, if it points to an existing file."""
    if not raw_path:
        return None
    path = Path(str(raw_path))
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
    id_hint: str | None,
) -> Path:
    """Find the last-layer embedding file corresponding to one labels.pt row."""
    # First honor an explicit embedding path if labels.pt provides one.
    if optional_path:
        explicit = resolve_cache_path(optional_path, dataset_dir, embedding_dir)
        if explicit is not None:
            return explicit

    candidate_keys: List[str] = []

    # Most direct fields.
    for raw in (fasta_id, enzyme_id, optional_path or "", id_hint or ""):
        _add_candidate_key(candidate_keys, raw)

    # Generic combined forms. Include both orders just in case a family uses
    # a combined embedding filename.
    fasta_clean = strip_known_feature_suffix(fasta_id)
    enzyme_clean = strip_known_feature_suffix(enzyme_id)
    if enzyme_clean and fasta_clean:
        _add_candidate_key(candidate_keys, f"{enzyme_clean}_{fasta_clean}")
        _add_candidate_key(candidate_keys, f"{fasta_clean}_{enzyme_clean}")

    # UGT-specific observed form, e.g.:
    #   fasta_id  = Aann_UGT25_PWA63700.1
    #   enzyme_id = Aann_AaUGT25_PWA63700.1
    #   embedding = Aann_AaUGT25_UGT25_PWA63700.1_hidden_layer_steps10.pt
    add_ugt_style_candidate_keys(candidate_keys, fasta_id, enzyme_id)

    # Preserve order while removing duplicates.
    deduped_keys = list(dict.fromkeys(key for key in candidate_keys if key))

    for key in deduped_keys:
        matches = embedding_index.get(key, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous embedding match for sample index {sample_idx}, key={key!r}: "
                + ", ".join(str(path) for path in matches[:10])
            )

    raise FileNotFoundError(
        "Could not find last-layer embedding for "
        f"sample_idx={sample_idx}, fasta_id={fasta_id!r}, enzyme_id={enzyme_id!r}. "
        f"Tried keys: {deduped_keys[:20]}"
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


def pool_embedding_mean(embedding: np.ndarray) -> np.ndarray:
    """Mean-pool [residues, hidden_dim] -> [hidden_dim]."""
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D embedding, got shape {embedding.shape}")
    return embedding.mean(axis=0).reshape(-1).astype(np.float32, copy=False)


def build_embedding_feature_matrix(
    *,
    dataset_dir: str | Path,
    embedding_dir: str | Path,
    cache: Mapping[str, Any],
    embedding_glob: str,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, List[str], List[List[int]], str | None]:
    """
    Build fixed-length mean-pooled features in the same row order as labels.pt.

    Returns:
        X: [num_samples, hidden_dim]
        embedding_paths: resolved embedding file path for each sample
        embedding_shapes: original [residues, hidden_dim] shape for each sample
        cache_match_key: labels.pt key used as a path or ID hint, if any
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

    vectors: List[np.ndarray] = []
    resolved_paths: List[str] = []
    original_shapes: List[List[int]] = []
    hidden_dim: int | None = None

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
        if hidden_dim is None:
            hidden_dim = int(embedding.shape[1])
        elif int(embedding.shape[1]) != hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch for {path}: expected {hidden_dim}, got {embedding.shape[1]}"
            )
        vectors.append(pool_embedding_mean(embedding))
        resolved_paths.append(str(path))
        original_shapes.append([int(dim) for dim in embedding.shape])

    if not vectors:
        raise ValueError(f"No last-layer embeddings loaded from {embedding_dir}")

    X = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    return X, resolved_paths, original_shapes, path_cache_key or id_hint_cache_key


# -----------------------------------------------------------------------------
# Logistic regression and metrics. Kept intentionally similar to train_lr_baseline.py.
# -----------------------------------------------------------------------------

def make_pipeline(params: dict | None = None) -> Pipeline:
    p = {**DEFAULT_LOGISTIC_PARAMS, **(params or {})}
    clf = LogisticRegression(
        C=p["C"],
        penalty=p["penalty"],
        solver=p["solver"],
        tol=p["tol"],
        class_weight=p["class_weight"],
        max_iter=p["max_iter"],
        fit_intercept=p["fit_intercept"],
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


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


def fit_binary_model(X: np.ndarray, y: np.ndarray,
                     params: dict | None = None) -> Dict[str, Any]:
    """Fit one binary model for one label, with the same constant fallback as the old script."""
    classes = np.unique(y)
    if classes.size < 2:
        return {
            "mode": "constant",
            "constant": int(classes[0]) if classes.size == 1 else 0,
            "pipeline": None,
        }

    pipeline = make_pipeline(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        pipeline.fit(X, y)
    return {
        "mode": "trained",
        "constant": None,
        "pipeline": pipeline,
    }


def score_binary_model(model: Dict[str, Any], X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if model["mode"] == "constant":
        constant = int(model["constant"])
        pred = np.full((X_test.shape[0],), constant, dtype=np.uint8)
        score = np.full((X_test.shape[0],), float(constant), dtype=np.float32)
        return pred, score

    pipeline = model["pipeline"]
    pred = pipeline.predict(X_test).astype(np.uint8, copy=False)
    score = pipeline.decision_function(X_test).astype(np.float32, copy=False)
    return pred, score


def run_cv_multilabel(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cv_mode: str,
    logistic_params: dict | None = None,
) -> Dict[str, Any]:
    """Manual one-label-at-a-time CV loop, matching train_lr_baseline.py."""
    n_samples, n_labels = y.shape
    pred_matrix = np.zeros((n_samples, n_labels), dtype=np.uint8)
    score_matrix = np.zeros((n_samples, n_labels), dtype=np.float32)

    per_fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train = X[train_idx]
        X_test  = X[test_idx]
        y_train = y[train_idx]
        fold_score_buffer = np.zeros((len(test_idx), n_labels), dtype=np.float32)

        for label_idx in range(n_labels):
            model = fit_binary_model(X_train, y_train[:, label_idx], logistic_params)
            pred, score = score_binary_model(model, X_test)
            pred_matrix[test_idx, label_idx]  = pred
            score_matrix[test_idx, label_idx] = score
            fold_score_buffer[:, label_idx]   = score

        if cv_mode != "loocv":
            y_fold = y[test_idx]
            fold_aupr_vals: List[float] = []
            fold_auroc_vals: List[float] = []
            for li in range(n_labels):
                a = safe_average_precision(y_fold[:, li], fold_score_buffer[:, li])
                r = safe_roc_auc(y_fold[:, li], fold_score_buffer[:, li])
                if a is not None: fold_aupr_vals.append(a)
                if r is not None: fold_auroc_vals.append(r)
            per_fold_metrics.append({
                "fold":        fold_idx,
                "n_test":      int(len(test_idx)),
                "micro_aupr":  safe_average_precision(y_fold.reshape(-1), fold_score_buffer.reshape(-1)),
                "micro_auroc": safe_roc_auc(y_fold.reshape(-1), fold_score_buffer.reshape(-1)),
                "macro_aupr":  float(np.mean(fold_aupr_vals))  if fold_aupr_vals  else None,
                "macro_auroc": float(np.mean(fold_auroc_vals)) if fold_auroc_vals else None,
            })

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

    micro_aupr = safe_average_precision(y.reshape(-1), score_matrix.reshape(-1))
    micro_auroc = safe_roc_auc(y.reshape(-1), score_matrix.reshape(-1))

    return {
        "pred_matrix": pred_matrix,
        "score_matrix": score_matrix,
        "metrics": {
            "num_samples":      int(n_samples),
            "num_labels":       int(n_labels),
            "cv_mode":          str(cv_mode),
            "num_folds":        int(len(cv_splits)),
            "micro_aupr":       micro_aupr,
            "micro_auroc":      micro_auroc,
            "macro_aupr":       float(np.mean(aupr_values))  if aupr_values  else None,
            "macro_auroc":      float(np.mean(auroc_values)) if auroc_values else None,
            "per_label":        per_label_metrics,
            "per_fold_metrics": per_fold_metrics,
        },
    }


def fit_full_models(X: np.ndarray, y: np.ndarray, label_names: Sequence[str],
                    logistic_params: dict | None = None) -> Dict[str, Any]:
    p = {**DEFAULT_LOGISTIC_PARAMS, **(logistic_params or {})}
    models: Dict[str, Any] = {}
    for label_idx, label_name in enumerate(label_names):
        models[str(label_name)] = fit_binary_model(X, y[:, label_idx], p)
    return {
        "label_names": list(label_names),
        "feature_type": "esm3_last_layer_embedding_mean_pool",
        "pooling": "mean",
        "logistic_params": p,
        "models": models,
    }


# -----------------------------------------------------------------------------
# Output writers
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


# -----------------------------------------------------------------------------
# Family/task runner
# -----------------------------------------------------------------------------

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
    random_seed: int,
    min_positive_count: int,
    write_predictions: bool,
    pickle_models: bool,
    logistic_params: dict | None = None,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    embedding_dir = Path(embedding_dir)
    cache = load_dataset_cache(dataset_dir)
    resolved_device = get_device(device)

    X, embedding_paths, embedding_shapes, cache_match_key = build_embedding_feature_matrix(
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

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature/label row mismatch for family={family} task={task_name}: "
            f"X has {X.shape[0]} rows, y has {y.shape[0]} rows"
        )

    cv_splits = build_cv_splits(
        num_samples=X.shape[0],
        cv_mode=cv_mode,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    effective_params = {**DEFAULT_LOGISTIC_PARAMS, **(logistic_params or {})}
    result = run_cv_multilabel(X, y, label_names, cv_splits=cv_splits, cv_mode=cv_mode,
                               logistic_params=effective_params)

    metrics_payload = {
        "family": family,
        "task": task_name,
        "dataset_dir": str(dataset_dir),
        "embedding_dir": str(embedding_dir),
        "embedding_glob": str(embedding_glob),
        "embedding_cache_match_key": cache_match_key,
        "device": str(resolved_device),
        "cv_mode": cv_mode,
        "n_splits": int(n_splits) if cv_mode == "kfold" else None,
        "random_seed": int(random_seed),
        "feature_type": "esm3_last_layer_embedding_mean_pool",
        "pooling": "mean",
        "input_shape_example": embedding_shapes[0] if embedding_shapes else None,
        "pooled_feature_dim": int(X.shape[1]),
        "min_positive_count": int(min_positive_count),
        "logistic_params": effective_params,
        "kept_labels": list(label_names),
        "kept_label_positive_counts": kept_counts,
        "metrics": result["metrics"],
    }

    stem = f"{family.lower()}_{task_name}"
    write_json(run_dir / f"{stem}_metrics.json", metrics_payload)

    fasta_ids = get_cache_sequence(cache, "fasta_ids", X.shape[0])
    enzyme_ids = get_cache_sequence(cache, "enzyme_ids", X.shape[0])

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

    if pickle_models:
        model_bundle = fit_full_models(X, y, label_names, effective_params)
        model_bundle["embedding_paths"] = embedding_paths
        model_bundle["embedding_shapes"] = embedding_shapes
        model_path = run_dir / f"{stem}_models.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model_bundle, handle)

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

    logistic_params = {**DEFAULT_LOGISTIC_PARAMS, "C": args.C, "penalty": args.penalty}

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
                    random_seed=args.random_seed,
                    min_positive_count=args.min_positive_count,
                    write_predictions=args.write_predictions,
                    pickle_models=args.pickle_models,
                    logistic_params=logistic_params,
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
            "feature_type": "esm3_last_layer_embedding_mean_pool",
            "pooling": "mean",
            "manual_multilabel_loop": True,
            "write_predictions": args.write_predictions,
            "pickle_models": args.pickle_models,
            "logistic_params": logistic_params,
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
                        f"{item['family'].lower()}_{item['task']}_models.pkl"
                        if args.pickle_models
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
