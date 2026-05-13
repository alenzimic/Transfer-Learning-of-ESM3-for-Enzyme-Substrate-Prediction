#!/usr/bin/env python3
"""
Shared utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, LeaveOneOut

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BAHD_DATASET_DIR = SCRIPT_DIR / "BAHD_dataset"
DEFAULT_UGT_DATASET_DIR = SCRIPT_DIR / "UGT_dataset"


def get_device(device: str | None = None) -> torch.device:
    if device is None or str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def build_cv_splits(
    num_samples: int,
    cv_mode: str = "kfold",
    n_splits: int = 5,
    random_seed: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    mode = str(cv_mode).strip().lower()
    if num_samples < 2:
        raise ValueError("At least 2 samples are required for cross-validation.")

    if mode == "loocv":
        splitter = LeaveOneOut()
        dummy = np.empty((num_samples, 1), dtype=np.uint8)
        return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(dummy)]

    if mode == "kfold":
        if n_splits < 2:
            raise ValueError("--n_splits must be >= 2 for kfold.")
        if n_splits > num_samples:
            raise ValueError(
                f"--n_splits ({n_splits}) cannot exceed the number of samples ({num_samples})."
            )
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
        dummy = np.empty((num_samples, 1), dtype=np.uint8)
        return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(dummy)]

    raise ValueError(f"Unsupported cv_mode '{cv_mode}'. Expected 'kfold' or 'loocv'.")


def load_dataset_cache(dataset_dir: str | Path) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    cache_path = dataset_dir / "labels.pt"
    cache = torch.load(cache_path, map_location="cpu")
    if not isinstance(cache, dict):
        raise TypeError(f"Expected dict payload in {cache_path}, found {type(cache)}")
    return cache


def dataset_dir_for_family(family: str) -> Path:
    family_key = str(family).strip().upper()
    if family_key == "BAHD":
        return DEFAULT_BAHD_DATASET_DIR
    if family_key == "UGT":
        return DEFAULT_UGT_DATASET_DIR
    raise ValueError(f"Unsupported family '{family}'. Expected BAHD or UGT.")


def load_function_logits_vector(logits_path: str | Path, device: str | torch.device = "cpu") -> np.ndarray:
    sequence_logits = load_function_logits_sequence(logits_path, device=device)
    pooled = sequence_logits.mean(axis=0)
    return pooled.reshape(-1).astype(np.float32, copy=False)


def load_function_logits_sequence(
    logits_path: str | Path,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    torch_device = get_device(str(device))
    tensor = torch.load(logits_path, map_location=torch_device)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {logits_path}, found {type(tensor)}")

    arr = tensor.detach().float()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected logits with shape [residues, steps, channels] after squeeze, got {tuple(arr.shape)}"
        )

    return arr.detach().cpu().numpy().astype(np.float32, copy=False)


def build_feature_matrix(
    dataset_dir: str | Path,
    cache: Dict[str, Any],
    device: str | torch.device = "cpu",
) -> np.ndarray:
    dataset_dir = Path(dataset_dir)
    rel_paths = cache.get("function_logit_paths")
    if not isinstance(rel_paths, Sequence):
        raise TypeError("labels.pt is missing 'function_logit_paths'")

    vectors: List[np.ndarray] = []
    for rel_path in rel_paths:
        abs_path = dataset_dir / str(rel_path)
        vectors.append(load_function_logits_vector(abs_path, device=device))

    if not vectors:
        raise ValueError(f"No function logit vectors found under {dataset_dir}")
    return np.stack(vectors, axis=0)


def build_sequence_feature_list(
    dataset_dir: str | Path,
    cache: Dict[str, Any],
    device: str | torch.device = "cpu",
    flatten: bool = False,
) -> List[np.ndarray]:
    dataset_dir = Path(dataset_dir)
    rel_paths = cache.get("function_logit_paths")
    if not isinstance(rel_paths, Sequence):
        raise TypeError("labels.pt is missing 'function_logit_paths'")

    sequences: List[np.ndarray] = []
    for rel_path in rel_paths:
        abs_path = dataset_dir / str(rel_path)
        seq = load_function_logits_sequence(abs_path, device=device)
        if flatten:
            seq = seq.reshape(seq.shape[0], -1)
        sequences.append(seq.astype(np.float32, copy=False))

    if not sequences:
        raise ValueError(f"No function logit sequences found under {dataset_dir}")
    return sequences


def get_task_definition(cache: Dict[str, Any], task_name: str) -> Tuple[np.ndarray, List[str]]:
    label_names = [str(name) for name in cache.get("label_names", [])]
    label_tensor = cache.get("labels")
    if not isinstance(label_tensor, torch.Tensor):
        raise TypeError("labels.pt is missing tensor 'labels'")

    labels_np = label_tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
    task_key = str(task_name).strip().lower()

    if task_key == "donor":
        family = str(cache.get("family", "")).upper()
        prefix = "donor_type::" if family == "BAHD" else "donor_compound::"
    elif task_key == "acceptor":
        prefix = "acceptor_superclass::"
    else:
        raise ValueError(f"Unsupported task '{task_name}'. Expected donor or acceptor.")

    selected = [idx for idx, name in enumerate(label_names) if name.startswith(prefix)]
    if not selected:
        raise ValueError(f"No labels found for task '{task_name}' with prefix '{prefix}'")

    task_labels = [label_names[idx] for idx in selected]
    task_matrix = labels_np[:, selected]
    return task_matrix, task_labels


def filter_labels_by_support(
    y: np.ndarray,
    label_names: Sequence[str],
    min_positive_count: int,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D label matrix, got shape {y.shape}")

    positive_counts = y.sum(axis=0).astype(int)
    keep_mask = positive_counts >= int(min_positive_count)
    kept_names = [str(name) for name, keep in zip(label_names, keep_mask) if keep]
    kept_counts = {
        str(name): int(count)
        for name, count, keep in zip(label_names, positive_counts.tolist(), keep_mask)
        if keep
    }
    return y[:, keep_mask], kept_names, kept_counts


def label_indices_to_names(row: np.ndarray, label_names: Sequence[str]) -> List[str]:
    row_arr = np.asarray(row).astype(bool, copy=False)
    return [str(name) for name, value in zip(label_names, row_arr) if bool(value)]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
