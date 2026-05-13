#!/usr/bin/env python3
"""
linear_interpretability.py

Exact per-residue and per-feature attribution for ridge/logistic models trained
on mean-pooled ESM3 features. Works for both:

  * "logits_ridge_*" pickles  -> RidgeClassifier on mean-pooled function logits (D=2080)
  * "hidden_layer_lr_*" pickles -> LogisticRegression on mean-pooled last hidden states (D=1536)

The math is the same in both cases because both models are linear and mean
pooling is linear, so:

    score = sum_i w_i * (x_i - mu_i) / sigma_i + b
          = (1/T) sum_t sum_i w_i / sigma_i * (h_{t,i} - mu_i) + b

The bracketed inner sum is residue t's signed contribution to the score; they
sum to (score - bias) by construction.

Outputs per protein per task:
    <out_dir>/<run_label>/per_protein/<fasta_id>.pt
plus a sanity-check TSV at:
    <out_dir>/<run_label>/sanity_check.tsv

Usage (BAHD, all four pickles):
    python linear_interpretability.py \
        --root_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability \
        --family BAHD

Or override individual paths; see --help for all flags.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Family configuration
# --------------------------------------------------------------------------------------

@dataclass
class FeatureSourceConfig:
    """Where to find per-residue tensors and how to name them."""
    name: str                        # "last_hidden" or "function_logits"
    feature_type_label: str          # human-readable feature type
    folder_name: str                 # subfolder under root_dir (or override path)
    layout: str                      # "flat" or "per_protein_subfolder"
    filename_template: str           # e.g. "{fasta_id}_hidden_layer_steps10.pt"
    expected_dim: Optional[int]      # warn if mismatch (None = no check)


@dataclass
class PickleConfig:
    """One pickle to process."""
    pickle_filename: str             # under model_pkl_files/
    task: str                        # "donor" or "acceptor"
    model_kind: str                  # "logistic" or "ridge"
    feature_source: str              # key into FAMILY_CONFIGS[family]["feature_sources"]
    run_label: str                   # output subfolder name


# Family configs. Add new families by adding entries here.
FAMILY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "BAHD": {
        "labels_subpath": "BAHD_dataset/labels.pt",
        "structures_subpath": "BAHD_dataset/BAHD_structures",
        "feature_sources": {
            "last_hidden": FeatureSourceConfig(
                name="last_hidden",
                feature_type_label="esm3_last_layer_embedding",
                folder_name="BAHD_lastLayer_embeddings",
                layout="flat",
                filename_template="{fasta_id}_hidden_layer_steps10.pt",
                expected_dim=1536,
            ),
            "function_logits": FeatureSourceConfig(
                name="function_logits",
                feature_type_label="function_logits_structure_steps10",
                folder_name="BAHD_dataset/BAHD_structures",
                layout="per_protein_subfolder",
                filename_template="{fasta_id}_function_logits_structure_steps10.pt",
                expected_dim=2080,
            ),
        },
        "pickles": [
            PickleConfig(
                pickle_filename="hidden_layer_lr_bahd_donor_models.pkl",
                task="donor", model_kind="logistic",
                feature_source="last_hidden", run_label="bahd_donor_lasthidden",
            ),
            PickleConfig(
                pickle_filename="hidden_layer_lr_bahd_acceptor_models.pkl",
                task="acceptor", model_kind="logistic",
                feature_source="last_hidden", run_label="bahd_acceptor_lasthidden",
            ),
            PickleConfig(
                pickle_filename="logits_ridge_bahd_donor_models.pkl",
                task="donor", model_kind="ridge",
                feature_source="function_logits", run_label="bahd_donor_funclogit",
            ),
            PickleConfig(
                pickle_filename="logits_ridge_bahd_acceptor_models.pkl",
                task="acceptor", model_kind="ridge",
                feature_source="function_logits", run_label="bahd_acceptor_funclogit",
            ),
        ],
    },
}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-residue and per-feature attribution for linear protein-function models."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Root folder. The default subfolders under it are "
                             "model_pkl_files/, BAHD_dataset/, BAHD_lastLayer_embeddings/, etc.")
    parser.add_argument("--family", default="BAHD", choices=list(FAMILY_CONFIGS.keys()),
                        help="Which protein family to process. New families can be added "
                             "to FAMILY_CONFIGS in this script.")
    parser.add_argument("--pickle_dir", default=None,
                        help="Folder containing pickle files. Default: <root_dir>/model_pkl_files")
    parser.add_argument("--labels_path", default=None,
                        help="Path to labels.pt. Default: family-specific path under root_dir.")
    parser.add_argument("--last_hidden_dir", default=None,
                        help="Override last-hidden-layer folder. Default: family-specific.")
    parser.add_argument("--function_logits_dir", default=None,
                        help="Override function-logit folder. Default: family-specific.")
    parser.add_argument("--structures_dir", default=None,
                        help="Folder of per-protein PDBs, used to detect ESM3 BOS/EOS padding "
                             "for last-hidden features by comparing PDB residue count to tensor "
                             "rows. Default: family-specific (same as function_logits_dir for BAHD).")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write interpretability outputs. "
                             "Default: <root_dir>/interpretability_outputs")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip per-protein files that already exist.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N proteins (for quick testing).")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Helpers: cache + per-residue tensor loading
# --------------------------------------------------------------------------------------

def load_labels_cache(labels_path: Path) -> Dict[str, Any]:
    cache = torch.load(labels_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise TypeError(f"Expected dict in {labels_path}, got {type(cache)}")
    return cache


def get_cache_seq(cache: Mapping[str, Any], key: str, expected_len: int) -> List[str]:
    value = cache.get(key)
    if value is None:
        return [""] * expected_len
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if len(value) != expected_len:
        raise ValueError(f"cache[{key!r}] length {len(value)} != expected {expected_len}")
    return [str(item) for item in value]


def resolve_per_residue_tensor_path(
    fasta_id: str,
    enzyme_id: str,
    feature_source: FeatureSourceConfig,
    base_folder: Path,
) -> Optional[Path]:
    """Find the per-residue tensor for one protein given its IDs."""
    # Build candidate filenames using both fasta_id and enzyme_id (UGT-style mismatch hedge).
    candidates_ids = [fasta_id, enzyme_id]
    candidates_ids = [c for c in candidates_ids if c]
    candidates_ids = list(dict.fromkeys(candidates_ids))  # dedup, preserve order

    if feature_source.layout == "flat":
        # Single folder, one file per protein.
        for cid in candidates_ids:
            cand = base_folder / feature_source.filename_template.format(fasta_id=cid)
            if cand.is_file():
                return cand
        # Last-resort: scan with a permissive glob using the fasta_id stem.
        if candidates_ids:
            stem = candidates_ids[0]
            stripped = re.sub(r"\.pt$", "", feature_source.filename_template.format(fasta_id=stem))
            globbed = list(base_folder.glob(f"*{stem}*"))
            for path in globbed:
                if path.is_file() and path.suffix == ".pt":
                    return path

    elif feature_source.layout == "per_protein_subfolder":
        # base_folder/<fasta_id>/<fasta_id>_function_logits_*.pt or similar.
        for cid in candidates_ids:
            subfolder = base_folder / cid
            if not subfolder.is_dir():
                # Tolerate IDs that may have been stripped/altered.
                continue
            target = subfolder / feature_source.filename_template.format(fasta_id=cid)
            if target.is_file():
                return target
            # Also tolerate the file living in an `embeddings/` subfolder (FuncPred layout).
            inner = subfolder / "embeddings" / feature_source.filename_template.format(fasta_id=cid)
            if inner.is_file():
                return inner
            # Last-resort glob within the subfolder.
            for path in subfolder.rglob(f"*function_logits_structure_steps*.pt"):
                if path.is_file():
                    return path
    else:
        raise ValueError(f"Unknown layout {feature_source.layout!r}")
    return None


def load_per_residue_tensor(path: Path, expected_dim: Optional[int]) -> np.ndarray:
    """
    Load a per-residue tensor as float32 numpy of shape (T, D).

    Tolerates several common layouts:
      * (T, D)                     -> used as-is
      * (1, T, D)                  -> batch dim squeezed
      * (1, T, A, B) or (T, A, B)  -> trailing axes flattened to D = A*B
                                       (ESM3 function logits: A=8, B=260, D=2080)

    The flatten step is safe for the linearity decomposition because the
    training pipeline applied the same flatten before mean-pooling and
    standardization; standardization is per flattened-feature, and the math
    we use only requires that "model input == mean over residues of the
    per-residue feature vector", which still holds after flattening.
    """
    t = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{path}: expected torch.Tensor, got {type(t)}")
    t = t.detach().float()

    # Squeeze a leading batch dim of 1 if present.
    if t.ndim >= 3 and t.shape[0] == 1:
        t = t.squeeze(0)

    if t.ndim == 2:
        flat = t
    elif t.ndim >= 3:
        T = int(t.shape[0])
        trailing = 1
        for d in t.shape[1:]:
            trailing *= int(d)
        flat = t.reshape(T, trailing)
    else:
        raise ValueError(f"{path}: cannot interpret shape {tuple(t.shape)} as (T, D).")

    if expected_dim is not None and int(flat.shape[1]) != expected_dim:
        raise ValueError(
            f"{path}: feature dim {flat.shape[1]} != expected {expected_dim} "
            f"(raw shape was {tuple(t.shape)}). "
            "Did you point at the wrong feature folder for this pickle?"
        )
    if not torch.isfinite(flat).all():
        raise ValueError(f"{path}: contains NaN/Inf values.")
    return flat.numpy().astype(np.float32, copy=False)


def count_pdb_residues(pdb_path: Path) -> int:
    """Count unique (chain, resseq, icode) ATOM residues in the first model."""
    seen: set = set()
    in_first_model = True
    with pdb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                if seen:
                    in_first_model = False
                continue
            if line.startswith("ENDMDL"):
                in_first_model = False
                continue
            if not in_first_model:
                continue
            if line.startswith("ATOM"):
                seen.add((line[21:22], line[22:26], line[26:27]))
    return len(seen)


def find_matching_pdb(
    fasta_id: str, structures_root: Optional[Path],
) -> Optional[Path]:
    """Look for a PDB under <structures_root>/<fasta_id>/. Returns None if absent."""
    if structures_root is None:
        return None
    sub = structures_root / fasta_id
    if not sub.is_dir():
        return None
    pdbs = sorted(sub.rglob("*.pdb"))
    return pdbs[0] if pdbs else None


def strip_bos_eos_if_needed(
    H: np.ndarray, fasta_id: str, structures_root: Optional[Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect ESM3's BOS/EOS padding and strip the first and last rows when present.

    Heuristic: if a matching PDB exists under <structures_root>/<fasta_id>/ and
    the tensor has exactly PDB_residue_count + 2 rows, treat row 0 as BOS and
    row -1 as EOS. Otherwise return the tensor unchanged.

    Returns (H_real, info) where:
      H_real is shape (T_real, D)
      info contains:
        - 'bos_eos_stripped' (bool)
        - 'bos_row', 'eos_row' (rows that were sliced off; D-vectors) when stripped
        - 't_pdb', 't_tensor', 'pdb_path'
        - 'reason' (str describing detection outcome)
    """
    info: Dict[str, Any] = {
        "bos_eos_stripped": False,
        "t_tensor": int(H.shape[0]),
    }
    pdb_path = find_matching_pdb(fasta_id, structures_root)
    if pdb_path is None:
        info["reason"] = "no_matching_pdb"
        info["pdb_path"] = None
        return H, info

    info["pdb_path"] = str(pdb_path)
    try:
        t_pdb = count_pdb_residues(pdb_path)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"pdb_parse_error: {exc}"
        return H, info
    info["t_pdb"] = int(t_pdb)

    if H.shape[0] == t_pdb + 2:
        info["bos_eos_stripped"] = True
        info["bos_row"] = H[0].copy()
        info["eos_row"] = H[-1].copy()
        info["reason"] = "stripped_bos_eos"
        return H[1:-1], info
    elif H.shape[0] == t_pdb:
        info["reason"] = "already_aligned"
    else:
        info["reason"] = f"unexpected_offset_tensor_minus_pdb={H.shape[0] - t_pdb}"
    return H, info


# --------------------------------------------------------------------------------------
# Core attribution math
# --------------------------------------------------------------------------------------

def extract_pipeline_params(pipeline: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Return (mu, sigma, w, b) from a Pipeline(StandardScaler -> linear classifier).

    Handles both LogisticRegression (coef_ shape (1, D)) and RidgeClassifier
    (coef_ shape (D,)) for binary tasks.
    """
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    mu = np.asarray(scaler.mean_, dtype=np.float64)
    sigma = np.asarray(scaler.scale_, dtype=np.float64)

    coef = np.asarray(clf.coef_, dtype=np.float64)
    if coef.ndim == 2:
        if coef.shape[0] != 1:
            raise ValueError(f"Unexpected non-binary coef shape {coef.shape}")
        w = coef[0]
    elif coef.ndim == 1:
        w = coef
    else:
        raise ValueError(f"Unexpected coef ndim {coef.ndim}")

    intercept = clf.intercept_
    if hasattr(intercept, "__len__"):
        b = float(np.asarray(intercept, dtype=np.float64)[0])
    else:
        b = float(intercept)
    return mu, sigma, w, b


def compute_attributions_for_protein(
    H_full: np.ndarray,                    # (T_full, D) per-residue tensor INCLUDING any BOS/EOS rows
    H_real: np.ndarray,                    # (T_real, D) per-residue tensor with BOS/EOS rows removed (or same as H_full)
    bos_eos_stripped: bool,
    bundle_models: Mapping[str, Any],
    label_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    For each label, compute:
      - hand-computed score = w . x' + b on the SAME pooled vector the model was trained on
      - per-feature contributions  (D,)  : w_i * x'_i, signed
      - per-residue contributions  (T_real,) on the BOS/EOS-stripped axis,
        redistributing the contribution from any stripped boundary rows evenly so the
        per-residue contributions still sum exactly to (score - bias).
      - bias
      - constant-fallback flag
      - per-label boundary (BOS/EOS) contribution magnitudes, for diagnostic transparency

    The pooled input to the model is the mean of H_full (matching how training pooled),
    so the score is faithful to the deployed model. Residue contributions are computed
    on H_real (the real protein residues) and then have the boundary contribution
    redistributed across them to preserve the exact decomposition.
    """
    T_full, D = H_full.shape
    T_real = H_real.shape[0]
    n_labels = len(label_names)

    # Mean pool exactly the way the training pipeline did (over the full padded tensor).
    x_full = H_full.mean(axis=0).astype(np.float64)  # (D,)

    scores = np.zeros(n_labels, dtype=np.float64)
    biases = np.zeros(n_labels, dtype=np.float64)
    feat_contribs = np.zeros((n_labels, D), dtype=np.float64)
    res_contribs = np.zeros((n_labels, T_real), dtype=np.float64)
    predicted = np.zeros(n_labels, dtype=np.uint8)
    is_constant = np.zeros(n_labels, dtype=np.uint8)
    per_label_check_error = np.zeros(n_labels, dtype=np.float64)
    boundary_contrib = np.zeros(n_labels, dtype=np.float64)  # contribution from BOS+EOS rows, in standardized log-odds units

    for li, label in enumerate(label_names):
        entry = bundle_models[label]
        if entry["mode"] == "constant":
            constant = int(entry.get("constant") or 0)
            scores[li] = float(constant)
            biases[li] = float(constant)
            predicted[li] = constant
            is_constant[li] = 1
            continue

        pipeline = entry["pipeline"]
        mu, sigma, w, b = extract_pipeline_params(pipeline)
        if mu.shape[0] != D:
            raise ValueError(
                f"Label {label!r}: scaler dim {mu.shape[0]} != per-residue dim {D}. "
                "Wrong feature source for this pickle."
            )

        # Standardized pooled input — based on the full pooled vector the model saw.
        x_std = (x_full - mu) / sigma            # (D,)

        # Per-feature contributions (signed).
        contribs_feat = w * x_std                 # (D,)
        score_hand = float(contribs_feat.sum() + b)

        # Per-residue contributions over the *full* T_full axis.
        # contribution_t = (1/T_full) * sum_i (w_i / sigma_i) * (h_{t,i} - mu_i)
        per_res_dim_weight = (w / sigma).astype(np.float64)               # (D,)
        H_centered_full = H_full.astype(np.float64) - mu                  # (T_full, D)
        contribs_res_full = (H_centered_full @ per_res_dim_weight) / float(T_full)  # (T_full,)

        if bos_eos_stripped:
            # Slice off the first and last contributions (BOS and EOS), then redistribute
            # their sum evenly across the real residues to preserve the exact decomposition.
            bos_eos_sum = float(contribs_res_full[0] + contribs_res_full[-1])
            contribs_res_real = contribs_res_full[1:-1].copy()
            if T_real > 0:
                contribs_res_real += bos_eos_sum / float(T_real)
            boundary_contrib[li] = bos_eos_sum
        else:
            contribs_res_real = contribs_res_full
            boundary_contrib[li] = 0.0

        # Sanity: pipeline.decision_function on the pooled full vector.
        score_sklearn = float(pipeline.decision_function(x_full.reshape(1, -1).astype(np.float32))[0])
        per_label_check_error[li] = abs(score_hand - score_sklearn)

        scores[li] = score_hand
        biases[li] = b
        feat_contribs[li] = contribs_feat
        res_contribs[li] = contribs_res_real
        predicted[li] = 1 if score_hand > 0 else 0

    return {
        "scores": scores.astype(np.float32),
        "biases": biases.astype(np.float32),
        "feature_contribs": feat_contribs.astype(np.float32),
        "residue_contribs": res_contribs.astype(np.float32),
        "predicted_labels": predicted,
        "constant_label_mask": is_constant,
        "per_label_check_error": per_label_check_error.astype(np.float64),
        "boundary_contrib": boundary_contrib.astype(np.float32),
    }


def maybe_sigmoid(scores: np.ndarray, model_kind: str) -> Optional[np.ndarray]:
    """For logistic regression, scores are log-odds; sigmoid -> probability."""
    if model_kind != "logistic":
        return None
    # Numerically stable sigmoid.
    s = scores.astype(np.float64)
    out = np.where(s >= 0,
                   1.0 / (1.0 + np.exp(-s)),
                   np.exp(s) / (1.0 + np.exp(s)))
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Main per-pickle driver
# --------------------------------------------------------------------------------------

def process_pickle(
    *,
    family: str,
    pickle_cfg: PickleConfig,
    pickle_dir: Path,
    labels_path: Path,
    feature_folder: Path,
    feature_source: FeatureSourceConfig,
    structures_root: Optional[Path],
    output_dir: Path,
    skip_existing: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    pkl_path = pickle_dir / pickle_cfg.pickle_filename
    if not pkl_path.is_file():
        print(f"[WARN] Skipping missing pickle: {pkl_path}")
        return {"run_label": pickle_cfg.run_label, "status": "missing_pickle"}

    print(f"\n=== {pickle_cfg.run_label} ===")
    print(f"  pickle:        {pkl_path}")
    print(f"  feature dir:   {feature_folder}")
    print(f"  feature type:  {feature_source.feature_type_label}")
    print(f"  model kind:    {pickle_cfg.model_kind}")

    with pkl_path.open("rb") as h:
        bundle = pickle.load(h)
    label_names: List[str] = list(bundle["label_names"])
    bundle_models: Dict[str, Any] = bundle["models"]
    print(f"  num labels:    {len(label_names)}")

    cache = load_labels_cache(labels_path)
    label_tensor = cache["labels"]
    n_samples = int(label_tensor.shape[0])
    fasta_ids = get_cache_seq(cache, "fasta_ids", n_samples)
    enzyme_ids = get_cache_seq(cache, "enzyme_ids", n_samples)
    print(f"  num proteins:  {n_samples}")

    run_dir = output_dir / pickle_cfg.run_label
    per_protein_dir = run_dir / "per_protein"
    per_protein_dir.mkdir(parents=True, exist_ok=True)

    sanity_rows: List[Dict[str, Any]] = []
    n_done = 0
    n_skipped = 0
    n_missing = 0
    n_failed = 0
    n_bos_eos_stripped = 0
    n_already_aligned = 0
    n_no_pdb = 0
    n_unexpected_offset = 0

    proteins_to_process = list(zip(fasta_ids, enzyme_ids))
    if limit is not None:
        proteins_to_process = proteins_to_process[: int(limit)]

    for fasta_id, enzyme_id in proteins_to_process:
        out_path = per_protein_dir / f"{fasta_id}.pt"
        if skip_existing and out_path.is_file():
            n_skipped += 1
            continue

        tensor_path = resolve_per_residue_tensor_path(
            fasta_id=fasta_id,
            enzyme_id=enzyme_id,
            feature_source=feature_source,
            base_folder=feature_folder,
        )
        if tensor_path is None:
            print(f"  [MISS] {fasta_id}: no per-residue tensor found under {feature_folder}")
            n_missing += 1
            continue

        try:
            H_full = load_per_residue_tensor(tensor_path, expected_dim=feature_source.expected_dim)
        except Exception as exc:
            print(f"  [FAIL load] {fasta_id}: {exc}")
            n_failed += 1
            continue

        # Detect ESM3 BOS/EOS padding by comparing tensor row count to the matching PDB.
        # We only try this for last-hidden features (per-residue ESM3 trunk output);
        # function-logit tensors come from a different head and should already be
        # length T_protein, not T_protein + 2.
        H_real = H_full
        boundary_info: Dict[str, Any] = {"bos_eos_stripped": False, "reason": "not_attempted"}
        if feature_source.name == "last_hidden":
            H_real, boundary_info = strip_bos_eos_if_needed(H_full, fasta_id, structures_root)

        try:
            attrib = compute_attributions_for_protein(
                H_full=H_full,
                H_real=H_real,
                bos_eos_stripped=bool(boundary_info.get("bos_eos_stripped")),
                bundle_models=bundle_models,
                label_names=label_names,
            )
        except Exception as exc:
            print(f"  [FAIL compute] {fasta_id}: {exc}")
            traceback.print_exc()
            n_failed += 1
            continue

        # Probabilities only meaningful for logistic.
        probs = maybe_sigmoid(attrib["scores"], pickle_cfg.model_kind)

        # Save bundle for this protein.
        out_bundle = {
            "fasta_id": fasta_id,
            "enzyme_id": enzyme_id,
            "family": family,
            "task": pickle_cfg.task,
            "model_kind": pickle_cfg.model_kind,
            "feature_type": feature_source.feature_type_label,
            "embedding_path": str(tensor_path),
            "T": int(H_real.shape[0]),
            "T_full": int(H_full.shape[0]),
            "D": int(H_real.shape[1]),
            "bos_eos_stripped": bool(boundary_info.get("bos_eos_stripped")),
            "bos_eos_reason": str(boundary_info.get("reason", "")),
            "matched_pdb_path": boundary_info.get("pdb_path"),
            "label_names": label_names,
            "scores": torch.from_numpy(attrib["scores"]),
            "probabilities": (torch.from_numpy(probs) if probs is not None else None),
            "predicted_labels": torch.from_numpy(attrib["predicted_labels"]),
            "biases": torch.from_numpy(attrib["biases"]),
            "feature_contribs": torch.from_numpy(attrib["feature_contribs"]),
            "residue_contribs": torch.from_numpy(attrib["residue_contribs"]),
            "constant_label_mask": torch.from_numpy(attrib["constant_label_mask"]),
            "boundary_contrib_per_label": torch.from_numpy(attrib["boundary_contrib"]),
        }
        torch.save(out_bundle, out_path)

        # Tally BOS/EOS detection outcome for this protein.
        reason = str(boundary_info.get("reason", ""))
        if boundary_info.get("bos_eos_stripped"):
            n_bos_eos_stripped += 1
        elif reason == "already_aligned":
            n_already_aligned += 1
        elif reason == "no_matching_pdb":
            n_no_pdb += 1
        elif reason.startswith("unexpected_offset"):
            n_unexpected_offset += 1

        # Sanity rows.
        for li, label in enumerate(label_names):
            if attrib["constant_label_mask"][li]:
                continue
            res_sum = float(attrib["residue_contribs"][li].sum())
            implied_score = res_sum + float(attrib["biases"][li])
            sanity_rows.append({
                "fasta_id": fasta_id,
                "label": label,
                "score": float(attrib["scores"][li]),
                "score_minus_bias": float(attrib["scores"][li]) - float(attrib["biases"][li]),
                "residue_sum": res_sum,
                "residue_sum_plus_bias": implied_score,
                "abs_decomp_error": abs(float(attrib["scores"][li]) - implied_score),
                "abs_sklearn_check_error": float(attrib["per_label_check_error"][li]),
            })

        n_done += 1
        if n_done % 25 == 0:
            print(f"  ... {n_done} proteins processed")

    # Write sanity-check TSV.
    tsv_path = run_dir / "sanity_check.tsv"
    fieldnames = [
        "fasta_id", "label",
        "score", "score_minus_bias",
        "residue_sum", "residue_sum_plus_bias",
        "abs_decomp_error", "abs_sklearn_check_error",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(sanity_rows)

    # Summary.
    if sanity_rows:
        decomp_errors = np.array([r["abs_decomp_error"] for r in sanity_rows])
        sklearn_errors = np.array([r["abs_sklearn_check_error"] for r in sanity_rows])
        print(f"  done. processed={n_done}  skipped={n_skipped}  missing={n_missing}  failed={n_failed}")
        print(f"  BOS/EOS detection: stripped={n_bos_eos_stripped}  "
              f"already_aligned={n_already_aligned}  no_pdb={n_no_pdb}  "
              f"unexpected_offset={n_unexpected_offset}")
        print(f"  decomposition error (residue_sum + bias vs hand score):  "
              f"max={decomp_errors.max():.2e}  mean={decomp_errors.mean():.2e}")
        print(f"  sklearn check error (hand score vs decision_function):    "
              f"max={sklearn_errors.max():.2e}  mean={sklearn_errors.mean():.2e}")
    else:
        print(f"  done. processed={n_done}  skipped={n_skipped}  missing={n_missing}  failed={n_failed}")
        print("  no sanity rows recorded (all-constant pickle?)")

    return {
        "run_label": pickle_cfg.run_label,
        "status": "ok",
        "processed": n_done,
        "skipped": n_skipped,
        "missing": n_missing,
        "failed": n_failed,
        "sanity_check_tsv": str(tsv_path),
        "per_protein_dir": str(per_protein_dir),
    }


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root_dir not found: {root}")

    fam_cfg = FAMILY_CONFIGS[args.family]
    pickle_dir = Path(args.pickle_dir).resolve() if args.pickle_dir else (root / "model_pkl_files")
    labels_path = Path(args.labels_path).resolve() if args.labels_path else (root / fam_cfg["labels_subpath"])
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (root / "interpretability_outputs")
    structures_root = Path(args.structures_dir).resolve() if args.structures_dir else (
        root / fam_cfg.get("structures_subpath", "")
    )
    if not structures_root or not structures_root.is_dir():
        print(f"[WARN] structures_dir not found ({structures_root}); BOS/EOS detection will be skipped.")
        structures_root = None
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_overrides: Dict[str, Path] = {}
    if args.last_hidden_dir:
        feature_overrides["last_hidden"] = Path(args.last_hidden_dir).resolve()
    if args.function_logits_dir:
        feature_overrides["function_logits"] = Path(args.function_logits_dir).resolve()

    if not pickle_dir.is_dir():
        raise SystemExit(f"Pickle folder not found: {pickle_dir}")
    if not labels_path.is_file():
        raise SystemExit(f"labels.pt not found: {labels_path}")

    print(f"root_dir:      {root}")
    print(f"pickle_dir:    {pickle_dir}")
    print(f"labels_path:   {labels_path}")
    print(f"structures_dir:{structures_root if structures_root else '(skipped)'}")
    print(f"output_dir:    {output_dir}")
    print(f"family:        {args.family}")

    summaries: List[Dict[str, Any]] = []
    for pickle_cfg in fam_cfg["pickles"]:
        feature_source: FeatureSourceConfig = fam_cfg["feature_sources"][pickle_cfg.feature_source]
        feature_folder = feature_overrides.get(feature_source.name, root / feature_source.folder_name)
        if not feature_folder.is_dir():
            print(f"[WARN] Feature folder not found, skipping {pickle_cfg.run_label}: {feature_folder}")
            summaries.append({"run_label": pickle_cfg.run_label, "status": "missing_feature_folder"})
            continue
        summary = process_pickle(
            family=args.family,
            pickle_cfg=pickle_cfg,
            pickle_dir=pickle_dir,
            labels_path=labels_path,
            feature_folder=feature_folder,
            feature_source=feature_source,
            structures_root=structures_root,
            output_dir=output_dir,
            skip_existing=bool(args.skip_existing),
            limit=args.limit,
        )
        summaries.append(summary)

    # Final overview.
    print("\n=== Overall summary ===")
    for s in summaries:
        if s.get("status") != "ok":
            print(f"  {s['run_label']}: {s.get('status')}")
        else:
            print(f"  {s['run_label']}: processed={s['processed']}  "
                  f"skipped={s['skipped']}  missing={s['missing']}  failed={s['failed']}")
    print(f"\nOutputs under: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
