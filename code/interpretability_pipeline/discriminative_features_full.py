#!/usr/bin/env python3
"""
discriminative_features_full.py

Full-sweep version of discriminative_features.py.

For ALL acceptor labels (one-vs-rest) and ALL donor labels (one-vs-rest),
test every (feature_dim, residue_position_bin) cell — using ALL 1536 hidden
dimensions and 50 relative-position bins — for ability to discriminate
positive vs negative proteins.

Key design points:
  * The full (n_proteins, 1536, 50) activation tensor is built ONCE and reused
    across every label task. This is the major speedup over running the
    smaller-scope script 25 times.
  * AUROC is rank-based, so we do NOT standardize features. Saves time and
    avoids the per-label scaler dependency.
  * Per-label permutation null with empirical p-values. We add a small
    deterministic jitter to break ties in activation values.
  * Two summary outputs:
      - ranked_summary.tsv : every label ranked by signal lift over chance.
      - per_label_top_cells/<safe_label>_top_cells.tsv : every cell with
        empirical p ≤ 0.01 (i.e., neg_log10_p ≥ 2).
  * Per-label discriminative-heatmap PNGs are emitted only for labels with
    meaningful signal (configurable threshold, default frac_cells_above_null > 8%).

Usage:
    python discriminative_features_full.py \
        --root_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability \
        --metrics_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability/5fold_cv_ll_lr_preds
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-sweep AUROC analysis: 1536 features x 50 bins x all labels."
    )
    p.add_argument("--root_dir", required=True)
    p.add_argument("--metrics_dir", default=None,
                   help="Folder containing the *_metrics.json files. Default: search under root_dir.")
    p.add_argument("--last_hidden_dir", default=None,
                   help="Default: <root>/BAHD_lastLayer_embeddings")
    p.add_argument("--labels_path", default=None,
                   help="Default: <root>/BAHD_dataset/labels.pt")
    p.add_argument("--structures_dir", default=None,
                   help="Default: <root>/BAHD_dataset/BAHD_structures (used for BOS/EOS detection)")
    p.add_argument("--out_dir", default=None,
                   help="Default: <root>/discriminative_features_full")
    p.add_argument("--n_relpos_bins", type=int, default=50)
    p.add_argument("--n_permutations", type=int, default=200)
    p.add_argument("--heatmap_signal_threshold", type=float, default=0.08,
                   help="Generate heatmap only for labels whose frac_above_null exceeds this.")
    p.add_argument("--top_cells_neg_log10_p_threshold", type=float, default=3.0,
                   help="Include cells with neg_log10_p >= this in per-label top-cells TSV. "
                        "Default 3.0 (p <= 0.001). Use 2.0 for p <= 0.01 (much larger files).")
    p.add_argument("--top_features_in_heatmap", type=int, default=200,
                   help="When rendering heatmaps, show only the top-N features by max-bin AUROC.")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--cache_activation_tensor", action="store_true",
                   help="Save the (n_proteins, 1536, n_bins) tensor to disk for reuse.")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# IO helpers (mirrors existing scripts)
# --------------------------------------------------------------------------------------

def find_lasthidden_tensor(fasta_id: str, embeddings_dir: Path) -> Optional[Path]:
    p = embeddings_dir / f"{fasta_id}_hidden_layer_steps10.pt"
    if p.is_file():
        return p
    matches = list(embeddings_dir.glob(f"*{fasta_id}*hidden_layer*.pt"))
    return matches[0] if len(matches) == 1 else None


def count_pdb_residues(pdb_path: Path) -> int:
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


def load_real_residue_tensor(
    fasta_id: str, embeddings_dir: Path, structures_root: Optional[Path],
) -> Optional[np.ndarray]:
    p = find_lasthidden_tensor(fasta_id, embeddings_dir)
    if p is None:
        return None
    t = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(t, torch.Tensor):
        return None
    t = t.detach().float()
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 2:
        return None
    arr = t.numpy().astype(np.float32, copy=False)
    if structures_root is not None:
        sub = structures_root / fasta_id
        if sub.is_dir():
            pdbs = list(sub.rglob("*.pdb"))
            if pdbs:
                t_pdb = count_pdb_residues(pdbs[0])
                if arr.shape[0] == t_pdb + 2:
                    arr = arr[1:-1]
    return arr


def bin_to_relpos(values_t_d: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Mean-bin a (T, D) array onto a [0, 1] residue axis, returning (D, n_bins).
    Vectorized across feature dimensions for speed.
    """
    T, D = values_t_d.shape
    if T == 0:
        return np.zeros((D, n_bins), dtype=np.float32)
    pos = (np.arange(T) + 0.5) / T
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)  # (T,)

    sums = np.zeros((n_bins, D), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(sums, idx, values_t_d.astype(np.float64))
    np.add.at(counts, idx, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(counts[:, None] > 0, sums / counts[:, None], 0.0)
    return avg.T.astype(np.float32, copy=False)  # (D, n_bins)


# --------------------------------------------------------------------------------------
# Vectorized AUROC over many cells
# --------------------------------------------------------------------------------------

def auroc_vectorized(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    AUROC across many parallel binary problems with shared labels.

    scores: (n_proteins, n_cells)  - cell scores per protein
    labels: (n_proteins,)          - 0/1 across proteins, shared across cells
    Returns: (n_cells,) AUROC values, NaN where degenerate.
    """
    n_proteins, n_cells = scores.shape
    n_pos = int(labels.sum())
    n_neg = int(n_proteins - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(n_cells, np.nan)

    s = scores
    if not np.isfinite(s).all():
        s = np.where(np.isnan(s), -np.inf, s)

    order = np.argsort(s, axis=0, kind="mergesort")              # (n_proteins, n_cells)
    ranks = np.empty_like(order, dtype=np.float64)
    rank_values = np.arange(1, n_proteins + 1, dtype=np.float64)
    np.put_along_axis(
        ranks, order,
        np.broadcast_to(rank_values[:, None], (n_proteins, n_cells)).copy(),
        axis=0,
    )
    pos_idx = np.where(labels == 1)[0]
    pos_rank_sum = ranks[pos_idx].sum(axis=0)                    # (n_cells,)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def precompute_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Sort-once helper. Returns ranks of shape (n_proteins, n_cells), float32,
    where ranks[i, c] is the rank (1-indexed) of protein i in cell c.

    For the AUROC permutation loop we only need to compute this once and then
    permute label assignments cheaply.
    """
    n_proteins, n_cells = scores.shape
    s = scores
    if not np.isfinite(s).all():
        s = np.where(np.isnan(s), -np.inf, s)
    order = np.argsort(s, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rank_values = np.arange(1, n_proteins + 1, dtype=np.float32)
    np.put_along_axis(
        ranks, order,
        np.broadcast_to(rank_values[:, None], (n_proteins, n_cells)).copy(),
        axis=0,
    )
    return ranks


def auroc_from_precomputed_ranks(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute AUROC for many cells given precomputed per-cell ranks and a binary
    label vector. Fast path for the permutation loop.

    ranks:  (n_proteins, n_cells) float32 — output of precompute_ranks
    labels: (n_proteins,)         uint8/bool
    Returns: (n_cells,) float32 AUROC
    """
    n_proteins = ranks.shape[0]
    n_pos = int(labels.sum())
    n_neg = n_proteins - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(ranks.shape[1], np.nan, dtype=np.float32)
    pos_idx = np.where(labels == 1)[0]
    pos_rank_sum = ranks[pos_idx].sum(axis=0, dtype=np.float64)
    auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (float(n_pos) * float(n_neg))
    return auroc.astype(np.float32)


# --------------------------------------------------------------------------------------
# Activation tensor: built once, reused across all labels
# --------------------------------------------------------------------------------------

def build_activation_tensor(
    fasta_ids: Sequence[str],
    embeddings_dir: Path,
    structures_root: Optional[Path],
    n_bins: int,
    cache_path: Optional[Path],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build (n_proteins, D, n_bins) tensor of bin-averaged raw activations.
    Returns (activations, valid_mask, D).
    """
    # If a cached tensor exists, use it.
    if cache_path is not None and cache_path.is_file():
        print(f"[cache] loading activation tensor from {cache_path}")
        payload = np.load(cache_path)
        return payload["activations"], payload["valid_mask"], int(payload["D"])

    n_proteins = len(fasta_ids)
    print(f"[build] reading {n_proteins} per-residue tensors and binning to {n_bins} bins...")
    t0 = time.time()

    activations: Optional[np.ndarray] = None
    valid_mask = np.zeros(n_proteins, dtype=bool)
    D: Optional[int] = None

    for pi, fid in enumerate(fasta_ids):
        H_real = load_real_residue_tensor(fid, embeddings_dir, structures_root)
        if H_real is None:
            continue
        if D is None:
            D = int(H_real.shape[1])
            activations = np.zeros((n_proteins, D, n_bins), dtype=np.float32)
            print(f"[build] hidden dim D = {D}; allocating activation tensor of size "
                  f"{(n_proteins * D * n_bins * 4) / 1e6:.1f} MB")
        elif int(H_real.shape[1]) != D:
            print(f"[warn] {fid}: D mismatch ({H_real.shape[1]} vs {D}); skipping")
            continue
        activations[pi] = bin_to_relpos(H_real, n_bins)
        valid_mask[pi] = True
        if (pi + 1) % 50 == 0:
            print(f"[build]   loaded {pi + 1}/{n_proteins}  "
                  f"elapsed={time.time()-t0:.1f}s")

    if activations is None or D is None:
        raise SystemExit("No valid per-residue tensors loaded.")

    # Add a tiny deterministic jitter to break ties in AUROC ranking.
    # Scale based on observed activation magnitudes to ensure jitter is far smaller
    # than any real signal but breaks ties.
    obs_scale = np.std(activations[valid_mask])
    jitter_scale = obs_scale * 1e-9
    activations += rng.standard_normal(activations.shape).astype(np.float32) * jitter_scale

    print(f"[build] complete in {time.time()-t0:.1f}s; "
          f"valid proteins: {valid_mask.sum()}/{n_proteins}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[cache] saving to {cache_path}")
        np.savez_compressed(cache_path, activations=activations, valid_mask=valid_mask, D=np.array(D))

    return activations, valid_mask, D


# --------------------------------------------------------------------------------------
# Per-label analysis
# --------------------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def analyze_one_label(
    *,
    label: str,
    label_idx_in_all: int,
    activations: np.ndarray,                  # (n_proteins, D, n_bins)
    valid_mask: np.ndarray,                   # (n_proteins,)
    labels_full: np.ndarray,                  # (n_proteins, n_total_labels)
    n_permutations: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    n_proteins, D, n_bins = activations.shape
    y = labels_full[:, label_idx_in_all].astype(np.uint8)
    valid_idx = np.where(valid_mask)[0]
    y_valid = y[valid_idx]
    n_pos = int(y_valid.sum())
    n_neg = int(len(valid_idx) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return {"label": label, "skip_reason": "too_few_pos_or_neg",
                "n_pos": n_pos, "n_neg": n_neg}

    # Flatten activations to (n_valid_proteins, D * n_bins) once.
    flat = activations[valid_idx].reshape(len(valid_idx), D * n_bins)

    # Sort scores ONCE; reuse for all permutations. This is the major speedup.
    ranks = precompute_ranks(flat)

    # Real AUROC.
    auroc_real = auroc_from_precomputed_ranks(ranks, y_valid).reshape(D, n_bins)

    # Permutation null. We track:
    #  - higher: per-cell count of perms with AUROC >= real_auroc (one-sided high)
    #  - lower:  per-cell count of perms with AUROC <= real_auroc (one-sided low)
    # Two-sided p = 2 * min(higher, lower) / (n_perm + 1), or simply use whichever side
    # is more extreme. For simplicity we use a TWO-SIDED test by taking the deviation
    # from 0.5 in absolute value.
    abs_dev = np.abs(auroc_real - 0.5)
    higher_or_equal = np.zeros((D, n_bins), dtype=np.int32)
    perm_max_devs: List[float] = []
    for k in range(n_permutations):
        y_perm = rng.permutation(y_valid)
        a = auroc_from_precomputed_ranks(ranks, y_perm).reshape(D, n_bins)
        a_dev = np.abs(a - 0.5)
        higher_or_equal += (a_dev >= abs_dev).astype(np.int32)
        perm_max_devs.append(float(np.nanmax(a_dev)))

    p_emp = (higher_or_equal + 1) / float(n_permutations + 1)
    neg_log10_p = -np.log10(np.clip(p_emp, 1e-6, None))

    perm_threshold_95 = float(np.percentile(perm_max_devs, 95)) if perm_max_devs else float("nan")
    n_cells_above_perm = int(np.sum(np.abs(auroc_real - 0.5) > perm_threshold_95))

    return {
        "label": label,
        "skip_reason": "",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc_real": auroc_real,
        "p_emp": p_emp,
        "neg_log10_p": neg_log10_p,
        "perm_threshold_95_dev": perm_threshold_95,
        "max_auroc": float(np.nanmax(auroc_real)),
        "min_auroc": float(np.nanmin(auroc_real)),
        "max_abs_dev": float(np.nanmax(np.abs(auroc_real - 0.5))),
        "frac_cells_above_perm_95": n_cells_above_perm / float(D * n_bins),
        "n_cells_above_perm_95": n_cells_above_perm,
        "n_cells_neg_log10_p_ge_2": int(np.sum(neg_log10_p >= 2.0)),
        "n_cells_neg_log10_p_ge_3": int(np.sum(neg_log10_p >= 3.0)),
        "frac_cells_neg_log10_p_ge_2": float(np.mean(neg_log10_p >= 2.0)),
    }


# --------------------------------------------------------------------------------------
# Heatmap rendering
# --------------------------------------------------------------------------------------

def render_heatmap_for_label(
    label: str,
    auroc_real: np.ndarray,
    neg_log10_p: np.ndarray,
    perm_threshold_95_dev: float,
    n_permutations: int,
    n_pos: int,
    n_neg: int,
    out_path: Path,
    top_features: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D, n_bins = auroc_real.shape

    row_max_abs_dev = np.nanmax(np.abs(auroc_real - 0.5), axis=1)
    row_order = np.argsort(-row_max_abs_dev)
    show_n = min(top_features, D)
    sel = row_order[:show_n]
    auroc_sorted = auroc_real[sel]
    p_sorted = neg_log10_p[sel]

    fig, axes = plt.subplots(2, 1, figsize=(13, max(8.0, 0.04 * show_n + 4)), sharex=True)

    auroc_disp = np.where(np.isnan(auroc_sorted), 0.5, auroc_sorted) - 0.5
    finite_devs = auroc_disp[np.isfinite(auroc_disp)]
    vmax = float(max(np.percentile(np.abs(finite_devs), 99) if finite_devs.size else 0.05, 0.05))

    im0 = axes[0].imshow(auroc_disp, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[0].set_ylabel(f"feature rank (top {show_n} of {D})")
    axes[0].set_title(
        f"{label}\nAUROC − 0.5 per (feature, bin)   "
        f"|   n_pos={n_pos}, n_neg={n_neg}   "
        f"|   perm 95th-pct max |dev| = {perm_threshold_95_dev:.3f}"
    )
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.02)
    cbar0.set_label("AUROC − 0.5", fontsize=9)
    cbar0.ax.tick_params(labelsize=8)

    p_disp = np.where(np.isnan(p_sorted), 0.0, p_sorted)
    vmax_p = float(max(p_disp.max(), 1.0))
    im1 = axes[1].imshow(p_disp, aspect="auto", cmap="viridis",
                         vmin=0.0, vmax=vmax_p, interpolation="nearest")
    axes[1].set_ylabel("feature rank")
    axes[1].set_xlabel(f"relative residue position bin (n={n_bins})")
    axes[1].set_title(
        f"-log10(empirical p) — two-sided test, {n_permutations} permutations\n"
        "rank order matches top panel"
    )
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.02)
    cbar1.set_label("-log10(p_emp)", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Per-label TSV writer
# --------------------------------------------------------------------------------------

def write_top_cells_tsv(
    label: str,
    auroc_real: np.ndarray,
    neg_log10_p: np.ndarray,
    p_emp: np.ndarray,
    n_bins: int,
    threshold_neg_log10_p: float,
    out_path: Path,
) -> int:
    """Write all cells with neg_log10_p >= threshold to TSV. Returns row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D = auroc_real.shape[0]

    keep = neg_log10_p >= threshold_neg_log10_p
    if not keep.any():
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["rank", "feature_idx", "bin_idx", "relpos_lo", "relpos_hi",
                        "auroc", "auroc_dev", "p_emp", "neg_log10_p"])
        return 0

    rows = []
    for fi in range(D):
        for bi in range(n_bins):
            if not keep[fi, bi]:
                continue
            a = float(auroc_real[fi, bi])
            rows.append((fi, bi, a, abs(a - 0.5),
                         float(p_emp[fi, bi]), float(neg_log10_p[fi, bi])))
    # Sort by absolute deviation from 0.5 (most discriminative first).
    rows.sort(key=lambda r: -r[3])

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["rank", "feature_idx", "bin_idx", "relpos_lo", "relpos_hi",
                    "auroc", "auroc_dev", "p_emp", "neg_log10_p"])
        for rank, (fi, bi, a, dev, pe, nlp) in enumerate(rows, start=1):
            w.writerow([
                rank, int(fi), int(bi),
                f"{bi / n_bins:.3f}", f"{(bi + 1) / n_bins:.3f}",
                f"{a:.4f}", f"{dev:.4f}",
                f"{pe:.4f}", f"{nlp:.3f}",
            ])
    return len(rows)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    metrics_dir = Path(args.metrics_dir).resolve() if args.metrics_dir else root
    embeddings_dir = (
        Path(args.last_hidden_dir).resolve() if args.last_hidden_dir
        else (root / "BAHD_lastLayer_embeddings")
    )
    labels_path = Path(args.labels_path).resolve() if args.labels_path else (root / "BAHD_dataset" / "labels.pt")
    structures_root_arg = (
        Path(args.structures_dir).resolve() if args.structures_dir
        else (root / "BAHD_dataset" / "BAHD_structures")
    )
    structures_root: Optional[Path] = structures_root_arg if structures_root_arg.is_dir() else None
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "discriminative_features_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.random_seed))

    print(f"[paths] root_dir:    {root}")
    print(f"[paths] embeddings:  {embeddings_dir}")
    print(f"[paths] labels:      {labels_path}")
    print(f"[paths] structures:  {structures_root if structures_root else '(skipped)'}")
    print(f"[paths] out_dir:     {out_dir}")
    print(f"[config] n_bins:     {args.n_relpos_bins}")
    print(f"[config] n_perm:     {args.n_permutations}")
    print(f"[config] heatmap signal threshold (frac_above_null): {args.heatmap_signal_threshold}")
    print(f"[config] top-cells neg_log10_p threshold:             {args.top_cells_neg_log10_p_threshold}")

    # --- Load label vocabulary
    cache = torch.load(labels_path, map_location="cpu", weights_only=False)
    fasta_ids = [str(x) for x in cache["fasta_ids"]]
    all_label_names = [str(x) for x in cache["label_names"]]
    labels_full = cache["labels"].detach().cpu().numpy().astype(np.uint8)
    family = str(cache.get("family", "")).upper()

    # Determine which labels we'll evaluate.
    donor_prefix = "donor_type::" if family == "BAHD" else "donor_compound::"
    acceptor_prefix = "acceptor_superclass::"
    donor_idx = [i for i, n in enumerate(all_label_names) if n.startswith(donor_prefix)]
    acceptor_idx = [i for i, n in enumerate(all_label_names) if n.startswith(acceptor_prefix)]
    print(f"[labels] {len(donor_idx)} donor labels, {len(acceptor_idx)} acceptor labels")

    # Optional: load metrics JSONs to attach CV AUPR for context. Soft-fail if missing.
    cv_aupr_for_label: Dict[str, float] = {}
    cv_npos_for_label: Dict[str, int] = {}
    for task_name in ("acceptor", "donor"):
        for pat in [f"*hidden*lr*{task_name}*metrics.json",
                    f"*{task_name}*metrics.json"]:
            cands = sorted(metrics_dir.rglob(pat))
            if cands:
                with cands[0].open() as fh:
                    m = json.load(fh)
                for r in m.get("metrics", {}).get("per_label", []):
                    cv_aupr_for_label[r["label"]] = float(r.get("aupr") or 0.0)
                    cv_npos_for_label[r["label"]] = int(r.get("positive_count") or 0)
                break

    # --- Build the activation tensor once.
    cache_path = (out_dir / f"_activation_tensor_n{args.n_relpos_bins}.npz") if args.cache_activation_tensor else None
    activations, valid_mask, D = build_activation_tensor(
        fasta_ids=fasta_ids,
        embeddings_dir=embeddings_dir,
        structures_root=structures_root,
        n_bins=int(args.n_relpos_bins),
        cache_path=cache_path,
        rng=rng,
    )

    # --- Per-label analysis
    label_results: List[Dict[str, Any]] = []
    label_indices_to_run = donor_idx + acceptor_idx
    print(f"\n[run] starting analysis on {len(label_indices_to_run)} labels...")

    per_label_top_cells_dir = out_dir / "per_label_top_cells"
    per_label_heatmap_dir = out_dir / "per_label_heatmaps"

    t_start = time.time()
    for k, label_idx in enumerate(label_indices_to_run):
        label = all_label_names[label_idx]
        t0 = time.time()
        result = analyze_one_label(
            label=label, label_idx_in_all=label_idx,
            activations=activations, valid_mask=valid_mask,
            labels_full=labels_full,
            n_permutations=int(args.n_permutations),
            rng=rng,
        )
        if result is None:
            continue
        elapsed = time.time() - t0
        skip = result.get("skip_reason") or ""
        if skip:
            print(f"[{k+1}/{len(label_indices_to_run)}] {label}  -> SKIP ({skip})  "
                  f"npos={result.get('n_pos')}  ({elapsed:.1f}s)")
            label_results.append({
                "label": label,
                "n_pos": result.get("n_pos", 0),
                "n_neg": result.get("n_neg", 0),
                "skip_reason": skip,
                "max_auroc": float("nan"),
                "max_abs_dev": float("nan"),
                "perm_threshold_95_dev": float("nan"),
                "n_cells_above_perm_95": 0,
                "frac_cells_above_perm_95": 0.0,
                "n_cells_neg_log10_p_ge_2": 0,
                "n_cells_neg_log10_p_ge_3": 0,
                "cv_aupr": cv_aupr_for_label.get(label, float("nan")),
                "cv_positive_count": cv_npos_for_label.get(label, 0),
                "heatmap_path": "",
                "top_cells_path": "",
                "top_cells_kept": 0,
            })
            continue

        # Per-label TSV
        safe = safe_filename(label)
        tsv_path = per_label_top_cells_dir / f"{safe}__top_cells.tsv"
        n_kept = write_top_cells_tsv(
            label=label,
            auroc_real=result["auroc_real"],
            neg_log10_p=result["neg_log10_p"],
            p_emp=result["p_emp"],
            n_bins=int(args.n_relpos_bins),
            threshold_neg_log10_p=float(args.top_cells_neg_log10_p_threshold),
            out_path=tsv_path,
        )

        # Conditionally render heatmap
        heatmap_path = ""
        if result["frac_cells_above_perm_95"] >= args.heatmap_signal_threshold:
            png_path = per_label_heatmap_dir / f"{safe}__discriminative.png"
            render_heatmap_for_label(
                label=label,
                auroc_real=result["auroc_real"],
                neg_log10_p=result["neg_log10_p"],
                perm_threshold_95_dev=result["perm_threshold_95_dev"],
                n_permutations=int(args.n_permutations),
                n_pos=result["n_pos"], n_neg=result["n_neg"],
                out_path=png_path,
                top_features=int(args.top_features_in_heatmap),
            )
            heatmap_path = str(png_path)

        print(f"[{k+1}/{len(label_indices_to_run)}] {label}  "
              f"npos={result['n_pos']:3d}  "
              f"max_auroc={result['max_auroc']:.3f}  "
              f"max_dev={result['max_abs_dev']:.3f}  "
              f"frac>null={result['frac_cells_above_perm_95']:.3f}  "
              f"top_cells_kept={n_kept}  ({elapsed:.1f}s)")

        label_results.append({
            "label": label,
            "n_pos": result["n_pos"],
            "n_neg": result["n_neg"],
            "skip_reason": "",
            "max_auroc": result["max_auroc"],
            "min_auroc": result["min_auroc"],
            "max_abs_dev": result["max_abs_dev"],
            "perm_threshold_95_dev": result["perm_threshold_95_dev"],
            "n_cells_above_perm_95": result["n_cells_above_perm_95"],
            "frac_cells_above_perm_95": result["frac_cells_above_perm_95"],
            "n_cells_neg_log10_p_ge_2": result["n_cells_neg_log10_p_ge_2"],
            "n_cells_neg_log10_p_ge_3": result["n_cells_neg_log10_p_ge_3"],
            "cv_aupr": cv_aupr_for_label.get(label, float("nan")),
            "cv_positive_count": cv_npos_for_label.get(label, 0),
            "heatmap_path": heatmap_path,
            "top_cells_path": str(tsv_path),
            "top_cells_kept": n_kept,
        })

    print(f"\n[run] all labels analyzed in {time.time()-t_start:.1f}s")

    # --- Ranked summary (sort by frac_cells_above_perm_95 descending; nan/skip last)
    label_results.sort(
        key=lambda r: (
            -1.0 if r.get("skip_reason") else 0.0,
            -float(r.get("frac_cells_above_perm_95") or 0.0),
            -float(r.get("max_abs_dev") or 0.0),
        )
    )
    summary_path = out_dir / "ranked_summary.tsv"
    fieldnames = [
        "rank", "label", "cv_aupr", "cv_positive_count",
        "n_pos", "n_neg",
        "max_auroc", "min_auroc", "max_abs_dev", "perm_threshold_95_dev",
        "n_cells_above_perm_95", "frac_cells_above_perm_95",
        "n_cells_neg_log10_p_ge_2", "n_cells_neg_log10_p_ge_3",
        "top_cells_kept", "skip_reason",
        "top_cells_path", "heatmap_path",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for rank, r in enumerate(label_results, start=1):
            row = {k_: r.get(k_, "") for k_ in fieldnames}
            row["rank"] = rank
            for k_ in ("max_auroc", "min_auroc", "max_abs_dev", "perm_threshold_95_dev",
                       "frac_cells_above_perm_95", "cv_aupr"):
                v = row.get(k_)
                if isinstance(v, float):
                    row[k_] = "" if (np.isnan(v) if v == v else False) or v != v else f"{v:.4f}"
            w.writerow(row)
    print(f"[summary] wrote {summary_path}")

    print("\nDone. Outputs at", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
