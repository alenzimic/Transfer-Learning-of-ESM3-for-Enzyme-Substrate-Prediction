#!/usr/bin/env python3
"""
figure_3_extended.py

Extended version of Figure 3 from writeup_figures.py: cross-protein feature
activation heatmaps for top features, but showing ALL proteins in the dataset
(positives in the top section, negatives below a divider) rather than just
the positive class.

Purpose: visually answer "are these features Phenolic-acid-specific or generic
across all proteins?" If specific, the top section (positives) should show
red bands at consistent positions while the bottom section (negatives) should
look different. If generic, the whole heatmap will look uniform top to bottom.

For each top feature:
  - Sort positives by model score on the label (most confident positive at top)
  - Sort negatives by model score (least confident negative just below divider,
    most confident negative at bottom)
  - Stack them; draw a divider line at the positive/negative boundary
  - Color = sign-aligned, standardized activation, mean-binned to 50 relative
    position bins (matches the existing Figure 3 visual)

Usage:
    python figure_3_extended.py \
        --root_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability \
        --label "Phenolic acids (C6-C1)" \
        --n_top_features 3
"""

from __future__ import annotations

import argparse
import pickle
import sys
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
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True)
    p.add_argument("--label", nargs="+", default=["Phenolic acids (C6-C1)"],
                   help="Short label name(s) without prefix. Pass multiple to process more "
                        "than one label in a single run.")
    p.add_argument("--task_prefix", default="acceptor_superclass::")
    p.add_argument("--n_top_features", type=int, default=3)
    p.add_argument("--metrics_dir", default=None,
                   help="Accepted for launcher compatibility but not used by this script.")
    p.add_argument("--n_relpos_bins", type=int, default=50)
    p.add_argument("--canvas_format", choices=["png", "html", "both"], default="png",
                   help="Format for the per-feature canvas figures. "
                        "'png' = matplotlib 9x5 sphere-scatter grid. "
                        "'html' = py3Dmol cartoon ribbon viewers (one HTML per feature). "
                        "'both' = produce both. Default 'png'.")
    p.add_argument("--pickle_dir", default=None)
    p.add_argument("--last_hidden_dir", default=None)
    p.add_argument("--structures_dir", default=None)
    p.add_argument("--labels_path", default=None)
    p.add_argument("--interp_dir", default=None)
    p.add_argument("--top_cells_dir", default=None,
                   help="Folder containing the per-label discriminative-features TSVs. "
                        "Default: <root>/discriminative_features_full/per_label_top_cells")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# IO (mirrors the other scripts)
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


def bin_to_relpos(values_t: np.ndarray, n_bins: int) -> np.ndarray:
    T = len(values_t)
    if T == 0:
        return np.full(n_bins, np.nan, dtype=np.float64)
    pos = (np.arange(T) + 0.5) / T
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(sums, idx, values_t.astype(np.float64))
    np.add.at(counts, idx, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def load_pipeline_for_label(pickle_path: Path, label: str):
    with pickle_path.open("rb") as fh:
        bundle = pickle.load(fh)
    if label not in bundle["models"]:
        raise KeyError(label)
    entry = bundle["models"][label]
    if entry["mode"] != "trained":
        raise RuntimeError(f"{label}: constant fallback")
    pipe = entry["pipeline"]
    mu = np.asarray(pipe.named_steps["scaler"].mean_, dtype=np.float64)
    sigma = np.asarray(pipe.named_steps["scaler"].scale_, dtype=np.float64)
    coef = np.asarray(pipe.named_steps["clf"].coef_, dtype=np.float64)
    w = coef[0] if coef.ndim == 2 else coef
    return mu, sigma, w


def load_per_protein_scores(
    label: str, interp_dir: Path, interp_subdir: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pdir = interp_dir / interp_subdir / "per_protein"
    if not pdir.is_dir():
        return out
    for f in pdir.glob("*.pt"):
        try:
            b = torch.load(f, map_location="cpu", weights_only=False)
            names = list(b["label_names"])
            if label not in names:
                continue
            li = names.index(label)
            s = b["scores"]
            if hasattr(s, "numpy"):
                s = s.numpy()
            out[str(b["fasta_id"])] = float(s[li])
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------------------
# Build the heatmap data
# --------------------------------------------------------------------------------------

def build_heatmap_data(
    feature_idx: int,
    fasta_ids_ordered: List[str],
    embeddings_dir: Path,
    structures_root: Optional[Path],
    mu: np.ndarray,
    sigma: np.ndarray,
    sign_aligned: bool,
    coef_sign: float,
    n_bins: int,
) -> np.ndarray:
    """
    Returns an (n_proteins, n_bins) array of standardized, mean-binned
    activations for the given feature, ordered by fasta_ids_ordered.
    Sign-aligned so red consistently means "pushes prediction toward positive."
    Missing proteins yield NaN rows.
    """
    n = len(fasta_ids_ordered)
    out = np.full((n, n_bins), np.nan, dtype=np.float32)
    for i, fid in enumerate(fasta_ids_ordered):
        H = load_real_residue_tensor(fid, embeddings_dir, structures_root)
        if H is None or feature_idx >= H.shape[1]:
            continue
        col = (H[:, feature_idx].astype(np.float64) - mu[feature_idx]) / sigma[feature_idx]
        if sign_aligned:
            col = col * coef_sign
        out[i] = bin_to_relpos(col, n_bins).astype(np.float32, copy=False)
    return out


def plot_extended_heatmap_grid(
    *,
    label: str,
    short_label: str,
    feature_groups: List[Dict[str, Any]],     # list of {"method", "method_label", "features"}
    fasta_order_pos: List[str],
    fasta_order_neg: List[str],
    embeddings_dir: Path,
    structures_root: Optional[Path],
    mu: np.ndarray,
    sigma: np.ndarray,
    w_over_sigma: np.ndarray,
    n_bins: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fasta_order_all = fasta_order_pos + fasta_order_neg
    n_pos = len(fasta_order_pos)
    n_neg = len(fasta_order_neg)
    n_total = n_pos + n_neg
    n_methods = len(feature_groups)
    n_per_method = max(len(g["features"]) for g in feature_groups) if feature_groups else 0

    # Cache so a feature appearing in multiple columns is only read once.
    data_cache: Dict[Tuple[int, int], np.ndarray] = {}  # (feature_idx, sign) -> heatmap

    def _get_data(fi: int) -> Tuple[np.ndarray, float]:
        coef = float(w_over_sigma[fi])
        sign = 1.0 if coef >= 0 else -1.0
        key = (int(fi), int(sign))
        if key in data_cache:
            return data_cache[key], sign
        data = build_heatmap_data(
            feature_idx=int(fi),
            fasta_ids_ordered=fasta_order_all,
            embeddings_dir=embeddings_dir,
            structures_root=structures_root,
            mu=mu, sigma=sigma,
            sign_aligned=True, coef_sign=sign,
            n_bins=n_bins,
        )
        data_cache[key] = data
        return data, sign

    # Layout: rows = ranks, columns = methods.
    fig_w = 4.6 * n_methods + 1.5
    fig_h = 4.0 * n_per_method + 2.0
    fig, axes = plt.subplots(n_per_method, n_methods, figsize=(fig_w, fig_h))
    if n_per_method == 1 and n_methods == 1:
        axes = np.array([[axes]])
    elif n_per_method == 1:
        axes = axes[np.newaxis, :]
    elif n_methods == 1:
        axes = axes[:, np.newaxis]

    # Column header titles (one per method).
    # Panels stop at y=0.87 (per tight_layout rect), suptitle 3 lines reaches down
    # to about y=0.92. Column labels sit between, near y=0.89.
    for col, group in enumerate(feature_groups):
        x_center = (col + 0.5) / n_methods
        fig.text(
            x_center, 0.89,
            group["method_label"],
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    for row in range(n_per_method):
        for col, group in enumerate(feature_groups):
            ax = axes[row, col]
            features = group["features"]
            if row >= len(features):
                ax.set_axis_off()
                continue
            fi_info = features[row]
            fi = int(fi_info["feature_idx"])
            data, sign = _get_data(fi)
            print(f"  [{group['method']}] rank {row+1}: feature {fi}  "
                  f"raw_coef={float(w_over_sigma[fi]):+.4f}")

            finite = data[np.isfinite(data)]
            vmax = float(max(np.percentile(np.abs(finite), 99) if finite.size else 1.0, 1e-3))

            im = ax.imshow(
                data, aspect="auto", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax, interpolation="nearest",
            )
            # Divider line between positives and negatives.
            ax.axhline(y=n_pos - 0.5, color="black", linewidth=1.0, linestyle="-", alpha=0.85)

            # Build a compact title with the criterion's score for this feature.
            metric_str = group.get("score_label_fn", lambda info: "")(fi_info)
            title = (
                f"rank {row+1}: feat {fi}  |  raw coef = {float(w_over_sigma[fi]):+.4f}\n"
                f"{metric_str}"
            )
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"protein index\n(top {n_pos} = positives,\nbottom {n_neg} = negatives)",
                              fontsize=8)
            else:
                ax.set_yticks([])
            if row == n_per_method - 1:
                ax.set_xlabel(f"relative residue position bin (n={n_bins})", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=7)

            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Cross-protein feature activation heatmaps for {short_label}\n"
        f"({n_total} proteins shown: {n_pos} positives at top, {n_neg} negatives below black line; "
        f"each section sorted by model score descending)\n"
        f"Three feature-selection criteria compared (columns); top 3 features per criterion (rows). "
        f"Red = activation pushes prediction toward positive class.",
        fontsize=11, y=0.985,
    )
    # Reserve top ~13% for suptitle + column labels so panels don't crowd them.
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_extended_heatmap(
    *,
    label: str,
    short_label: str,
    feature_infos: List[Dict[str, Any]],
    fasta_order_pos: List[str],
    fasta_order_neg: List[str],
    embeddings_dir: Path,
    structures_root: Optional[Path],
    mu: np.ndarray,
    sigma: np.ndarray,
    w_over_sigma: np.ndarray,
    n_bins: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fasta_order_all = fasta_order_pos + fasta_order_neg
    n_pos = len(fasta_order_pos)
    n_neg = len(fasta_order_neg)
    n_total = n_pos + n_neg
    n_features = len(feature_infos)

    fig, axes = plt.subplots(
        n_features, 1,
        figsize=(13, max(8.0, 0.018 * n_total * n_features + 1.5)),
        sharex=True,
    )
    if n_features == 1:
        axes = [axes]

    for k, fi_info in enumerate(feature_infos):
        fi = int(fi_info["feature_idx"])
        coef = float(w_over_sigma[fi])
        sign = 1.0 if coef >= 0 else -1.0
        print(f"  building heatmap for feature {fi}  raw_coef={coef:+.4f}  sign={int(sign):+d}")
        data = build_heatmap_data(
            feature_idx=fi,
            fasta_ids_ordered=fasta_order_all,
            embeddings_dir=embeddings_dir,
            structures_root=structures_root,
            mu=mu, sigma=sigma,
            sign_aligned=True, coef_sign=sign,
            n_bins=n_bins,
        )
        finite = data[np.isfinite(data)]
        vmax = float(max(np.percentile(np.abs(finite), 99) if finite.size else 1.0, 1e-3))

        ax = axes[k]
        im = ax.imshow(
            data, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.axhline(y=n_pos - 0.5, color="black", linewidth=1.4, linestyle="-", alpha=0.85)
        ax.text(n_bins + 0.5, n_pos - 0.5, "← positives / negatives →",
                va="center", ha="left", fontsize=8, color="#333", rotation=90)

        ax.set_ylabel(
            f"protein index\n(positives n={n_pos} top, negatives n={n_neg} bottom;\n"
            "each section sorted by model score, descending)",
            fontsize=8,
        )
        title = (
            f"Feature {fi}  |  AUROC={fi_info.get('auroc', float('nan')):.3f}  |  "
            f"raw-space coefficient = {coef:+.4f}  |  "
            f"best discriminative bin: relpos {fi_info['relpos_lo']:.2f}-{fi_info['relpos_hi']:.2f}"
        )
        ax.set_title(title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("sign-aligned, standardized activation", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    axes[-1].set_xlabel(f"relative residue position bin (n={n_bins})")
    fig.suptitle(
        f"Extended Figure 3 — cross-protein feature activations for top {n_features} "
        f"discriminative features\n"
        f"label: {short_label}  (n_proteins shown: {n_total} = {n_pos} positives + {n_neg} negatives)\n"
        f"sign-aligned: red = activation pushes prediction toward positive class.",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def compute_full_pos_neg_means(
    fasta_ids: List[str],
    truth: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    embeddings_dir: Path,
    structures_root: Optional[Path],
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute per-feature, per-bin standardized activation means for positives and
    negatives across the dataset.

    Returns:
        pos_means:  (D, n_bins) standardized mean over positives
        neg_means:  (D, n_bins) standardized mean over negatives
        n_pos_used, n_neg_used
    """
    pos_sums: Optional[np.ndarray] = None
    pos_counts: Optional[np.ndarray] = None
    neg_sums: Optional[np.ndarray] = None
    neg_counts: Optional[np.ndarray] = None
    n_pos_used = 0
    n_neg_used = 0

    for fid, y in zip(fasta_ids, truth):
        H = load_real_residue_tensor(fid, embeddings_dir, structures_root)
        if H is None:
            continue
        T, D = H.shape
        H_std = (H.astype(np.float64) - mu[None, :D]) / sigma[None, :D]
        # Bin each feature column to (D, n_bins). Vectorized version of bin_to_relpos.
        pos = (np.arange(T) + 0.5) / T
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)

        # Per-protein bin sums of standardized activation, vectorized over D.
        binned_sums = np.zeros((n_bins, D), dtype=np.float64)
        binned_counts = np.zeros(n_bins, dtype=np.float64)
        np.add.at(binned_sums, bin_idx, H_std)
        np.add.at(binned_counts, bin_idx, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            binned_means = np.where(
                binned_counts[:, None] > 0,
                binned_sums / binned_counts[:, None], np.nan,
            )
        protein_means_DxB = binned_means.T.astype(np.float64)  # (D, n_bins)

        if pos_sums is None:
            pos_sums = np.zeros_like(protein_means_DxB)
            pos_counts = np.zeros_like(protein_means_DxB)
            neg_sums = np.zeros_like(protein_means_DxB)
            neg_counts = np.zeros_like(protein_means_DxB)
        valid_mask = ~np.isnan(protein_means_DxB)
        if y == 1:
            pos_sums[valid_mask] += protein_means_DxB[valid_mask]
            pos_counts[valid_mask] += 1.0
            n_pos_used += 1
        else:
            neg_sums[valid_mask] += protein_means_DxB[valid_mask]
            neg_counts[valid_mask] += 1.0
            n_neg_used += 1

    if pos_sums is None:
        raise SystemExit("could not load any per-residue tensors")

    with np.errstate(invalid="ignore", divide="ignore"):
        pos_means = np.where(pos_counts > 0, pos_sums / pos_counts, np.nan)
        neg_means = np.where(neg_counts > 0, neg_sums / neg_counts, np.nan)
    return pos_means, neg_means, n_pos_used, n_neg_used


# --------------------------------------------------------------------------------------
# Protein canvas grid: 9 features × 5 canvases of (pos_mean - neg_mean) signed diff
# --------------------------------------------------------------------------------------

def find_pdb(fasta_id: str, structures_root: Path) -> Optional[Path]:
    sub = structures_root / fasta_id
    if not sub.is_dir():
        return None
    pdbs = sorted(sub.rglob("*.pdb"))
    return pdbs[0] if pdbs else None


def load_ca_coords(pdb_path: Path) -> Optional[np.ndarray]:
    """Return (T, 3) Cα coordinates from the first model of the PDB."""
    coords: List[Tuple[float, float, float]] = []
    seen: set = set()
    in_first_model = True
    with pdb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                if coords:
                    in_first_model = False
                continue
            if line.startswith("ENDMDL"):
                in_first_model = False
                continue
            if not in_first_model:
                continue
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21:22]; resseq = line[22:26]; icode = line[26:27]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    return np.asarray(coords, dtype=np.float32) if coords else None


def render_canvas_panel(
    ax,
    ca_coords: np.ndarray,
    signed_diff_per_bin: np.ndarray,
    n_bins: int,
    norm,
    cmap,
    marker_size: float = 18.0,
    title: Optional[str] = None,
) -> None:
    """Render one Cα-scatter panel colored by per-residue (pos-neg) signed diff."""
    T = len(ca_coords)
    pos = (np.arange(T) + 0.5) / T
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)
    per_residue = signed_diff_per_bin[bin_idx]
    per_residue_safe = np.where(np.isfinite(per_residue), per_residue, 0.0)

    ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
            color="#cccccc", linewidth=0.5, alpha=0.6, zorder=1)
    ax.scatter(
        ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
        c=per_residue_safe, cmap=cmap, norm=norm,
        s=marker_size, depthshade=True,
        linewidths=0.2, edgecolors=(0, 0, 0, 0.2), zorder=3,
    )

    mn = ca_coords.min(axis=0); mx = ca_coords.max(axis=0)
    rng = (mx - mn).max() / 2.0
    mid = (mn + mx) / 2.0
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((1, 1, 1, 0))
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    ax.view_init(elev=20, azim=35)


def plot_protein_canvas_grid(
    *,
    label: str,
    short_label: str,
    feature_groups: List[Dict[str, Any]],   # same shape as for plot_extended_heatmap_grid
    canvas_ids: List[str],                  # 5 positive proteins
    pos_means: np.ndarray,                  # (D, n_bins) standardized
    neg_means: np.ndarray,                  # (D, n_bins) standardized
    structures_root: Path,
    n_bins: int,
    out_path: Path,
) -> None:
    """
    For each (method, rank, feature) and each canvas protein, render a Cα-scatter
    colored by the dataset-level signed (pos_mean - neg_mean) signal at that
    residue's relative-position bin.
    """
    from matplotlib.colors import TwoSlopeNorm

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten the feature groups into a single list of (method, rank, feature_info)
    # so we can iterate row-by-row in the same order as the heatmap grid:
    #   method 1 rank 1, method 2 rank 1, method 3 rank 1, method 1 rank 2, etc.
    # Actually for readability, group by method (3 rows per method consecutively).
    rows: List[Dict[str, Any]] = []
    for group in feature_groups:
        for rank, fi_info in enumerate(group["features"], start=1):
            rows.append({
                "method": group["method"],
                "method_label": group["method_label"],
                "rank": rank,
                "feature_idx": int(fi_info["feature_idx"]),
                "info": fi_info,
            })
    n_rows = len(rows)
    n_cols = len(canvas_ids)

    # Pre-load Cα for each canvas (used for every row).
    print(f"  loading Cα coords for {n_cols} canvases...")
    canvas_data: Dict[str, np.ndarray] = {}
    for fid in canvas_ids:
        pdb = find_pdb(fid, structures_root)
        if pdb is None:
            print(f"    [warn] no PDB for {fid}")
            continue
        ca = load_ca_coords(pdb)
        if ca is None:
            print(f"    [warn] could not parse Cα for {fid}")
            continue
        canvas_data[fid] = ca
    valid_canvases = [fid for fid in canvas_ids if fid in canvas_data]
    n_cols = len(valid_canvases)
    if n_cols == 0:
        print("  [skip] no usable canvases")
        return
    print(f"  using {n_cols} valid canvases")

    # Compute the full (D, n_bins) signed diff once. We'll slice rows by feature_idx.
    diff_full = pos_means - neg_means

    # Shared color scale across all 9*5 panels.
    used_diffs = []
    for r in rows:
        used_diffs.append(diff_full[r["feature_idx"]])
    used_diffs_arr = np.stack(used_diffs, axis=0)
    finite = used_diffs_arr[np.isfinite(used_diffs_arr)]
    if finite.size:
        scale = float(np.percentile(np.abs(finite), 95))
        if scale == 0:
            scale = float(np.max(np.abs(finite))) if finite.size else 1.0
        if scale == 0:
            scale = 1.0
    else:
        scale = 1.0
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    cmap = plt.get_cmap("RdBu_r")
    print(f"  shared color scale across all panels: ±{scale:.3f}")

    # Layout
    cell_w = 2.4    # inches per panel
    cell_h = 2.4
    left_label_w = 1.8
    top_header_h = 0.85
    bottom_cbar_h = 0.7
    fig_w = left_label_w + cell_w * n_cols + 0.4
    fig_h = top_header_h + cell_h * n_rows + bottom_cbar_h + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Per-residue (positive − negative) mean activation difference, "
        f"projected onto positive {short_label} structures\n"
        f"Rows = top features (3 selection methods × 3 ranks)  |  "
        f"Columns = top {n_cols} positive proteins (same canvases each row)\n"
        f"Color = sign-aligned mean(positive activation) − mean(negative activation) at each residue's bin",
        fontsize=11, y=0.995,
    )

    # Position math (in figure-fraction units).
    grid_left = left_label_w / fig_w
    grid_right = 1.0 - 0.02
    grid_top = 1.0 - (top_header_h / fig_h)
    grid_bottom = bottom_cbar_h / fig_h + 0.04
    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom
    panel_w = grid_w / n_cols
    panel_h = grid_h / n_rows

    # Column header: protein IDs.
    for c, fid in enumerate(valid_canvases):
        x_center = grid_left + (c + 0.5) * panel_w
        y = grid_top + 0.005
        fig.text(x_center, y, fid, ha="center", va="bottom", fontsize=8,
                 color="#8b1a14")

    # Light horizontal separator after every 3 rows (between methods).
    for k in range(1, len(feature_groups)):
        # The separator goes between the (3*k - 1)-th and (3*k)-th rows (0-indexed).
        boundary_y = grid_top - (3 * k) * panel_h
        fig.lines.append(plt.Line2D(
            [grid_left - 0.015, grid_right],
            [boundary_y, boundary_y],
            transform=fig.transFigure,
            color="#888", linestyle="-", linewidth=0.6,
        ))

    # Render rows.
    for r, row_info in enumerate(rows):
        fi = int(row_info["feature_idx"])
        diff_for_feat = diff_full[fi]
        # Row label on the left.
        method_short = {
            "coef": "by |w/σ|",
            "auroc": "by AUROC",
            "diff": "by |Δμ|",
        }.get(row_info["method"], row_info["method"])
        row_text = (
            f"{method_short}\n"
            f"rank {row_info['rank']}\n"
            f"feat {fi}"
        )
        y_center = grid_top - (r + 0.5) * panel_h
        fig.text(grid_left - 0.012, y_center, row_text,
                 ha="right", va="center", fontsize=8)

        for c, fid in enumerate(valid_canvases):
            ca = canvas_data[fid]
            x_left = grid_left + c * panel_w
            y_bottom = grid_top - (r + 1) * panel_h
            ax = fig.add_axes([x_left, y_bottom, panel_w * 0.95, panel_h * 0.95],
                              projection="3d")
            ax.set_facecolor("white")
            render_canvas_panel(
                ax=ax, ca_coords=ca, signed_diff_per_bin=diff_for_feat,
                n_bins=n_bins, norm=norm, cmap=cmap,
                marker_size=18.0, title=None,
            )

    # Shared colorbar at the bottom.
    cbar_ax = fig.add_axes([0.30, 0.022, 0.40, 0.018])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "mean(positive) − mean(negative) standardized activation per bin",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=140, bbox_inches=None)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Cartoon-ribbon HTML output: one HTML per feature, 5 cartoon viewers per file
# --------------------------------------------------------------------------------------

def _build_pdb_residue_order(src_pdb: Path) -> List[Tuple[str, str, str]]:
    """Walk the first model and return ordered (chain, resseq, icode) tuples."""
    order: List[Tuple[str, str, str]] = []
    seen: set = set()
    in_first_model = True
    with src_pdb.open("r", encoding="utf-8") as fh:
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
                key = (line[21:22], line[22:26], line[26:27])
                if key not in seen:
                    seen.add(key)
                    order.append(key)
    return order


def _write_recolored_pdb_for_html(
    src_pdb: Path,
    per_residue_values: np.ndarray,
    out_path: Path,
) -> Tuple[bool, int, int]:
    """
    Copy src_pdb to out_path with per-residue values placed in the B-factor column.
    The values array must already be aligned with the BOS/EOS-stripped residue order.
    Returns (alignment_ok, n_pdb_residues, n_array_values).
    """
    order = _build_pdb_residue_order(src_pdb)
    n_pdb = len(order)
    T = int(per_residue_values.shape[0])
    n_assign = min(n_pdb, T)
    res_to_b: Dict[Tuple[str, str, str], float] = {}
    for i in range(n_assign):
        v = float(per_residue_values[i])
        if not np.isfinite(v):
            v = 0.0
        res_to_b[order[i]] = v
    for i in range(n_assign, n_pdb):
        res_to_b[order[i]] = 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with src_pdb.open("r", encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain = line[21:22]; resseq = line[22:26]; icode = line[26:27]
                key = (chain, resseq, icode)
                b = res_to_b.get(key, 0.0)
                # PDB B-factor column is 6 chars wide. Clamp to that range.
                b_clamped = max(-99.99, min(999.99, b))
                b_str = f"{b_clamped:6.2f}"
                new_line = line[:60] + b_str + line[66:]
                out.write(new_line)
            else:
                out.write(line)
    return (n_pdb == T), n_pdb, T


CARTOON_GRID_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 16px;
    background: #fafafa;
  }}
  h2 {{ margin: 0 0 4px 0; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 12px; line-height: 1.5; }}
  .legend {{
    font-size: 12px;
    color: #444;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 16px;
    line-height: 1.5;
  }}
  .grid {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .cell {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px;
    width: 360px;
  }}
  .cell-title {{
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 4px;
    color: #222;
    word-break: break-all;
  }}
  .cell-meta {{
    font-size: 11px;
    color: #666;
    margin-bottom: 6px;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }}
  .viewer {{
    width: 350px;
    height: 320px;
    position: relative;
    border: 1px solid #ccc;
    background: #fff;
  }}
</style>
</head><body>
<h2>{title}</h2>
<div class="meta">{meta}</div>
<div class="legend">
  <b>Each residue is colored by the dataset-level discriminative signal at its
  relative-position bin: mean(positive activation) − mean(negative activation).</b><br>
  <b style="color:#c0392b">Red</b> = positives have higher activation than negatives at this residue's bin &nbsp;|&nbsp;
  <b style="color:#2c5fa6">Blue</b> = negatives have higher &nbsp;|&nbsp;
  white ≈ no difference.<br>
  Color scale is shared across all viewers in this file: ±{vmax:.3f}.
  All 5 viewers show the SAME signal painted on different positive-protein folds.
</div>

<div class="grid">
{cells}
</div>

<script>
const proteins = {proteins_json};
function renderAll() {{
  for (const p of proteins) {{
    const viewer = $3Dmol.createViewer(p.viewer_id, {{ backgroundColor: "white" }});
    viewer.addModel(p.pdb_data, "pdb");
    viewer.setStyle({{}}, {{ cartoon: {{
      colorscheme: {{ prop: "b", gradient: "rwb", min: p.vmax, max: p.vmin }}
    }} }});
    viewer.zoomTo();
    viewer.render();
  }}
}}
window.addEventListener("DOMContentLoaded", renderAll);
</script>
</body></html>
"""


def _format_cartoon_cell(p_data: Dict[str, Any]) -> str:
    return (
        f'  <div class="cell">\n'
        f'    <div class="cell-title">{html_escape(p_data["fasta_id"])}</div>\n'
        f'    <div class="cell-meta">{html_escape(p_data["meta"])}</div>\n'
        f'    <div class="viewer" id="{p_data["viewer_id"]}"></div>\n'
        f'  </div>\n'
    )


def html_escape(s: str) -> str:
    """Minimal HTML escape (avoids importing html module just for this)."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _render_one_feature_html(
    *,
    label: str,
    short_label: str,
    method: str,
    rank: int,
    feature_idx: int,
    feature_info: Dict[str, Any],
    diff_per_bin: np.ndarray,        # (n_bins,) signed diff
    canvas_ids: List[str],
    structures_root: Path,
    n_bins: int,
    shared_vmax: float,
    out_path: Path,
) -> None:
    """Emit one HTML file with a row of 5 cartoon viewers for this feature."""
    cells_payload: List[Dict[str, Any]] = []
    cells_html: List[str] = []
    for i, fid in enumerate(canvas_ids):
        pdb_src = find_pdb(fid, structures_root)
        if pdb_src is None:
            print(f"      [warn] no PDB for {fid}, skipping in HTML")
            continue
        # Build per-residue array by mapping each residue to its relative-position bin.
        ca = load_ca_coords(pdb_src)
        if ca is None:
            print(f"      [warn] no Cα for {fid}, skipping in HTML")
            continue
        T = len(ca)
        pos = (np.arange(T) + 0.5) / T
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)
        per_residue = diff_per_bin[bin_idx]

        # Write a temp PDB with B-factors set, then read it back into the HTML.
        tmp_pdb = out_path.parent / f".__tmp_{out_path.stem}_{i}_{safe_filename(fid)}.pdb"
        ok, n_pdb, T_arr = _write_recolored_pdb_for_html(
            src_pdb=pdb_src,
            per_residue_values=per_residue.astype(np.float32, copy=False),
            out_path=tmp_pdb,
        )
        pdb_text = tmp_pdb.read_text(encoding="utf-8")
        # Clean up the temp file. We could keep it but the data is embedded in HTML.
        try:
            tmp_pdb.unlink()
        except Exception:
            pass

        viewer_id = f"viewer_{i}"
        cell = {
            "fasta_id": fid,
            "viewer_id": viewer_id,
            "meta": (
                f"T={T_arr}  PDB residues={n_pdb}  "
                f"{'aligned' if ok else 'partial'}"
            ),
        }
        cells_html.append(_format_cartoon_cell(cell))
        cells_payload.append({
            "viewer_id": viewer_id,
            "pdb_data": pdb_text,
            "vmin": -shared_vmax,
            "vmax": shared_vmax,
        })

    if not cells_payload:
        print(f"      [skip] no viewers could be rendered for feature {feature_idx}")
        return

    method_short = {
        "coef": "by |w/σ|",
        "auroc": "by AUROC",
        "diff": "by |Δμ|",
    }.get(method, method)

    score_str = ""
    if "auroc" in feature_info:
        score_str += f"  AUROC = {feature_info['auroc']:.3f}"
    if "score" in feature_info:
        score_str += f"  selection score = {feature_info['score']:.4f}"

    title = (f"{short_label}  —  feature {feature_idx}  "
             f"({method_short}, rank {rank})")
    meta = (f"Label: {label}  |  Selection: {method_short} rank {rank}  |  "
            f"feature_idx={feature_idx}{score_str}")

    import json as _json
    rendered = CARTOON_GRID_HTML_TEMPLATE.format(
        title=html_escape(title),
        meta=html_escape(meta),
        vmax=shared_vmax,
        cells="".join(cells_html),
        proteins_json=_json.dumps(cells_payload),
    )
    out_path.write_text(rendered, encoding="utf-8")


def plot_protein_canvases_html(
    *,
    label: str,
    short_label: str,
    feature_groups: List[Dict[str, Any]],
    canvas_ids: List[str],
    pos_means: np.ndarray,
    neg_means: np.ndarray,
    structures_root: Path,
    n_bins: int,
    out_dir: Path,
) -> None:
    """Emit one HTML per feature into out_dir, with a shared color scale."""
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_full = pos_means - neg_means

    # Compute shared color scale across all features being rendered.
    selected_diffs = []
    for group in feature_groups:
        for fi_info in group["features"]:
            selected_diffs.append(diff_full[int(fi_info["feature_idx"])])
    if selected_diffs:
        arr = np.stack(selected_diffs, axis=0)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            shared_vmax = float(np.percentile(np.abs(finite), 95))
            if shared_vmax == 0:
                shared_vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
            if shared_vmax == 0:
                shared_vmax = 1.0
        else:
            shared_vmax = 1.0
    else:
        shared_vmax = 1.0
    print(f"  HTML shared color scale: ±{shared_vmax:.3f}")

    for group in feature_groups:
        for rank, fi_info in enumerate(group["features"], start=1):
            fi = int(fi_info["feature_idx"])
            method_short = {"coef": "byCoef", "auroc": "byAUROC", "diff": "byDiff"}.get(
                group["method"], group["method"]
            )
            html_name = f"{safe_filename(short_label)}__{method_short}_rank{rank}_feat{fi}.html"
            html_path = out_dir / html_name
            print(f"    rendering HTML: {html_name}")
            _render_one_feature_html(
                label=label,
                short_label=short_label,
                method=group["method"],
                rank=rank,
                feature_idx=fi,
                feature_info=fi_info,
                diff_per_bin=diff_full[fi],
                canvas_ids=canvas_ids,
                structures_root=structures_root,
                n_bins=n_bins,
                shared_vmax=shared_vmax,
                out_path=html_path,
            )


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    pickle_dir = (Path(args.pickle_dir).resolve() if args.pickle_dir
                  else (root / "model_pkl_files"))
    embeddings_dir = (Path(args.last_hidden_dir).resolve() if args.last_hidden_dir
                      else (root / "BAHD_lastLayer_embeddings"))
    structures_root_arg = (Path(args.structures_dir).resolve() if args.structures_dir
                           else (root / "BAHD_dataset" / "BAHD_structures"))
    structures_root: Optional[Path] = structures_root_arg if structures_root_arg.is_dir() else None
    labels_path = (Path(args.labels_path).resolve() if args.labels_path
                   else (root / "BAHD_dataset" / "labels.pt"))
    interp_dir = (Path(args.interp_dir).resolve() if args.interp_dir
                  else (root / "interpretability_outputs"))
    top_cells_dir = (Path(args.top_cells_dir).resolve() if args.top_cells_dir
                     else (root / "discriminative_features_full" / "per_label_top_cells"))
    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else (root / "figure_3_extended"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paths] pickle_dir:    {pickle_dir}")
    print(f"[paths] embeddings:    {embeddings_dir}")
    print(f"[paths] structures:    {structures_root if structures_root else '(skipped)'}")
    print(f"[paths] labels.pt:     {labels_path}")
    print(f"[paths] interp_dir:    {interp_dir}")
    print(f"[paths] top_cells_dir: {top_cells_dir}")
    print(f"[paths] out_dir:       {out_dir}")

    full_label_list = [f"{args.task_prefix}{lbl}" for lbl in args.label]

    cache = torch.load(labels_path, map_location="cpu", weights_only=False)
    fasta_ids = [str(x) for x in cache["fasta_ids"]]
    all_label_names = [str(x) for x in cache["label_names"]]
    labels_full = cache["labels"].detach().cpu().numpy().astype(np.uint8)

    for label_short, full_label in zip(args.label, full_label_list):
        print("\n" + "=" * 72)
        print(f"=== {full_label}")
        print("=" * 72)
        process_one_label(
            full_label=full_label,
            label_short=label_short,
            fasta_ids=fasta_ids,
            all_label_names=all_label_names,
            labels_full=labels_full,
            pickle_dir=pickle_dir,
            embeddings_dir=embeddings_dir,
            structures_root=structures_root,
            interp_dir=interp_dir,
            top_cells_dir=top_cells_dir,
            out_dir=out_dir,
            n_top_features=int(args.n_top_features),
            n_relpos_bins=int(args.n_relpos_bins),
            root=root,
            canvas_format=str(args.canvas_format),
        )

    print("\nDone.")


def process_one_label(
    *,
    full_label: str,
    label_short: str,
    fasta_ids: List[str],
    all_label_names: List[str],
    labels_full: np.ndarray,
    pickle_dir: Path,
    embeddings_dir: Path,
    structures_root: Optional[Path],
    interp_dir: Path,
    top_cells_dir: Path,
    out_dir: Path,
    n_top_features: int,
    n_relpos_bins: int,
    root: Path,
    canvas_format: str = "png",
) -> None:
    if full_label not in all_label_names:
        print(f"  [skip] label not found in labels.pt: {full_label}")
        return
    label_idx = all_label_names.index(full_label)
    truth = labels_full[:, label_idx]
    n_pos = int(truth.sum())
    n_neg = int(len(truth) - n_pos)
    print(f"[label] {full_label}  n_pos={n_pos}  n_neg={n_neg}")

    is_acceptor = full_label.startswith("acceptor_superclass::")
    pickle_filename = f"hidden_layer_lr_bahd_{'acceptor' if is_acceptor else 'donor'}_models.pkl"
    pkl_path = pickle_dir / pickle_filename
    if not pkl_path.is_file():
        candidates = list(pickle_dir.glob(f"*hidden*lr*{'acceptor' if is_acceptor else 'donor'}*.pkl"))
        if candidates:
            pkl_path = candidates[0]
        else:
            print(f"  [skip] no logistic regression pickle found in {pickle_dir}")
            return
    print(f"[pickle] {pkl_path}")

    mu, sigma, w = load_pipeline_for_label(pkl_path, full_label)
    w_over_sigma = w / sigma
    abs_w_raw = np.abs(w_over_sigma)
    D = len(w_over_sigma)

    # ---- Method 1: top features by |w / sigma|
    print(f"\n[method 1] picking top {n_top_features} features by |w / sigma| "
          f"(model coefficient magnitude)")
    by_coef = np.argsort(abs_w_raw)[::-1][:n_top_features]
    method1_features = [
        {"feature_idx": int(fi), "score": float(abs_w_raw[fi])}
        for fi in by_coef
    ]
    for k, info in enumerate(method1_features, start=1):
        print(f"  {k}. feat {info['feature_idx']:5d}  |w/sigma| = {info['score']:.4f}")

    # ---- Method 2: top features by max single-cell AUROC (from discriminative TSV)
    print(f"\n[method 2] picking top {n_top_features} features by max single-cell AUROC "
          f"(discriminative TSV)")
    tsv_path = top_cells_dir / f"{safe_filename(full_label)}__top_cells.tsv"
    method2_features: List[Dict[str, Any]] = []
    if tsv_path.is_file():
        import csv as _csv
        with tsv_path.open("r", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh, delimiter="\t"))
        best_per_feature: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            fi = int(r["feature_idx"])
            cur = best_per_feature.get(fi)
            if cur is None or float(r["auroc_dev"]) > float(cur["auroc_dev"]):
                best_per_feature[fi] = r
        sorted_feats = sorted(best_per_feature.values(),
                              key=lambda r: -float(r["auroc_dev"]))
        for r in sorted_feats[:n_top_features]:
            method2_features.append({
                "feature_idx": int(r["feature_idx"]),
                "auroc": float(r["auroc"]),
                "best_bin_relpos_lo": float(r["relpos_lo"]),
                "best_bin_relpos_hi": float(r["relpos_hi"]),
                "score": float(r["auroc_dev"]),
            })
        for k, info in enumerate(method2_features, start=1):
            print(f"  {k}. feat {info['feature_idx']:5d}  AUROC={info['auroc']:.3f}  "
                  f"best_bin_relpos=[{info['best_bin_relpos_lo']:.2f},"
                  f"{info['best_bin_relpos_hi']:.2f}]")
    else:
        print(f"  [warn] discriminative TSV not found at {tsv_path}")
        print(f"  [warn] method 2 will be skipped")

    # ---- Method 3: top features by max |bin-level mean(pos) - mean(neg)|
    print(f"\n[method 3] computing per-feature, per-bin pos/neg standardized means "
          f"across whole dataset...")
    print("           (this is the slow part — reads all 366 per-residue tensors)")
    pos_means, neg_means, n_pos_used, n_neg_used = compute_full_pos_neg_means(
        fasta_ids=fasta_ids, truth=truth,
        mu=mu, sigma=sigma,
        embeddings_dir=embeddings_dir, structures_root=structures_root,
        n_bins=n_relpos_bins,
    )
    print(f"           used {n_pos_used} positives and {n_neg_used} negatives")
    diff = pos_means - neg_means
    abs_diff = np.abs(diff)
    abs_diff_safe = np.where(np.isnan(abs_diff), -np.inf, abs_diff)
    max_diff_per_feature = np.max(abs_diff_safe, axis=1)
    argmax_bin = np.argmax(abs_diff_safe, axis=1)
    by_diff = np.argsort(max_diff_per_feature)[::-1][:n_top_features]
    method3_features: List[Dict[str, Any]] = []
    for fi in by_diff:
        method3_features.append({
            "feature_idx": int(fi),
            "score": float(max_diff_per_feature[fi]),
            "best_bin_idx": int(argmax_bin[fi]),
            "best_bin_relpos_lo": float(argmax_bin[fi]) / n_relpos_bins,
            "best_bin_relpos_hi": float(argmax_bin[fi] + 1) / n_relpos_bins,
        })
    for k, info in enumerate(method3_features, start=1):
        print(f"  {k}. feat {info['feature_idx']:5d}  "
              f"max|pos-neg|_per_bin = {info['score']:.3f}  "
              f"at bin {info['best_bin_idx']} (relpos "
              f"[{info['best_bin_relpos_lo']:.2f},{info['best_bin_relpos_hi']:.2f}])")

    # ---- Sort proteins
    scores_by_id = load_per_protein_scores(
        full_label, interp_dir,
        interp_subdir=("bahd_acceptor_lasthidden" if is_acceptor else "bahd_donor_lasthidden"),
    )
    print(f"\n[scores] loaded model scores for {len(scores_by_id)} proteins")

    pos_ids = sorted(
        [fid for fid, t in zip(fasta_ids, truth) if t == 1],
        key=lambda fid: -scores_by_id.get(fid, 0.0),
    )
    neg_ids = sorted(
        [fid for fid, t in zip(fasta_ids, truth) if t == 0],
        key=lambda fid: -scores_by_id.get(fid, 0.0),
    )
    print(f"[order] positives ({len(pos_ids)}) over negatives ({len(neg_ids)}); "
          "each section descending by model score")

    # ---- Build feature_groups
    feature_groups: List[Dict[str, Any]] = []
    feature_groups.append({
        "method": "coef",
        "method_label": "By |w/σ|\n(model coefficient magnitude)",
        "features": method1_features,
        "score_label_fn": lambda info: f"|w/σ| = {info['score']:.4f}",
    })
    if method2_features:
        feature_groups.append({
            "method": "auroc",
            "method_label": "By max single-cell AUROC\n(discriminative analysis)",
            "features": method2_features,
            "score_label_fn": lambda info: (
                f"AUROC = {info['auroc']:.3f}  "
                f"best bin: relpos [{info['best_bin_relpos_lo']:.2f},"
                f"{info['best_bin_relpos_hi']:.2f}]"
            ),
        })
    feature_groups.append({
        "method": "diff",
        "method_label": "By max |bin-mean(pos) − bin-mean(neg)|\n(direct activation difference)",
        "features": method3_features,
        "score_label_fn": lambda info: (
            f"max |pos-neg| = {info['score']:.3f}  "
            f"at bin {info['best_bin_idx']} (relpos "
            f"[{info['best_bin_relpos_lo']:.2f},{info['best_bin_relpos_hi']:.2f}])"
        ),
    })

    # ---- First figure: 3x3 heatmap grid
    out_path = out_dir / f"figure_3_extended_grid__{safe_filename(label_short)}.png"
    plot_extended_heatmap_grid(
        label=full_label,
        short_label=label_short,
        feature_groups=feature_groups,
        fasta_order_pos=pos_ids,
        fasta_order_neg=neg_ids,
        embeddings_dir=embeddings_dir,
        structures_root=structures_root,
        mu=mu, sigma=sigma,
        w_over_sigma=w_over_sigma,
        n_bins=n_relpos_bins,
        out_path=out_path,
    )

    # ---- Second figure(s): per-residue (pos - neg) signed diff projected onto
    # 5 positive protein canvases for each of the 9 selected features. Output
    # format is controlled by canvas_format.
    if structures_root is not None:
        n_canvases = 5
        canvas_ids = pos_ids[:n_canvases]
        print(f"\n[canvas grid] using {len(canvas_ids)} positive proteins as canvases:")
        for fid in canvas_ids:
            print(f"  {fid}")

        if canvas_format in ("png", "both"):
            canvas_out_path = out_dir / f"figure_3_extended_canvases__{safe_filename(label_short)}.png"
            plot_protein_canvas_grid(
                label=full_label,
                short_label=label_short,
                feature_groups=feature_groups,
                canvas_ids=canvas_ids,
                pos_means=pos_means,
                neg_means=neg_means,
                structures_root=structures_root,
                n_bins=n_relpos_bins,
                out_path=canvas_out_path,
            )

        if canvas_format in ("html", "both"):
            html_dir = out_dir / "canvases_html" / safe_filename(label_short)
            print(f"\n[canvas html] writing 9 HTML files to {html_dir}/")
            plot_protein_canvases_html(
                label=full_label,
                short_label=label_short,
                feature_groups=feature_groups,
                canvas_ids=canvas_ids,
                pos_means=pos_means,
                neg_means=neg_means,
                structures_root=structures_root,
                n_bins=n_relpos_bins,
                out_dir=html_dir,
            )
    else:
        print("\n[canvas grid] structures dir not available; skipping protein canvas figure")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
