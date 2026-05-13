#!/usr/bin/env python3
"""
final_analysis_plots.py

Generates the new figures needed for the writeup that aren't produced by the
existing pipeline:

  1. figure_lengths_by_label.png
       Boxplot per label: distribution of protein residue lengths, within-label
       (positive) vs out-of-label (negative). Diagnostic for whether observed
       discriminative signal at certain relative-position bins might be confounded
       by systematic length differences between subclasses.

  2. figure_diffuseness_supplementary.png
       Two-panel supplement to writeup_figures Figure 2:
         (A) Binned-residue version of the top1_share histogram. Same statistic
             but computed over 50 relative-position bins instead of individual
             residues, to show that even when contributions are aggregated to
             coarser regions, the top region carries only a small fraction.
         (B) Per-protein Lorenz curve of FEATURE contributions for a representative
             well-predicted positive protein, showing that no small subset of the
             1536 features dominates that protein's prediction either.

  3. figure_roc_top5__phenolic_acids.png
  4. figure_roc_top5__naphthalenes.png
       ROC curves (TPR vs FPR) for the top 5 single-cell features per label,
       overlaid in one panel. Each cell is treated as a one-feature classifier:
       use its activation value to rank proteins, then compute the ROC curve
       against the binary label.

  5. figure_roc_per_label.png
       One ROC curve per label (acceptor + donor with positive_count >= 10),
       each using that label's BEST single-cell feature. Colors are coordinated
       with the top-5 plots above so the same label uses the same hue across
       figures.

Usage:
    python final_analysis_plots.py \
        --root_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability \
        --metrics_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability/5fold_cv_ll_lr_preds
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True)
    p.add_argument("--metrics_dir", default=None,
                   help="Default: <root>/5fold_cv_ll_lr_preds")
    p.add_argument("--last_hidden_dir", default=None,
                   help="Default: <root>/BAHD_lastLayer_embeddings")
    p.add_argument("--structures_dir", default=None,
                   help="Default: <root>/BAHD_dataset/BAHD_structures")
    p.add_argument("--labels_path", default=None,
                   help="Default: <root>/BAHD_dataset/labels.pt")
    p.add_argument("--interp_dir", default=None,
                   help="Default: <root>/interpretability_outputs")
    p.add_argument("--top_cells_dir", default=None,
                   help="Default: <root>/discriminative_features_full/per_label_top_cells")
    p.add_argument("--out_dir", default=None,
                   help="Default: <root>/final_analysis_plots")
    p.add_argument("--min_positive_count", type=int, default=10,
                   help="Minimum positives for inclusion in length boxplot and per-label ROC.")
    p.add_argument("--n_top_features_per_label", type=int, default=5,
                   help="Number of top single-cell features for the top-5 ROC plots.")
    p.add_argument("--showcase_protein_for_lorenz", default=None,
                   help="Optional fasta_id to use for Figure 2 panel B. Defaults to "
                        "highest-scoring Phenolic acids positive.")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Loaders mirroring the rest of the pipeline
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


def safe_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


# --------------------------------------------------------------------------------------
# Figure 1: length-by-label boxplot
# --------------------------------------------------------------------------------------

def fig_length_by_label(
    labels_path: Path,
    embeddings_dir: Path,
    structures_root: Optional[Path],
    metrics_dir: Path,
    min_positive_count: int,
    out_path: Path,
) -> None:
    """Boxplot per label: within-label vs out-of-label residue length distribution."""
    print(f"\n[fig_length] computing protein lengths...")
    cache = torch.load(labels_path, map_location="cpu", weights_only=False)
    fasta_ids: List[str] = [str(x) for x in cache["fasta_ids"]]
    label_names: List[str] = [str(x) for x in cache["label_names"]]
    Y = cache["labels"].detach().cpu().numpy().astype(np.uint8)

    # Compute T (residue count) per protein from the per-residue tensors.
    T_per_protein: Dict[str, int] = {}
    for i, fid in enumerate(fasta_ids):
        H = load_real_residue_tensor(fid, embeddings_dir, structures_root)
        if H is not None:
            T_per_protein[fid] = int(H.shape[0])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(fasta_ids)}")
    print(f"  loaded T for {len(T_per_protein)}/{len(fasta_ids)} proteins")
    Ts = np.array([T_per_protein.get(fid, np.nan) for fid in fasta_ids], dtype=np.float64)

    # Eligible labels: positive_count >= min_positive_count for any well-defined label.
    eligible: List[Dict[str, Any]] = []
    for prefix in ("acceptor_superclass::", "donor_type::"):
        idxs = [i for i, n in enumerate(label_names) if n.startswith(prefix)]
        for i in idxs:
            n_pos = int(Y[:, i].sum())
            if n_pos >= min_positive_count:
                short = label_names[i].removeprefix(prefix)
                eligible.append({
                    "full": label_names[i],
                    "short": short,
                    "label_idx": i,
                    "n_pos": n_pos,
                    "prefix": prefix,
                })
    eligible.sort(key=lambda r: -r["n_pos"])
    print(f"  eligible labels (n_pos >= {min_positive_count}): {len(eligible)}")

    # Compose paired box-plot data.
    fig_h = max(5.0, 0.42 * len(eligible) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y_positions = np.arange(len(eligible))
    box_offset = 0.18
    pos_color = "#c0392b"
    neg_color = "#7f8c8d"

    for j, info in enumerate(eligible):
        truth = Y[:, info["label_idx"]]
        pos_lengths = Ts[(truth == 1) & np.isfinite(Ts)]
        neg_lengths = Ts[(truth == 0) & np.isfinite(Ts)]
        ax.boxplot(
            pos_lengths,
            positions=[y_positions[j] + box_offset],
            vert=False, widths=0.32, patch_artist=True,
            boxprops=dict(facecolor=pos_color, alpha=0.55, edgecolor=pos_color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=pos_color),
            capprops=dict(color=pos_color),
            flierprops=dict(marker="o", markersize=2.5, markeredgecolor="none",
                            markerfacecolor=pos_color, alpha=0.5),
        )
        ax.boxplot(
            neg_lengths,
            positions=[y_positions[j] - box_offset],
            vert=False, widths=0.32, patch_artist=True,
            boxprops=dict(facecolor=neg_color, alpha=0.45, edgecolor=neg_color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=neg_color),
            capprops=dict(color=neg_color),
            flierprops=dict(marker="o", markersize=2.5, markeredgecolor="none",
                            markerfacecolor=neg_color, alpha=0.4),
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{info['short']}\n(n_pos={info['n_pos']})" for info in eligible],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Protein length (number of residues)", fontsize=11)
    ax.set_title(
        "Protein length distribution by label: within-label (positive, red) vs "
        "out-of-label (negative, gray)\n"
        "Within-label medians close to the global median across all labels would "
        "indicate length is not a confound",
        fontsize=10,
    )

    # Reference: global median across the dataset.
    global_median = float(np.nanmedian(Ts))
    ax.axvline(global_median, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(global_median, -0.5,
            f"  global median = {global_median:.0f} residues",
            fontsize=8, va="bottom", color="black")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=pos_color, alpha=0.55, edgecolor=pos_color),
        plt.Rectangle((0, 0), 1, 1, facecolor=neg_color, alpha=0.45, edgecolor=neg_color),
    ]
    ax.legend(handles, ["positives (within label)", "negatives (out of label)"],
              loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figure 2 supplementary: binned top1_share + per-protein Lorenz over features
# --------------------------------------------------------------------------------------

def _try_load_per_protein_bundle(per_protein_dir: Path, fasta_id: str):
    f = per_protein_dir / f"{fasta_id}.pt"
    if not f.is_file():
        cands = list(per_protein_dir.glob(f"*{fasta_id}*.pt"))
        if not cands:
            return None
        f = cands[0]
    return torch.load(f, map_location="cpu", weights_only=False)


def fig_diffuseness_supplementary(
    labels_path: Path,
    interp_dir: Path,
    embeddings_dir: Path,
    structures_root: Optional[Path],
    showcase_protein_id: Optional[str],
    out_path: Path,
) -> None:
    """Two-panel supplement: binned top1_share histogram + per-protein feature Lorenz."""
    print(f"\n[fig_diffuseness] building two-panel supplementary figure...")

    # ---- Panel A: binned top1_share across (protein, label) pairs ----
    # Bin per-residue contributions into 50 bins; compute fraction of |contribution|
    # held by the largest bin.
    n_bins = 50
    interp_subdir = interp_dir / "bahd_acceptor_lasthidden" / "per_protein"
    if not interp_subdir.is_dir():
        # Try alternative subdir names that may have been used.
        candidates = list(interp_dir.glob("*acceptor*lasthidden*/per_protein"))
        if candidates:
            interp_subdir = candidates[0]
        else:
            print("  [warn] no per_protein bundles found; panel A will be empty")
            interp_subdir = None

    binned_top1_shares: List[float] = []
    if interp_subdir is not None:
        bundle_files = sorted(interp_subdir.glob("*.pt"))
        print(f"  reading {len(bundle_files)} per-protein bundles for panel A...")
        for f in bundle_files:
            try:
                b = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            res_contribs = b.get("residue_contribs")
            if res_contribs is None:
                # Fall back to alternate key names just in case.
                res_contribs = b.get("per_residue_contributions")
            if res_contribs is None:
                continue
            if hasattr(res_contribs, "numpy"):
                res_contribs = res_contribs.numpy()
            res_contribs = np.asarray(res_contribs)  # (n_labels, T)
            n_lbls, T = res_contribs.shape
            pos = (np.arange(T) + 0.5) / T
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_idx = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)
            for li in range(n_lbls):
                contrib = res_contribs[li]
                if not np.isfinite(contrib).all() or T == 0:
                    continue
                # Sum of |contributions| per bin.
                abs_c = np.abs(contrib)
                bin_sums = np.zeros(n_bins, dtype=np.float64)
                np.add.at(bin_sums, bin_idx, abs_c)
                total = bin_sums.sum()
                if total <= 0:
                    continue
                top1 = float(bin_sums.max() / total)
                binned_top1_shares.append(top1)
        print(f"  collected {len(binned_top1_shares)} (protein, label) pairs")

    # ---- Panel B: per-feature contribution Lorenz curve for a single protein ----
    print(f"  building panel B (per-feature contribution Lorenz)...")
    cache = torch.load(labels_path, map_location="cpu", weights_only=False)
    fasta_ids: List[str] = [str(x) for x in cache["fasta_ids"]]
    label_names: List[str] = [str(x) for x in cache["label_names"]]
    Y = cache["labels"].detach().cpu().numpy().astype(np.uint8)

    # Pick a default showcase protein if none given: the one with highest score
    # on Phenolic acids in its bundle.
    chosen_id = showcase_protein_id
    chosen_label_idx = None
    target_label = "acceptor_superclass::Phenolic acids (C6-C1)"
    if chosen_id is None and interp_subdir is not None and target_label in label_names:
        target_li = label_names.index(target_label)
        best_score = -np.inf
        for f in interp_subdir.glob("*.pt"):
            try:
                b = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            names = list(b.get("label_names", []))
            if target_label not in names:
                continue
            li = names.index(target_label)
            scores = b.get("scores")
            if hasattr(scores, "numpy"):
                scores = scores.numpy()
            if scores is None:
                continue
            s = float(scores[li])
            if s > best_score and Y[fasta_ids.index(str(b["fasta_id"])), target_li] == 1:
                best_score = s
                chosen_id = str(b["fasta_id"])
                chosen_label_idx = li
    feature_contribs: Optional[np.ndarray] = None
    chosen_label_name = ""
    if chosen_id is not None and interp_subdir is not None:
        b = _try_load_per_protein_bundle(interp_subdir, chosen_id)
        if b is not None:
            names = list(b.get("label_names", []))
            li = (chosen_label_idx if chosen_label_idx is not None else
                  (names.index(target_label) if target_label in names else 0))
            chosen_label_name = names[li] if names else ""
            pf = b.get("feature_contribs")
            if pf is None:
                pf = b.get("per_feature_contributions")
            if pf is not None:
                if hasattr(pf, "numpy"):
                    pf = pf.numpy()
                feature_contribs = np.asarray(pf[li], dtype=np.float64)
                print(f"    using protein {chosen_id} for label {chosen_label_name} "
                      f"(per_feature_contributions shape {feature_contribs.shape})")

    # ---- Render the two-panel figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel A
    ax = axes[0]
    if binned_top1_shares:
        ax.hist(binned_top1_shares, bins=40, color="#3a7ab8", edgecolor="white")
        med = float(np.median(binned_top1_shares))
        ax.axvline(med, color="black", linestyle="--", linewidth=1.0,
                   label=f"median = {med:.3f}")
        baseline = 1.0 / n_bins
        ax.axvline(baseline, color="#c0392b", linestyle=":", linewidth=1.0,
                   label=f"uniform baseline = 1/{n_bins} = {baseline:.3f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "no per-protein bundles found",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
    ax.set_xlabel("Top-1 bin's share of |residue contribution|", fontsize=10)
    ax.set_ylabel("Count of (protein, label) pairs", fontsize=10)
    ax.set_title(
        "(A) Binned diffuseness: even with 50 relative-position bins,\n"
        "the top bin carries only a small fraction of total |contribution|",
        fontsize=10,
    )

    # Panel B: Lorenz curve over feature contributions for one protein.
    ax = axes[1]
    if feature_contribs is not None:
        abs_c = np.abs(feature_contribs)
        sorted_desc = np.sort(abs_c)[::-1]
        cumsum = np.cumsum(sorted_desc)
        if cumsum[-1] > 0:
            cum_share = cumsum / cumsum[-1]
            x = np.arange(1, len(sorted_desc) + 1) / len(sorted_desc)
            ax.plot(x, cum_share, color="#3a7ab8", linewidth=2.0)
            ax.plot([0, 1], [0, 1], color="#888", linestyle=":", linewidth=1.0,
                    label="uniform reference (perfectly diffuse)")
            # Annotate "top 1% of features carry X% of |contribution|"
            for frac in (0.01, 0.05, 0.10):
                k = max(1, int(frac * len(sorted_desc)))
                share = cum_share[k - 1]
                ax.scatter([frac], [share], color="#c0392b", zorder=5, s=22)
                ax.annotate(
                    f"top {frac*100:.0f}% → {share:.1%} of |contrib|",
                    xy=(frac, share), xytext=(frac + 0.04, share - 0.06),
                    fontsize=8, color="#c0392b",
                    arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.6),
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Fraction of features (sorted by |contribution|, descending)",
                          fontsize=10)
            ax.set_ylabel("Cumulative share of |contribution|", fontsize=10)
            ax.legend(loc="lower right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no nonzero contributions",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
    else:
        ax.text(0.5, 0.5, "per-feature contributions unavailable",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
    ax.set_title(
        f"(B) Per-feature contribution Lorenz curve for one well-predicted protein\n"
        f"({chosen_id or 'n/a'} on {chosen_label_name or 'n/a'})",
        fontsize=10,
    )
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Diffuseness of the linear model's interpretation, at two aggregation levels",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figures 3 & 4: top-5-feature ROC curves for two showcase labels
# --------------------------------------------------------------------------------------

def _bin_feature_to_bin(values_t: np.ndarray, n_bins: int, bin_idx: int) -> float:
    """Mean activation for one specific bin, given a (T,) array."""
    T = len(values_t)
    if T == 0:
        return float("nan")
    pos = (np.arange(T) + 0.5) / T
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    res_bin = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)
    mask = res_bin == bin_idx
    if not mask.any():
        return float("nan")
    return float(values_t[mask].mean())


def _load_top_cells_for_label(
    top_cells_dir: Path, full_label: str, n_top: int,
) -> List[Dict[str, Any]]:
    """Read a discriminative TSV, return up to n_top distinct features by best cell."""
    tsv = top_cells_dir / f"{safe_filename(full_label)}__top_cells.tsv"
    if not tsv.is_file():
        return []
    with tsv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    rows.sort(key=lambda r: -float(r["auroc_dev"]))
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for r in rows:
        fi = int(r["feature_idx"])
        if fi in seen:
            continue
        seen.add(fi)
        out.append({
            "feature_idx": fi,
            "bin_idx": int(r["bin_idx"]),
            "auroc": float(r["auroc"]),
            "auroc_dev": float(r["auroc_dev"]),
            "p_emp": float(r["p_emp"]),
        })
        if len(out) >= n_top:
            break
    return out


def _compute_cell_score_per_protein(
    fasta_ids: List[str],
    feature_idx: int,
    bin_idx: int,
    n_bins: int,
    embeddings_dir: Path,
    structures_root: Optional[Path],
) -> np.ndarray:
    """Per-protein score = mean activation of feature_idx within bin_idx."""
    out = np.full(len(fasta_ids), np.nan, dtype=np.float64)
    for i, fid in enumerate(fasta_ids):
        H = load_real_residue_tensor(fid, embeddings_dir, structures_root)
        if H is None or feature_idx >= H.shape[1]:
            continue
        out[i] = _bin_feature_to_bin(H[:, feature_idx], n_bins, bin_idx)
    return out


def _fig_roc_top_n_for_label(
    *,
    label_short: str,
    full_label: str,
    fasta_ids: List[str],
    label_names: List[str],
    Y: np.ndarray,
    top_cells_dir: Path,
    embeddings_dir: Path,
    structures_root: Optional[Path],
    n_top: int,
    n_bins: int,
    palette: List[str],
    out_path: Path,
) -> Optional[List[Dict[str, Any]]]:
    """
    Produce one ROC plot with up to n_top features overlaid for the given label.
    Returns the list of features used (with computed AUROC etc.) so the caller can
    assemble cross-figure colour mappings if desired. Returns None on failure.
    """
    if full_label not in label_names:
        print(f"  [skip] {full_label} not found in labels.pt")
        return None
    li = label_names.index(full_label)
    truth = Y[:, li]

    top_features = _load_top_cells_for_label(top_cells_dir, full_label, n_top)
    if not top_features:
        print(f"  [skip] no top cells found for {full_label}")
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    used: List[Dict[str, Any]] = []
    for k, fi_info in enumerate(top_features):
        fi = int(fi_info["feature_idx"])
        bi = int(fi_info["bin_idx"])
        scores = _compute_cell_score_per_protein(
            fasta_ids=fasta_ids,
            feature_idx=fi, bin_idx=bi, n_bins=n_bins,
            embeddings_dir=embeddings_dir, structures_root=structures_root,
        )
        valid = np.isfinite(scores)
        if valid.sum() < 5:
            continue
        # AUROC > 0.5 means high score → positive; AUROC < 0.5 means we should
        # flip the score (negate) to make the ROC curve correspond to a positive-
        # discriminating classifier. Otherwise the curve will be below the diagonal,
        # which is the same information mirrored — but visually less clean.
        s = scores[valid].copy()
        y = truth[valid].astype(int)
        try:
            auroc_check = float(roc_auc_score(y, s))
        except Exception:
            continue
        if auroc_check < 0.5:
            s = -s
            auroc_check = 1.0 - auroc_check
        fpr, tpr, _ = roc_curve(y, s)
        color = palette[k % len(palette)]
        ax.plot(
            fpr, tpr, color=color, linewidth=1.8, alpha=0.92,
            label=(f"feat {fi} bin {bi} (relpos {bi/n_bins:.2f}\u2013"
                   f"{(bi+1)/n_bins:.2f})  AUROC={auroc_check:.3f}"),
        )
        used.append({
            "feature_idx": fi, "bin_idx": bi, "auroc": auroc_check,
            "color": color, "rank": k + 1,
        })

    ax.plot([0, 1], [0, 1], color="#888", linestyle=":", linewidth=0.9,
            label="chance (AUROC = 0.50)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title(
        f"ROC curves for top {len(used)} single-cell features for {label_short}\n"
        f"Each cell uses one ESM3 feature × one relative-position bin as a "
        f"univariate classifier",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return used


# --------------------------------------------------------------------------------------
# Figure 5: per-label best single-cell feature ROC overlay
# --------------------------------------------------------------------------------------

def fig_roc_per_label(
    *,
    labels_path: Path,
    metrics_dir: Path,
    top_cells_dir: Path,
    embeddings_dir: Path,
    structures_root: Optional[Path],
    min_positive_count: int,
    color_overrides: Dict[str, str],   # short_label -> color (must match top-5 plots)
    n_bins: int,
    out_path: Path,
) -> None:
    print(f"\n[fig_per_label_roc] building per-label ROC overlay...")
    cache = torch.load(labels_path, map_location="cpu", weights_only=False)
    fasta_ids: List[str] = [str(x) for x in cache["fasta_ids"]]
    label_names: List[str] = [str(x) for x in cache["label_names"]]
    Y = cache["labels"].detach().cpu().numpy().astype(np.uint8)

    eligible: List[Dict[str, Any]] = []
    for prefix in ("acceptor_superclass::", "donor_type::"):
        for i, n in enumerate(label_names):
            if not n.startswith(prefix):
                continue
            n_pos = int(Y[:, i].sum())
            if n_pos >= min_positive_count:
                eligible.append({
                    "full": n,
                    "short": n.removeprefix(prefix),
                    "label_idx": i,
                    "n_pos": n_pos,
                })
    eligible.sort(key=lambda r: -r["n_pos"])
    print(f"  {len(eligible)} labels with n_pos >= {min_positive_count}")

    # Use a categorical palette wide enough for all labels. tab20 has 20 colors.
    base_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    base_colors = [matplotlib.colors.to_hex(c) for c in base_palette]

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    used_curves: List[Dict[str, Any]] = []
    for j, info in enumerate(eligible):
        full = info["full"]
        short = info["short"]
        top_features = _load_top_cells_for_label(top_cells_dir, full, 1)
        if not top_features:
            print(f"    [skip] no top cells for {full}")
            continue
        fi_info = top_features[0]
        fi = int(fi_info["feature_idx"])
        bi = int(fi_info["bin_idx"])
        truth = Y[:, info["label_idx"]]
        scores = _compute_cell_score_per_protein(
            fasta_ids=fasta_ids,
            feature_idx=fi, bin_idx=bi, n_bins=n_bins,
            embeddings_dir=embeddings_dir, structures_root=structures_root,
        )
        valid = np.isfinite(scores)
        if valid.sum() < 5:
            continue
        s = scores[valid].copy()
        y = truth[valid].astype(int)
        try:
            auroc = float(roc_auc_score(y, s))
        except Exception:
            continue
        if auroc < 0.5:
            s = -s; auroc = 1.0 - auroc
        fpr, tpr, _ = roc_curve(y, s)

        # Use override color if provided, else round-robin from base palette.
        color = color_overrides.get(short, base_colors[j % len(base_colors)])
        ax.plot(
            fpr, tpr, color=color, linewidth=1.4, alpha=0.92,
            label=f"{short} (n_pos={info['n_pos']}, AUROC={auroc:.2f})",
        )
        used_curves.append({"label": short, "color": color, "auroc": auroc})

    ax.plot([0, 1], [0, 1], color="#888", linestyle=":", linewidth=0.9, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title(
        f"ROC curves for the BEST single-cell feature, per label "
        f"(n_pos >= {min_positive_count})\n"
        f"Each curve uses one (ESM3 feature × relative-position bin) as a "
        f"univariate classifier on the dataset",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    metrics_dir = (Path(args.metrics_dir).resolve() if args.metrics_dir
                   else (root / "5fold_cv_ll_lr_preds"))
    embeddings_dir = (Path(args.last_hidden_dir).resolve() if args.last_hidden_dir
                      else (root / "BAHD_lastLayer_embeddings"))
    structures_root_arg = (Path(args.structures_dir).resolve() if args.structures_dir
                           else (root / "BAHD_dataset" / "BAHD_structures"))
    structures_root = structures_root_arg if structures_root_arg.is_dir() else None
    labels_path = (Path(args.labels_path).resolve() if args.labels_path
                   else (root / "BAHD_dataset" / "labels.pt"))
    interp_dir = (Path(args.interp_dir).resolve() if args.interp_dir
                  else (root / "interpretability_outputs"))
    top_cells_dir = (Path(args.top_cells_dir).resolve() if args.top_cells_dir
                     else (root / "discriminative_features_full" / "per_label_top_cells"))
    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else (root / "final_analysis_plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paths] embeddings:    {embeddings_dir}")
    print(f"[paths] structures:    {structures_root if structures_root else '(skipped)'}")
    print(f"[paths] labels.pt:     {labels_path}")
    print(f"[paths] interp_dir:    {interp_dir}")
    print(f"[paths] top_cells_dir: {top_cells_dir}")
    print(f"[paths] out_dir:       {out_dir}")

    # ---- Figure: length-by-label boxplot ----
    fig_length_by_label(
        labels_path=labels_path,
        embeddings_dir=embeddings_dir,
        structures_root=structures_root,
        metrics_dir=metrics_dir,
        min_positive_count=int(args.min_positive_count),
        out_path=out_dir / "figure_lengths_by_label.png",
    )

    # ---- Figure: diffuseness supplementary ----
    fig_diffuseness_supplementary(
        labels_path=labels_path,
        interp_dir=interp_dir,
        embeddings_dir=embeddings_dir,
        structures_root=structures_root,
        showcase_protein_id=args.showcase_protein_for_lorenz,
        out_path=out_dir / "figure_diffuseness_supplementary.png",
    )

    # ---- Figures: ROC top-5 for two showcase labels ----
    cache = torch.load(labels_path, map_location="cpu", weights_only=False)
    fasta_ids: List[str] = [str(x) for x in cache["fasta_ids"]]
    label_names: List[str] = [str(x) for x in cache["label_names"]]
    Y = cache["labels"].detach().cpu().numpy().astype(np.uint8)

    showcase_palette = ["#c0392b", "#2c5fa6", "#27ae60", "#8e44ad", "#e67e22"]
    n_bins = 50

    print(f"\n[roc_top5] producing top-{int(args.n_top_features_per_label)} ROC plots...")
    used_phenolic = _fig_roc_top_n_for_label(
        label_short="Phenolic acids (C6-C1)",
        full_label="acceptor_superclass::Phenolic acids (C6-C1)",
        fasta_ids=fasta_ids, label_names=label_names, Y=Y,
        top_cells_dir=top_cells_dir,
        embeddings_dir=embeddings_dir, structures_root=structures_root,
        n_top=int(args.n_top_features_per_label),
        n_bins=n_bins, palette=showcase_palette,
        out_path=out_dir / "figure_roc_top5__phenolic_acids.png",
    )
    used_naphth = _fig_roc_top_n_for_label(
        label_short="Naphthalenes",
        full_label="acceptor_superclass::Naphthalenes",
        fasta_ids=fasta_ids, label_names=label_names, Y=Y,
        top_cells_dir=top_cells_dir,
        embeddings_dir=embeddings_dir, structures_root=structures_root,
        n_top=int(args.n_top_features_per_label),
        n_bins=n_bins, palette=showcase_palette,
        out_path=out_dir / "figure_roc_top5__naphthalenes.png",
    )

    # The two showcase labels' rank-1 colors are the SAME (palette[0]) by construction.
    # In the per-label overlay, give Phenolic acids the rank-1 color from the Phenolic
    # plot, and Naphthalenes the rank-1 color from the Naphthalenes plot. Since both
    # are palette[0] by default they'd collide, so reassign the second to palette[1].
    color_overrides: Dict[str, str] = {}
    if used_phenolic:
        color_overrides["Phenolic acids (C6-C1)"] = used_phenolic[0]["color"]
    if used_naphth:
        color_overrides["Naphthalenes"] = (
            used_naphth[1]["color"] if len(used_naphth) > 1 else used_naphth[0]["color"]
        )

    # ---- Figure: per-label ROC ----
    fig_roc_per_label(
        labels_path=labels_path,
        metrics_dir=metrics_dir,
        top_cells_dir=top_cells_dir,
        embeddings_dir=embeddings_dir,
        structures_root=structures_root,
        min_positive_count=int(args.min_positive_count),
        color_overrides=color_overrides,
        n_bins=n_bins,
        out_path=out_dir / "figure_roc_per_label.png",
    )

    print(f"\nAll plots written under {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
