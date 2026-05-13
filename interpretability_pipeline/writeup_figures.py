#!/usr/bin/env python3
"""
writeup_figures.py

Build a polished set of writeup figures (1-5) summarizing the BAHD acceptor
interpretability analysis.

Figures
-------
  Figure 1 — CV performance overview
    Panel A: per-label AUPR vs positive_count, with random-baseline curve and
             macro-AUPR reference line.
    Panel B: AUPR comparison between function-logit ridge and last-hidden logistic.

  Figure 2 — Diffuseness quantified (three panels)
    Panel A: Lorenz curve over |w/sigma| for one high-performing label, with a
             reference for "perfectly uniform" and a synthetic sparse curve.
    Panel B: distribution of top1_share across all (protein, label) pairs.
    Panel C: per-residue contribution profile for one high-scoring example
             (the canonical "no spikes" plot).

  Figure 3 — Cross-protein feature heatmaps for one well-performing label
    Stacks of (proteins x relative residue position) heatmaps, one per top
    feature dimension. Shows whether "important features" carry consistent
    spatial patterns across proteins or are themselves diffuse.

  Figure 4 — CV-held-out score-vs-truth for representative labels
    Small multiples grid using CV predictions, NOT training-set scores.
    Requires the training run's <task>_predictions.tsv files.

  Figure 5 — Showcase protein composite (per-residue bar + structure pointer)
    Side-by-side composite for a top-scoring well-predicted protein. The
    structural rendering is referenced via a path to the py3Dmol HTML viewer
    (which the user opens in a browser to take a real 3D screenshot).

Usage:
    python writeup_figures.py \
        --root_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability \
        --metrics_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability/metrics_jsons \
        --predictions_dir /content/drive/MyDrive/DeepLearning_final_proj/Model_Interpretability/predictions_tsvs \
        --showcase_label "acceptor_superclass::Phenolic acids (C6-C1)"

If --metrics_dir is omitted the script looks for *_metrics.json under root_dir.
If --predictions_dir is omitted Figure 4 is skipped with a warning.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# --------------------------------------------------------------------------------------
# CLI + config
# --------------------------------------------------------------------------------------

@dataclass
class RunSpec:
    label: str               # short ID for filenames
    pretty: str              # display name for plots
    metrics_filename_glob: str  # how to find this run's metrics JSON
    pickle_filename: str     # under model_pkl_files/
    interp_subdir: str       # under interpretability_outputs/
    feature_kind: str        # "last_hidden" or "function_logits"


# These describe the runs we expect to compare in Figure 1B.
RUN_SPECS: List[RunSpec] = [
    RunSpec(
        label="lasthidden",
        pretty="Last-hidden logistic",
        metrics_filename_glob="*hidden_layer*lr*acceptor*metrics.json",
        pickle_filename="hidden_layer_lr_bahd_acceptor_models.pkl",
        interp_subdir="bahd_acceptor_lasthidden",
        feature_kind="last_hidden",
    ),
    RunSpec(
        label="funclogit",
        pretty="Function-logit ridge",
        metrics_filename_glob="*logits*ridge*acceptor*metrics.json",
        pickle_filename="logits_ridge_bahd_acceptor_models.pkl",
        interp_subdir="bahd_acceptor_funclogit",
        feature_kind="function_logits",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build writeup figures (1-5) for BAHD interpretability.")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--metrics_dir", default=None,
                   help="Folder containing the *_metrics.json files. Default: search under root_dir.")
    p.add_argument("--predictions_dir", default=None,
                   help="Folder containing CV predictions TSVs (for Figure 4). Optional.")
    p.add_argument("--interp_dir", default=None,
                   help="Default: <root>/interpretability_outputs")
    p.add_argument("--last_hidden_embeddings_dir", default=None,
                   help="Default: <root>/BAHD_lastLayer_embeddings")
    p.add_argument("--out_dir", default=None,
                   help="Default: <root>/writeup_figures")
    p.add_argument("--showcase_label",
                   default="acceptor_superclass::Phenolic acids (C6-C1)",
                   help="A well-performing label used for diffuseness and heatmap figures.")
    p.add_argument("--n_top_features_for_heatmap", type=int, default=4)
    p.add_argument("--n_relpos_bins", type=int, default=50)
    p.add_argument("--max_proteins_per_heatmap", type=int, default=80,
                   help="Cap on positive proteins shown per heatmap row to keep figures readable.")
    p.add_argument("--showcase_protein", default=None,
                   help="Optional fasta_id to use for Fig 2C and Fig 5. "
                        "Defaults to the highest-scoring positive protein for the showcase label.")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------

def find_metrics_json(metrics_dir: Path, glob_pattern: str) -> Optional[Path]:
    """
    Find a metrics JSON. Tries the strict glob first, then falls back to looser
    patterns based on the task name (acceptor or donor) extracted from the glob.
    """
    matches = sorted(metrics_dir.rglob(glob_pattern))
    if matches:
        return matches[0]
    # Fall back: extract acceptor or donor from the original pattern and try generic forms.
    task = "acceptor" if "acceptor" in glob_pattern else ("donor" if "donor" in glob_pattern else None)
    if task is None:
        return None
    fallback_patterns = [
        f"*{task}*metrics.json",
        f"bahd_{task}*metrics.json",
        f"bahd_{task}_metrics.json",
    ]
    for pat in fallback_patterns:
        matches = sorted(metrics_dir.rglob(pat))
        if matches:
            return matches[0]
    return None


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def per_label_dataframe(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten per-label metrics into list of dicts."""
    rows = []
    for entry in metrics.get("metrics", {}).get("per_label", []):
        rows.append({
            "label": str(entry["label"]),
            "short": str(entry["label"]).split("::")[-1],
            "positive_count": int(entry.get("positive_count", 0)),
            "aupr": float(entry["aupr"]) if entry.get("aupr") is not None else float("nan"),
            "auroc": float(entry["auroc"]) if entry.get("auroc") is not None else float("nan"),
        })
    return rows


def load_predictions_tsv(path: Path) -> List[Dict[str, Any]]:
    """Load a predictions file. Auto-detects tab vs comma delimiter."""
    rows: List[Dict[str, Any]] = []
    # Sniff the delimiter from the header line.
    with path.open("r", encoding="utf-8") as fh:
        head = fh.readline()
    delimiter = "\t" if head.count("\t") >= head.count(",") else ","
    # The file's "vector" columns contain commas inside fields, so if the file
    # is tab-delimited we want tab; if it's truly comma-delimited as CSV we
    # rely on the writer having quoted those fields. The training script's
    # write_predictions_tsv always uses tabs, so prefer tab when tabs exist.
    if "\t" in head:
        delimiter = "\t"
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for r in reader:
            rows.append(r)
    return rows


def load_protein_bundle(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_all_protein_bundles(per_protein_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for f in sorted(per_protein_dir.glob("*.pt")):
        try:
            out.append(load_protein_bundle(f))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not load {f}: {exc}", file=sys.stderr)
    return out


def load_pipeline_for_label(pickle_path: Path, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (mu, sigma, w, b) for one label's pipeline. Raises on absence."""
    with pickle_path.open("rb") as fh:
        bundle = pickle.load(fh)
    if label not in bundle["models"]:
        raise KeyError(f"Label {label!r} not in pickle {pickle_path}")
    entry = bundle["models"][label]
    if entry["mode"] != "trained":
        raise RuntimeError(f"Label {label!r} has constant fallback in {pickle_path}; cannot extract weights")
    pipe = entry["pipeline"]
    mu = np.asarray(pipe.named_steps["scaler"].mean_, dtype=np.float64)
    sigma = np.asarray(pipe.named_steps["scaler"].scale_, dtype=np.float64)
    coef = np.asarray(pipe.named_steps["clf"].coef_, dtype=np.float64)
    w = coef[0] if coef.ndim == 2 else coef
    intercept = pipe.named_steps["clf"].intercept_
    b = float(intercept[0]) if hasattr(intercept, "__len__") else float(intercept)
    return mu, sigma, w, b


# --------------------------------------------------------------------------------------
# Figure 1: CV performance overview
# --------------------------------------------------------------------------------------

def figure1(
    last_hidden_metrics: Optional[Dict[str, Any]],
    func_logit_metrics: Optional[Dict[str, Any]],
    out_path: Path,
) -> None:
    if last_hidden_metrics is None and func_logit_metrics is None:
        print("[Fig1] Skipping — no metrics available.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Panel A: AUPR vs positive_count for the last-hidden run (the headline model)
    ax = axes[0]
    primary = last_hidden_metrics or func_logit_metrics
    primary_label = "Last-hidden logistic" if last_hidden_metrics is not None else "Function-logit ridge"
    rows = per_label_dataframe(primary)
    n_total = int(primary.get("metrics", {}).get("num_samples", 366))
    macro_aupr = float(primary["metrics"].get("macro_aupr") or float("nan"))

    pos_counts = np.array([r["positive_count"] for r in rows], dtype=float)
    auprs = np.array([r["aupr"] for r in rows], dtype=float)
    shorts = [r["short"] for r in rows]

    # Random AUPR baseline curve (positive_count / total).
    pos_curve_x = np.linspace(1, max(pos_counts.max() * 1.05, 2), 200)
    pos_curve_y = pos_curve_x / float(n_total)
    ax.plot(pos_curve_x, pos_curve_y, color="#999", linestyle=":", linewidth=1.2,
            label="random baseline (pos / total)")

    if not np.isnan(macro_aupr):
        ax.axhline(macro_aupr, color="#4a86c0", linestyle="--", linewidth=1.0,
                   label=f"macro AUPR = {macro_aupr:.2f}")

    # Color labels with very few positives differently to flag noise.
    sufficient = pos_counts >= 10
    ax.scatter(pos_counts[~sufficient], auprs[~sufficient], color="#bbbbbb", s=44,
               edgecolor="#666", linewidth=0.6, label="positives < 10 (noisy)")
    ax.scatter(pos_counts[sufficient], auprs[sufficient], color="#c0392b", s=58,
               edgecolor="black", linewidth=0.6, label="positives ≥ 10")

    # Annotate top performers among the well-supported labels.
    well_supported_idx = np.where(sufficient)[0]
    annotate_top_n = 5
    top_idx = well_supported_idx[np.argsort(-auprs[well_supported_idx])[:annotate_top_n]]
    for i in top_idx:
        ax.annotate(shorts[i], (pos_counts[i], auprs[i]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("positive count (log scale)")
    ax.set_ylabel("CV AUPR")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Panel A — {primary_label}: AUPR vs label support")
    ax.legend(loc="lower right", fontsize=8)

    # Panel B: comparison of last-hidden vs function-logit AUPR per label
    ax = axes[1]
    if last_hidden_metrics is None or func_logit_metrics is None:
        ax.text(0.5, 0.5, "Both metrics required for comparison",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#888")
        ax.axis("off")
    else:
        rows_lh = per_label_dataframe(last_hidden_metrics)
        rows_fl = per_label_dataframe(func_logit_metrics)
        # Match labels by name and sort by positive_count.
        lh_by_label = {r["label"]: r for r in rows_lh}
        fl_by_label = {r["label"]: r for r in rows_fl}
        common = [lbl for lbl in lh_by_label if lbl in fl_by_label]
        common.sort(key=lambda lbl: lh_by_label[lbl]["positive_count"])

        x = np.arange(len(common))
        width = 0.4
        lh_aupr = np.array([lh_by_label[l]["aupr"] for l in common])
        fl_aupr = np.array([fl_by_label[l]["aupr"] for l in common])
        ax.bar(x - width / 2, lh_aupr, width, label="Last-hidden logistic", color="#c0392b")
        ax.bar(x + width / 2, fl_aupr, width, label="Function-logit ridge", color="#4a86c0")

        labels_with_count = [
            f"{lh_by_label[l]['short']}  (n={lh_by_label[l]['positive_count']})" for l in common
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels_with_count, rotation=70, fontsize=7, ha="right")
        ax.set_ylabel("CV AUPR")
        ax.set_title("Panel B — AUPR by label, sorted by positive count")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig1] wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figure 2: Diffuseness, three panels
# --------------------------------------------------------------------------------------

def lorenz_curve(values_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cum_features, cum_signal) as fractions in [0, 1]."""
    s = np.sort(values_abs)[::-1]  # descending
    cum = np.cumsum(s)
    cum_signal = cum / cum[-1]
    cum_features = np.arange(1, len(s) + 1) / len(s)
    return cum_features, cum_signal


def figure2(
    pickle_dir: Path,
    showcase_label: str,
    showcase_pickle_filename: str,
    interp_dir: Path,
    interp_subdir: str,
    out_path: Path,
    showcase_protein: Optional[str],
) -> None:
    pkl_path = pickle_dir / showcase_pickle_filename
    if not pkl_path.is_file():
        print(f"[Fig2] Skipping — pickle not found at {pkl_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mu, sigma, w, b = load_pipeline_for_label(pkl_path, showcase_label)
    except Exception as exc:
        print(f"[Fig2] Could not load weights for {showcase_label}: {exc}")
        return

    abs_w = np.abs(w / sigma)
    cum_x, cum_y = lorenz_curve(abs_w)

    # Synthetic sparse comparison: a random model where 10 features carry 80%
    # of the signal and the rest carry the remaining 20%.
    D = len(abs_w)
    sparse = np.zeros(D)
    sparse[:10] = 0.08  # each carries 8%
    sparse[10:] = 0.20 / (D - 10)
    sx, sy = lorenz_curve(sparse)

    # --- Load all per-protein bundles for Panels B and C
    per_protein_dir = interp_dir / interp_subdir / "per_protein"
    proteins = load_all_protein_bundles(per_protein_dir) if per_protein_dir.is_dir() else []

    # Panel B: top1_share distribution across all (protein, label) pairs
    top1_shares: List[float] = []
    uniform_baselines: List[float] = []
    for b_dict in proteins:
        T = int(b_dict["T"])
        rc = b_dict["residue_contribs"].numpy() if hasattr(b_dict["residue_contribs"], "numpy") else np.asarray(b_dict["residue_contribs"])
        const_mask = b_dict["constant_label_mask"].numpy() if hasattr(b_dict["constant_label_mask"], "numpy") else np.asarray(b_dict["constant_label_mask"])
        for li in range(rc.shape[0]):
            if const_mask[li]:
                continue
            v = np.abs(rc[li])
            tot = v.sum()
            if tot == 0:
                continue
            top1_shares.append(v.max() / tot)
            uniform_baselines.append(1.0 / max(T, 1))

    # Panel C: residue contributions for a representative high-scoring positive
    showcase_bundle = None
    showcase_label_idx = None
    if proteins:
        # Pick highest-scoring protein on the showcase label, optionally overridden.
        if showcase_protein is not None:
            for b_dict in proteins:
                if str(b_dict.get("fasta_id")) == showcase_protein:
                    if showcase_label in list(b_dict["label_names"]):
                        showcase_bundle = b_dict
                        showcase_label_idx = list(b_dict["label_names"]).index(showcase_label)
                    break
        if showcase_bundle is None:
            best_score = -np.inf
            for b_dict in proteins:
                names = list(b_dict["label_names"])
                if showcase_label not in names:
                    continue
                li = names.index(showcase_label)
                scores = b_dict["scores"].numpy() if hasattr(b_dict["scores"], "numpy") else np.asarray(b_dict["scores"])
                if scores[li] > best_score:
                    best_score = float(scores[li])
                    showcase_bundle = b_dict
                    showcase_label_idx = li

    # --- Build figure
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel A: Lorenz curve over |w/sigma|
    ax_a.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1.0, label="perfectly uniform")
    ax_a.plot(sx, sy, color="#4a86c0", linewidth=1.4, label="hypothetical sparse\n(10 features carry 80%)")
    ax_a.plot(cum_x, cum_y, color="#c0392b", linewidth=1.8, label="this model")
    ax_a.set_xlabel("fraction of features (sorted by |weight|)")
    ax_a.set_ylabel("cumulative fraction of |weight|")
    ax_a.set_title("Panel A — weight magnitude is nearly uniform", fontsize=10)
    ax_a.legend(loc="lower right", fontsize=8)
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1.02)

    # Panel B: top1_share histogram + uniform-baseline shaded region
    if top1_shares:
        arr = np.array(top1_shares)
        ax_b.hist(arr, bins=40, color="#c0392b", edgecolor="white")
        med = float(np.median(arr))
        ax_b.axvline(med, color="black", linestyle="-", linewidth=1.0, label=f"median = {med:.4f}")
        ub_med = float(np.median(uniform_baselines))
        ax_b.axvline(ub_med, color="#888", linestyle="--", linewidth=1.0,
                     label=f"uniform baseline ≈ {ub_med:.4f}")
        ax_b.set_xlabel("top-1 residue share of |contribution|")
        ax_b.set_ylabel("count of (protein, label) pairs")
        ax_b.set_title("Panel B — most extreme residue is\nbarely above uniform", fontsize=10)
        ax_b.legend(loc="upper right", fontsize=8)
    else:
        ax_b.text(0.5, 0.5, "No per-protein bundles found", ha="center", va="center",
                  transform=ax_b.transAxes, color="#888")
        ax_b.axis("off")

    # Panel C: per-residue bars for the showcase example
    if showcase_bundle is not None and showcase_label_idx is not None:
        rc = showcase_bundle["residue_contribs"].numpy() if hasattr(showcase_bundle["residue_contribs"], "numpy") else np.asarray(showcase_bundle["residue_contribs"])
        contribs = rc[showcase_label_idx]
        score = float(showcase_bundle["scores"].numpy()[showcase_label_idx]) if hasattr(showcase_bundle["scores"], "numpy") else float(showcase_bundle["scores"][showcase_label_idx])
        T = len(contribs)
        positions = np.arange(T)
        colors = ["#c0392b" if v >= 0 else "#2c5fa6" for v in contribs]
        ax_c.bar(positions, contribs, color=colors, width=1.0, edgecolor="none")
        ax_c.axhline(0, color="black", linewidth=0.5)
        ax_c.set_xlim(-0.5, T - 0.5)
        ax_c.set_xlabel("residue position")
        ax_c.set_ylabel("contribution to score")
        ax_c.set_title(
            f"Panel C — {showcase_bundle['fasta_id']} on {showcase_label.split('::')[-1]}\n"
            f"score = {score:.2f}, T = {T} — no localized region",
            fontsize=10,
        )
    else:
        ax_c.text(0.5, 0.5, "No matching protein found for showcase",
                  ha="center", va="center", transform=ax_c.transAxes, color="#888")
        ax_c.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig2] wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figure 3: Cross-protein feature heatmaps
# --------------------------------------------------------------------------------------

def find_lasthidden_tensor(fasta_id: str, embeddings_dir: Path) -> Optional[Path]:
    """Find the last-hidden-layer .pt for one protein."""
    candidates = [
        embeddings_dir / f"{fasta_id}_hidden_layer_steps10.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Permissive glob fallback.
    matches = list(embeddings_dir.glob(f"*{fasta_id}*hidden_layer*.pt"))
    if len(matches) == 1:
        return matches[0]
    return None


def load_real_residue_tensor(
    fasta_id: str, embeddings_dir: Path, structures_root: Optional[Path],
) -> Optional[np.ndarray]:
    """Load (T_real, D) — strip BOS/EOS if PDB indicates they're present."""
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

    # Optional BOS/EOS strip if structures_root is provided
    if structures_root is not None:
        sub = structures_root / fasta_id
        if sub.is_dir():
            pdbs = list(sub.rglob("*.pdb"))
            if pdbs:
                # Fast residue count
                seen = set()
                with pdbs[0].open() as fh:
                    for line in fh:
                        if line.startswith("ATOM"):
                            seen.add((line[21:22], line[22:26], line[26:27]))
                t_pdb = len(seen)
                if arr.shape[0] == t_pdb + 2:
                    arr = arr[1:-1]
    return arr


def bin_to_relpos(values_t: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean-bin a (T,) array onto a [0, 1] axis with n_bins."""
    T = len(values_t)
    if T == 0:
        return np.zeros(n_bins)
    pos = (np.arange(T) + 0.5) / T
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.minimum(np.digitize(pos, edges) - 1, n_bins - 1)
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(sums, idx, values_t)
    np.add.at(counts, idx, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, sums / counts, np.nan)
    return out


def figure3(
    last_hidden_pickle: Path,
    showcase_label: str,
    interp_dir: Path,
    interp_subdir: str,
    embeddings_dir: Path,
    structures_root: Optional[Path],
    n_top_features: int,
    n_relpos_bins: int,
    max_proteins: int,
    out_path: Path,
) -> None:
    if not last_hidden_pickle.is_file():
        print(f"[Fig3] Skipping — pickle not found at {last_hidden_pickle}")
        return
    if not embeddings_dir.is_dir():
        print(f"[Fig3] Skipping — embeddings dir not found at {embeddings_dir}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mu, sigma, w, b_intercept = load_pipeline_for_label(last_hidden_pickle, showcase_label)
    except Exception as exc:
        print(f"[Fig3] Could not load weights for {showcase_label}: {exc}")
        return

    abs_w_raw = np.abs(w / sigma)
    top_features = np.argsort(abs_w_raw)[::-1][:n_top_features]
    top_signs = np.sign((w / sigma)[top_features])

    # Find positive proteins on this label using the per-protein interpretability outputs.
    per_protein_dir = interp_dir / interp_subdir / "per_protein"
    if not per_protein_dir.is_dir():
        print(f"[Fig3] Skipping — per_protein dir not found at {per_protein_dir}")
        return
    bundles = load_all_protein_bundles(per_protein_dir)
    pos_with_score: List[Tuple[Dict[str, Any], float]] = []
    for b in bundles:
        names = list(b["label_names"])
        if showcase_label not in names:
            continue
        li = names.index(showcase_label)
        scores = b["scores"].numpy() if hasattr(b["scores"], "numpy") else np.asarray(b["scores"])
        if scores[li] > 0:
            pos_with_score.append((b, float(scores[li])))
    pos_with_score.sort(key=lambda x: -x[1])
    pos_with_score = pos_with_score[:max_proteins]
    if not pos_with_score:
        print("[Fig3] No proteins with positive predicted score on showcase label; skipping.")
        return

    # Build heatmaps: one per top feature
    n_proteins = len(pos_with_score)
    heatmaps = np.zeros((n_top_features, n_proteins, n_relpos_bins), dtype=np.float64)
    valid_mask = np.zeros((n_top_features, n_proteins), dtype=bool)
    fasta_ids: List[str] = []
    sequence_lengths: List[int] = []

    for protein_idx, (bdict, _score) in enumerate(pos_with_score):
        fid = str(bdict["fasta_id"])
        fasta_ids.append(fid)
        H_real = load_real_residue_tensor(fid, embeddings_dir, structures_root)
        if H_real is None:
            sequence_lengths.append(0)
            continue
        sequence_lengths.append(H_real.shape[0])
        for k, fdim in enumerate(top_features):
            # Center and scale so the heatmap shows standardized activation,
            # which lines up with what the model "sees".
            std_act = (H_real[:, int(fdim)] - mu[int(fdim)]) / sigma[int(fdim)]
            # If this feature has a negative coefficient sign, flip the sign so
            # red always means "pushes toward predicting positive".
            std_act = std_act * top_signs[k]
            heatmaps[k, protein_idx] = bin_to_relpos(std_act, n_relpos_bins)
            valid_mask[k, protein_idx] = True

    # Symmetric color scale per feature — use 95th percentile of |values| to avoid outliers.
    fig, axes = plt.subplots(n_top_features, 1, figsize=(11, 2.4 * n_top_features), squeeze=False)
    for k in range(n_top_features):
        ax = axes[k][0]
        block = heatmaps[k]
        # Replace nans with 0 for display (binned positions with no residues).
        block_disp = np.where(np.isnan(block), 0.0, block)
        finite = block_disp[np.isfinite(block_disp)]
        vmax = float(np.percentile(np.abs(finite), 95)) if finite.size else 1.0
        if vmax == 0:
            vmax = 1.0
        im = ax.imshow(block_disp, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_ylabel("protein index\n(by score, desc)")
        if k == n_top_features - 1:
            ax.set_xlabel(f"relative residue position ({n_relpos_bins} bins)")
        coef_signed = float((w / sigma)[int(top_features[k])])
        ax.set_title(
            f"Feature {int(top_features[k])} | sign-aligned activation\n"
            f"raw-space coefficient = {coef_signed:+.4f}",
            fontsize=9.5,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Figure 3 — Cross-protein feature activation maps for top {n_top_features} features\n"
        f"showcase label: {showcase_label.split('::')[-1]}  "
        f"(n_proteins shown = {n_proteins})\n"
        "If signal is localized at the feature level, we expect vertical stripes; "
        "horizontal-only patterns indicate diffuse use.",
        fontsize=10.5, y=1.005,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig3] wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figure 4: CV-held-out score-vs-truth for representative labels
# --------------------------------------------------------------------------------------

def figure4(
    predictions_tsv: Optional[Path],
    metrics: Dict[str, Any],
    representative_labels: Sequence[str],
    out_path: Path,
) -> None:
    if predictions_tsv is None or not predictions_tsv.is_file():
        print(f"[Fig4] Skipping — no predictions TSV available at {predictions_tsv}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_predictions_tsv(predictions_tsv)
    if not rows:
        print(f"[Fig4] Skipping — empty predictions TSV at {predictions_tsv}")
        return

    # Reconstruct (true, score) per (protein, label) from the TSV.
    # Schema (from training script): true_vector, score_vector are comma-separated
    # in the same order as the kept_labels in metrics.
    label_names = list(metrics.get("kept_labels") or [])
    if not label_names:
        # try metrics.metrics.per_label
        label_names = [r["label"] for r in metrics.get("metrics", {}).get("per_label", [])]
    if not label_names:
        print("[Fig4] Skipping — could not infer label order.")
        return

    # Build (n_proteins, n_labels) score and truth matrices.
    n_proteins = len(rows)
    n_labels = len(label_names)
    truth = np.full((n_proteins, n_labels), -1, dtype=int)
    scores = np.zeros((n_proteins, n_labels), dtype=float)
    for i, r in enumerate(rows):
        try:
            t_vec = [int(x) for x in r["true_vector"].split(",")]
            s_vec = [float(x) for x in r["score_vector"].split(",")]
        except Exception:
            continue
        if len(t_vec) == n_labels and len(s_vec) == n_labels:
            truth[i] = t_vec
            scores[i] = s_vec

    # Filter to requested labels (with prefix tolerance).
    label_idx_map: Dict[str, int] = {}
    for needle in representative_labels:
        if needle in label_names:
            label_idx_map[needle] = label_names.index(needle)
        else:
            # tolerate short-name lookup
            for j, lbl in enumerate(label_names):
                if lbl.endswith(needle) or needle in lbl:
                    label_idx_map[lbl] = j
                    break
    if not label_idx_map:
        print("[Fig4] No representative labels matched the predictions TSV; skipping.")
        return

    # Lay out a grid.
    items = list(label_idx_map.items())
    n = len(items)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.0, rows_n * 3.4), squeeze=False)
    rng = np.random.default_rng(42)

    for k, (label, j) in enumerate(items):
        ax = axes[k // cols][k % cols]
        s = scores[:, j]
        t = truth[:, j]
        mask = t >= 0
        s = s[mask]; t = t[mask]
        s_neg = s[t == 0]
        s_pos = s[t == 1]
        if len(s_neg):
            j1 = rng.uniform(-0.18, 0.18, size=len(s_neg))
            ax.scatter(np.zeros_like(s_neg) + j1, s_neg, color="#446a98", alpha=0.55, s=14, edgecolor="none")
        if len(s_pos):
            j2 = rng.uniform(-0.18, 0.18, size=len(s_pos))
            ax.scatter(np.ones_like(s_pos) + j2, s_pos, color="#c0392b", alpha=0.55, s=14, edgecolor="none")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"neg\nn={len(s_neg)}", f"pos\nn={len(s_pos)}"], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("CV score (held-out)", fontsize=8)
        # AUPR for that label from metrics
        per_label = {r["label"]: r for r in metrics.get("metrics", {}).get("per_label", [])}
        aupr_val = per_label.get(label, {}).get("aupr")
        ax.set_title(
            f"{label.split('::')[-1]}\n"
            f"AUPR = {aupr_val:.2f}" if aupr_val is not None else label.split("::")[-1],
            fontsize=9,
        )

    for k in range(n, rows_n * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle("Figure 4 — Score vs ground truth on held-out CV folds", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig4] wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figure 5: Showcase composite (per-residue bar + structure pointer)
# --------------------------------------------------------------------------------------

def figure5(
    interp_dir: Path,
    interp_subdir: str,
    showcase_label: str,
    structures_root: Optional[Path],
    out_path: Path,
    showcase_protein: Optional[str],
) -> Optional[Path]:
    """Build a composite bar-plot figure and print the HTML viewer path for manual screenshotting."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_protein_dir = interp_dir / interp_subdir / "per_protein"
    if not per_protein_dir.is_dir():
        print(f"[Fig5] Skipping — per_protein dir missing at {per_protein_dir}")
        return None
    bundles = load_all_protein_bundles(per_protein_dir)
    chosen = None
    chosen_li = None

    if showcase_protein is not None:
        for b in bundles:
            if str(b.get("fasta_id")) == showcase_protein:
                names = list(b["label_names"])
                if showcase_label in names:
                    chosen = b
                    chosen_li = names.index(showcase_label)
                break

    if chosen is None:
        best_score = -np.inf
        for b in bundles:
            names = list(b["label_names"])
            if showcase_label not in names:
                continue
            li = names.index(showcase_label)
            scores = b["scores"].numpy() if hasattr(b["scores"], "numpy") else np.asarray(b["scores"])
            if scores[li] > best_score:
                best_score = float(scores[li])
                chosen = b
                chosen_li = li

    if chosen is None or chosen_li is None:
        print(f"[Fig5] Skipping — no positive protein found for {showcase_label}")
        return None

    rc = chosen["residue_contribs"].numpy() if hasattr(chosen["residue_contribs"], "numpy") else np.asarray(chosen["residue_contribs"])
    contribs = rc[chosen_li]
    score = float(chosen["scores"].numpy()[chosen_li]) if hasattr(chosen["scores"], "numpy") else float(chosen["scores"][chosen_li])
    fasta_id = str(chosen["fasta_id"])
    T = len(contribs)
    short_label = showcase_label.split("::")[-1]

    # Find the corresponding html viewer path (the user opens it in a browser to take a screenshot)
    safe_label = "".join(c if (c.isalnum() or c in "._-") else "_" for c in showcase_label)
    safe_id = "".join(c if (c.isalnum() or c in "._-") else "_" for c in fasta_id)
    viz_html = (interp_dir / "visualizations" / interp_subdir / "04_structures" / safe_label /
                f"{safe_id}__{safe_label}.html")

    fig, ax = plt.subplots(figsize=(11, 4))
    positions = np.arange(T)
    colors = ["#c0392b" if v >= 0 else "#2c5fa6" for v in contribs]
    ax.bar(positions, contribs, color=colors, width=1.0, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel("residue position")
    ax.set_ylabel("contribution to score")
    ax.set_title(
        f"Figure 5 — Showcase: {fasta_id} on {short_label}\n"
        f"score = {score:.2f}, T = {T}\n"
        f"Open the 3D viewer for the structural counterpart:\n"
        f"  {viz_html}",
        fontsize=9, loc="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig5] wrote {out_path}")
    print(f"[Fig5] structural HTML viewer expected at: {viz_html}")
    return viz_html


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    metrics_dir = Path(args.metrics_dir).resolve() if args.metrics_dir else root
    predictions_dir = Path(args.predictions_dir).resolve() if args.predictions_dir else None
    interp_dir = Path(args.interp_dir).resolve() if args.interp_dir else (root / "interpretability_outputs")
    embeddings_dir = (
        Path(args.last_hidden_embeddings_dir).resolve() if args.last_hidden_embeddings_dir
        else (root / "BAHD_lastLayer_embeddings")
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "writeup_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    structures_root = root / "BAHD_dataset" / "BAHD_structures"
    if not structures_root.is_dir():
        structures_root = None  # type: ignore

    # Locate metrics JSONs
    lh_metrics_path = find_metrics_json(metrics_dir, RUN_SPECS[0].metrics_filename_glob)
    fl_metrics_path = find_metrics_json(metrics_dir, RUN_SPECS[1].metrics_filename_glob)
    # If both globs resolve to the same file (because only one model's metrics
    # exist), check which model the JSON actually describes via its feature_type
    # field, and assign it to the correct slot only.
    if lh_metrics_path is not None and lh_metrics_path == fl_metrics_path:
        try:
            with lh_metrics_path.open() as fh:
                probe = json.load(fh)
            ft = str(probe.get("feature_type", "")).lower()
            if "last_layer" in ft or "hidden" in ft:
                fl_metrics_path = None
                print(f"[paths] only last-hidden metrics found; function-logit ridge metrics not present")
            elif "function_logits" in ft or "ridge" in ft:
                lh_metrics_path = None
                print(f"[paths] only function-logit metrics found; last-hidden logistic metrics not present")
            else:
                print(f"[paths] [warn] could not infer feature_type from {lh_metrics_path}")
                fl_metrics_path = None  # default to keeping it as last-hidden
        except Exception as exc:
            print(f"[paths] could not probe {lh_metrics_path}: {exc}")
            fl_metrics_path = None
    print(f"[paths] lh metrics:  {lh_metrics_path}")
    print(f"[paths] fl metrics:  {fl_metrics_path}")
    print(f"[paths] interp dir:  {interp_dir}")
    print(f"[paths] embeddings:  {embeddings_dir}")
    print(f"[paths] structures:  {structures_root}")
    print(f"[paths] predictions: {predictions_dir}")
    print(f"[paths] out_dir:     {out_dir}")

    lh_metrics = load_metrics(lh_metrics_path) if lh_metrics_path else None
    fl_metrics = load_metrics(fl_metrics_path) if fl_metrics_path else None

    # Figure 1
    figure1(lh_metrics, fl_metrics, out_dir / "figure_1_cv_performance.png")

    # Figure 2 (uses last-hidden pickle for weights, last-hidden interp for residue distributions)
    figure2(
        pickle_dir=root / "model_pkl_files",
        showcase_label=args.showcase_label,
        showcase_pickle_filename=RUN_SPECS[0].pickle_filename,
        interp_dir=interp_dir,
        interp_subdir=RUN_SPECS[0].interp_subdir,
        out_path=out_dir / "figure_2_diffuseness.png",
        showcase_protein=args.showcase_protein,
    )

    # Figure 3 (cross-protein feature heatmaps for last-hidden model)
    figure3(
        last_hidden_pickle=root / "model_pkl_files" / RUN_SPECS[0].pickle_filename,
        showcase_label=args.showcase_label,
        interp_dir=interp_dir,
        interp_subdir=RUN_SPECS[0].interp_subdir,
        embeddings_dir=embeddings_dir,
        structures_root=structures_root,
        n_top_features=int(args.n_top_features_for_heatmap),
        n_relpos_bins=int(args.n_relpos_bins),
        max_proteins=int(args.max_proteins_per_heatmap),
        out_path=out_dir / "figure_3_feature_heatmaps.png",
    )

    # Figure 4 (CV held-out score vs truth, only if predictions TSV available)
    representative_labels = [
        "acceptor_superclass::Phenolic acids (C6-C1)",  # high performer (n=78)
        "acceptor_superclass::Ornithine alkaloids",     # high performer (n=43)
        "acceptor_superclass::Naphthalenes",            # high performer (n=19)
        "acceptor_superclass::Flavonoids",              # medium (n=42)
        "acceptor_superclass::Tyrosine alkaloids",      # medium-low (n=14)
        "acceptor_superclass::Lignans",                 # too few positives (n=2)
    ]
    predictions_tsv = None
    if predictions_dir is not None and predictions_dir.is_dir():
        # Try increasingly permissive patterns to find the acceptor predictions file.
        # Supports both .tsv and .csv extensions; the loader auto-detects delimiter.
        patterns = [
            "*hidden*lr*acceptor*predictions.tsv",
            "*hidden*lr*acceptor*predictions.csv",
            "*acceptor*predictions.tsv",
            "*acceptor*predictions.csv",
            "bahd_acceptor_predictions.tsv",
            "bahd_acceptor_predictions.csv",
        ]
        for pat in patterns:
            candidates = list(predictions_dir.rglob(pat))
            if candidates:
                predictions_tsv = candidates[0]
                print(f"[paths] predictions file: {predictions_tsv}  (matched {pat!r})")
                break
        if predictions_tsv is None:
            print(f"[paths] no predictions file matched under {predictions_dir}")

    if lh_metrics is not None:
        figure4(predictions_tsv, lh_metrics, representative_labels, out_dir / "figure_4_score_vs_truth_cv.png")
    else:
        print("[Fig4] Skipping — last-hidden metrics missing.")

    # Figure 5
    figure5(
        interp_dir=interp_dir,
        interp_subdir=RUN_SPECS[0].interp_subdir,
        showcase_label=args.showcase_label,
        structures_root=structures_root,
        out_path=out_dir / "figure_5_showcase.png",
        showcase_protein=args.showcase_protein,
    )

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
