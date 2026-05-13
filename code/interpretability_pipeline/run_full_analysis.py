#!/usr/bin/env python3
"""
run_full_analysis.py

Orchestrate the BAHD interpretability analysis pipeline. Each step writes its
outputs under a single unified directory (Option B) so the final state is
organized and easy to share.

Layout under <out_dir>:
    interpretability_outputs/        from linear_interpretability.py
    discriminative_features_full/    from discriminative_features_full.py
    writeup_figures/                 from writeup_figures.py
    figure_3_extended/               from figure_3_extended.py
    final_analysis_plots/            from final_analysis_plots.py
    WRITEUP.docx                     from build_writeup_docx.py

Default --out_dir is <root>/analysis_outputs.

Steps that have already finished can be skipped with --skip:
    --skip linear_interpretability discriminative_features_full

Usage:
    python run_full_analysis.py \\
        --root_dir /content/drive/MyDrive/.../Model_Interpretability

Add --skip <step_name> ... to omit steps.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# All known steps in execution order.
STEPS: List[str] = [
    "linear_interpretability",
    "discriminative_features_full",
    "writeup_figures",
    "figure_3_extended",
    "final_analysis_plots",
    "build_writeup_docx",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True,
                   help="Project root containing the BAHD dataset, model pickles, "
                        "embeddings, and the analysis scripts.")
    p.add_argument("--out_dir", default=None,
                   help="Unified output directory. Default: <root>/analysis_outputs")
    p.add_argument("--metrics_dir", default=None,
                   help="Folder containing the *_metrics.json files for the trained "
                        "models. Default: <root>/5fold_cv_ll_lr_preds")
    p.add_argument("--predictions_dir", default=None,
                   help="Folder containing the *_predictions.tsv files for "
                        "writeup_figures. Default: <metrics_dir>")
    p.add_argument("--markdown_path", default=None,
                   help="Path to WRITEUP.md. Default: <root>/WRITEUP.md")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Step names to skip. Valid: " + ", ".join(STEPS))
    p.add_argument("--figure_3_labels", nargs="*", default=None,
                   help="Override the figure_3_extended label list. By default, "
                        "the launcher picks all labels with positive_count >= 10. "
                        "If you pass labels, they must all be from the same task "
                        "(all acceptor or all donor), since --task_prefix can only "
                        "take one value.")
    p.add_argument("--figure_3_task", default="acceptor",
                   choices=["acceptor", "donor"],
                   help="Which task --figure_3_labels belong to (only used when "
                        "--figure_3_labels is set).")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable to use for child processes.")
    p.add_argument("--continue_on_error", action="store_true",
                   help="If a step fails, log it and proceed instead of aborting.")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Step runner
# --------------------------------------------------------------------------------------

def run_step(name: str, cmd: List[str], continue_on_error: bool) -> bool:
    """Run a single step. Returns True on success."""
    print("\n" + "=" * 78)
    print(f"[step] {name}")
    print(f"[cmd]  {shlex.join(cmd)}")
    print("=" * 78)
    rc = subprocess.call(cmd)
    if rc != 0:
        msg = f"[step] {name}: FAILED with exit code {rc}"
        if continue_on_error:
            print(msg + " (continuing because --continue_on_error)")
            return False
        raise SystemExit(msg)
    print(f"[step] {name}: ok")
    return True


# --------------------------------------------------------------------------------------
# figure_3_extended label discovery
# --------------------------------------------------------------------------------------

def figure_3_label_groups(metrics_dir: Path, min_pos: int = 10) -> Tuple[List[str], List[str]]:
    """
    Read the metrics JSONs and return (acceptor_labels, donor_labels) — short label
    names (no task_prefix) that have positive_count >= min_pos.
    """
    acceptor: List[str] = []
    donor: List[str] = []

    acc_glob = list(metrics_dir.rglob("*acceptor*metrics.json"))
    don_glob = list(metrics_dir.rglob("*donor*metrics.json"))

    def _collect(paths: List[Path], dest: List[str], prefix: str) -> None:
        if not paths:
            return
        # Pick the most recent metrics file if multiple exist.
        paths_sorted = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
        with paths_sorted[0].open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data.get("metrics", {}).get("per_label", []):
            label_full = str(entry.get("label", ""))
            n_pos = int(entry.get("positive_count", 0))
            if n_pos >= min_pos and label_full.startswith(prefix):
                short = label_full[len(prefix):]
                dest.append(short)

    _collect(acc_glob, acceptor, "acceptor_superclass::")
    _collect(don_glob, donor, "donor_type::")
    return acceptor, donor


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root_dir not found: {root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "analysis_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = (Path(args.metrics_dir).resolve() if args.metrics_dir
                   else (root / "5fold_cv_ll_lr_preds"))
    predictions_dir = (Path(args.predictions_dir).resolve() if args.predictions_dir
                       else metrics_dir)
    markdown_path = (Path(args.markdown_path).resolve() if args.markdown_path
                     else (root / "WRITEUP.md"))

    # Per-step output subdirectories under the unified out_dir.
    interp_subdir   = out_dir / "interpretability_outputs"
    discrim_subdir  = out_dir / "discriminative_features_full"
    top_cells_dir   = discrim_subdir / "per_label_top_cells"
    writeup_subdir  = out_dir / "writeup_figures"
    fig3ext_subdir  = out_dir / "figure_3_extended"
    finalplt_subdir = out_dir / "final_analysis_plots"
    docx_path       = out_dir / "WRITEUP.docx"

    skip = set(args.skip)
    unknown = skip - set(STEPS)
    if unknown:
        raise SystemExit(f"Unknown --skip values: {sorted(unknown)}. Valid: {STEPS}")

    print("=" * 78)
    print(f"[paths] root_dir:        {root}")
    print(f"[paths] out_dir:         {out_dir}")
    print(f"[paths] metrics_dir:     {metrics_dir}")
    print(f"[paths] predictions_dir: {predictions_dir}")
    print(f"[paths] markdown_path:   {markdown_path}")
    print(f"[paths] python:          {args.python}")
    if skip:
        print(f"[paths] skipping steps:  {sorted(skip)}")
    print("=" * 78)

    py = args.python

    # ---------- Step 1: linear_interpretability ----------
    if "linear_interpretability" not in skip:
        cmd = [
            py, str(root / "linear_interpretability.py"),
            "--root_dir", str(root),
            "--family", "BAHD",
            "--output_dir", str(interp_subdir),
            "--skip_existing",
        ]
        run_step("linear_interpretability", cmd, args.continue_on_error)

    # ---------- Step 2: discriminative_features_full ----------
    if "discriminative_features_full" not in skip:
        cmd = [
            py, str(root / "discriminative_features_full.py"),
            "--root_dir", str(root),
            "--out_dir", str(discrim_subdir),
        ]
        run_step("discriminative_features_full", cmd, args.continue_on_error)

    # ---------- Step 3: writeup_figures ----------
    if "writeup_figures" not in skip:
        cmd = [
            py, str(root / "writeup_figures.py"),
            "--root_dir", str(root),
            "--metrics_dir", str(metrics_dir),
            "--predictions_dir", str(predictions_dir),
            "--interp_dir", str(interp_subdir),
            "--out_dir", str(writeup_subdir),
        ]
        run_step("writeup_figures", cmd, args.continue_on_error)

    # ---------- Step 4: figure_3_extended ----------
    if "figure_3_extended" not in skip:
        # Decide which labels to run.
        if args.figure_3_labels:
            label_groups: List[Tuple[str, List[str]]] = [
                (args.figure_3_task, list(args.figure_3_labels))
            ]
        else:
            acceptor_labels, donor_labels = figure_3_label_groups(metrics_dir, min_pos=10)
            print(f"  [info] {len(acceptor_labels)} acceptor labels, "
                  f"{len(donor_labels)} donor labels with n_pos >= 10")
            label_groups = []
            if acceptor_labels:
                label_groups.append(("acceptor", acceptor_labels))
            if donor_labels:
                label_groups.append(("donor", donor_labels))

        for task, labels in label_groups:
            if not labels:
                continue
            task_prefix = "acceptor_superclass::" if task == "acceptor" else "donor_type::"
            cmd = [
                py, str(root / "figure_3_extended.py"),
                "--root_dir", str(root),
                "--metrics_dir", str(metrics_dir),
                "--task_prefix", task_prefix,
                "--canvas_format", "png",
                "--interp_dir", str(interp_subdir),
                "--top_cells_dir", str(top_cells_dir),
                "--out_dir", str(fig3ext_subdir),
                "--label", *labels,
            ]
            run_step(f"figure_3_extended ({task})", cmd, args.continue_on_error)

    # ---------- Step 5: final_analysis_plots ----------
    if "final_analysis_plots" not in skip:
        cmd = [
            py, str(root / "final_analysis_plots.py"),
            "--root_dir", str(root),
            "--metrics_dir", str(metrics_dir),
            "--interp_dir", str(interp_subdir),
            "--top_cells_dir", str(top_cells_dir),
            "--out_dir", str(finalplt_subdir),
        ]
        run_step("final_analysis_plots", cmd, args.continue_on_error)

    # ---------- Step 6: build_writeup_docx ----------
    if "build_writeup_docx" not in skip:
        if not markdown_path.is_file():
            print(f"\n[step] build_writeup_docx: SKIPPING — markdown not found at {markdown_path}")
        else:
            cmd = [
                py, str(root / "build_writeup_docx.py"),
                "--root_dir", str(root),
                "--out_dir", str(out_dir),
                "--markdown_path", str(markdown_path),
                "--output_path", str(docx_path),
            ]
            run_step("build_writeup_docx", cmd, args.continue_on_error)

    print("\n" + "=" * 78)
    print("Pipeline complete.")
    print(f"All outputs are under: {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
