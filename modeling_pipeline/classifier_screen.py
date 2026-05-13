#!/usr/bin/env python3
"""
classifier_screen.py

Screens a suite of conventional sklearn classifiers on mean-pooled ESM3 features
(function logits or last-layer embeddings) using the same 5-fold CV protocol and
micro AUPR metric as the main BAHD workflow.

Inspired by LazyPredict (https://github.com/shankarpandala/lazypredict), which
pioneered rapid multi-classifier screening. This implementation replaces
LazyPredict's single train/test split and ROC AUC metric with 5-fold CV and
micro AUPR for direct comparability with sections 3.1-3.6 results.

Feature representations:
    function_logits : [residues, 8, 260] flattened to [residues, 2080], then mean-pooled -> [2080]
    last_layer      : [residues, 1536] mean-pooled -> [1536]

Typical usage:
    python classifier_screen.py --feature_type function_logits --dataset BAHD
    python classifier_screen.py --feature_type last_layer --dataset BAHD
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from utils import (
    build_cv_splits,
    build_sequence_feature_list,
    dataset_dir_for_family,
    filter_labels_by_support,
    get_device,
    get_task_definition,
    load_dataset_cache,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen sklearn classifiers on mean-pooled ESM3 features via 5-fold CV."
    )
    parser.add_argument("--feature_type", choices=("function_logits", "last_layer"),
                        required=True, help="Which ESM3 feature to screen on.")
    parser.add_argument("--dataset", choices=("BAHD", "UGT", "both"), default="BAHD")
    parser.add_argument("--task", choices=("donor", "acceptor", "both"), default="both")
    parser.add_argument("--bahd_dataset_dir",
                        default=str(dataset_dir_for_family("BAHD")))
    parser.add_argument("--ugt_dataset_dir",
                        default=str(dataset_dir_for_family("UGT")))
    # Last-layer only
    parser.add_argument("--bahd_embedding_dir",
                        default=str(SCRIPT_DIR / "BAHD_lastLayer_embeddings"))
    parser.add_argument("--ugt_embedding_dir",
                        default=str(SCRIPT_DIR / "UGT_lastLayer_embeddings"))
    parser.add_argument("--embedding_glob", default="*_hidden_layer_steps10.pt")
    parser.add_argument("--output_dir",
                        default=str(SCRIPT_DIR / "results" / "classifier_screen"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--cv_mode", choices=("kfold", "loocv"), default="kfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--min_positive_count", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


# ── Classifier registry ───────────────────────────────────────────────────────

def build_classifier_list(random_seed: int) -> List[Tuple[str, Any]]:
    classifiers = [
        ("LogisticRegression",       LogisticRegression(max_iter=1000, random_state=random_seed)),
        ("RidgeClassifier",          RidgeClassifier()),
        ("LinearSVC",                LinearSVC(max_iter=5000, random_state=random_seed)),
        ("SGDClassifier",            SGDClassifier(max_iter=1000, random_state=random_seed)),
        ("RandomForest",             RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=1)),
        ("ExtraTrees",               ExtraTreesClassifier(n_estimators=100, random_state=random_seed, n_jobs=1)),
        ("DecisionTree",             DecisionTreeClassifier(random_state=random_seed)),
        ("HistGradientBoosting",     HistGradientBoostingClassifier(random_state=random_seed)),
        ("KNeighbors",               KNeighborsClassifier(n_neighbors=5, n_jobs=1)),
        ("GaussianNB",               GaussianNB()),
        ("BernoulliNB",              BernoulliNB()),
        ("SVC_RBF",                  SVC(kernel="rbf", probability=True, random_state=random_seed)),
        ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
    ]
    try:
        from lightgbm import LGBMClassifier
        classifiers.append(
            ("LightGBM", LGBMClassifier(random_state=random_seed, n_jobs=1, verbose=-1))
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        classifiers.append(
            ("XGBoost", XGBClassifier(random_state=random_seed, n_jobs=1,
                                      eval_metric="logloss", verbosity=0))
        )
    except ImportError:
        pass
    return classifiers


# ── Feature loading ───────────────────────────────────────────────────────────

def load_fl_features(dataset_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mean-pool per-residue function logit tensors -> [n_proteins, 2080]."""
    cache = load_dataset_cache(dataset_dir)
    sequences = build_sequence_feature_list(dataset_dir, cache, device="cpu", flatten=True)
    X = np.stack([seq.mean(axis=0) for seq in sequences], axis=0).astype(np.float32)
    return X, cache


def _strip_hidden_layer_suffix(stem: str) -> str:
    marker = "_hidden_layer_steps"
    return stem.split(marker)[0] if marker in stem else stem


def _build_embedding_index(embedding_dir: Path, glob: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(embedding_dir.glob(glob)):
        index[path.stem] = path
        index[_strip_hidden_layer_suffix(path.stem)] = path
    return index


def load_ll_features(
    dataset_dir: Path,
    embedding_dir: Path,
    embedding_glob: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mean-pool last-layer hidden state tensors -> [n_proteins, 1536]."""
    cache = load_dataset_cache(dataset_dir)
    fasta_ids = list(cache.get("fasta_ids", []))
    index = _build_embedding_index(embedding_dir, embedding_glob)
    vectors: List[np.ndarray] = []

    for idx, fid in enumerate(fasta_ids):
        path = index.get(str(fid))
        if path is None:
            enzyme_id = str(list(cache.get("enzyme_ids", []))[idx]) if idx < len(cache.get("enzyme_ids", [])) else ""
            path = index.get(enzyme_id)
        if path is None:
            raise FileNotFoundError(
                f"No embedding found for fasta_id={fid} in {embedding_dir}"
            )
        tensor = torch.load(path, map_location="cpu").detach().float()
        if tensor.ndim != 2:
            raise ValueError(f"Expected [residues, hidden_dim], got {tuple(tensor.shape)}")
        vectors.append(tensor.mean(dim=0).numpy().astype(np.float32))

    X = np.stack(vectors, axis=0)
    return X, cache


# ── Scoring helpers ───────────────────────────────────────────────────────────

def get_oof_scores(clf: OneVsRestClassifier, X: np.ndarray) -> np.ndarray:
    """Return probability/decision scores, preferring predict_proba."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return clf.decision_function(X)


def safe_micro_aupr(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if y_true.sum() == 0:
        return None
    val = float(average_precision_score(y_true.reshape(-1), y_score.reshape(-1)))
    return val if np.isfinite(val) else None


def safe_macro_aupr(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        val = float(average_precision_score(y_true[:, i], y_score[:, i]))
        if np.isfinite(val):
            scores.append(val)
    return float(np.mean(scores)) if scores else None


def safe_micro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    val = float(roc_auc_score(y_true.reshape(-1), y_score.reshape(-1)))
    return val if np.isfinite(val) else None


def safe_macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        val = float(roc_auc_score(y_true[:, i], y_score[:, i]))
        if np.isfinite(val):
            scores.append(val)
    return float(np.mean(scores)) if scores else None


# ── CV loop ───────────────────────────────────────────────────────────────────

def run_cv_for_classifier(
    name: str,
    base_clf: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    random_seed: int,
) -> Dict[str, Any]:
    n_samples, n_labels = y.shape
    score_matrix = np.zeros((n_samples, n_labels), dtype=np.float32)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = OneVsRestClassifier(
            base_clf.__class__(**base_clf.get_params()),
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        scores = get_oof_scores(clf, X_test)

        # BernoulliNB / GaussianNB clamp values; ensure shape matches
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        score_matrix[test_idx] = scores[:, :n_labels]

    return {
        "micro_aupr":  safe_micro_aupr(y, score_matrix),
        "macro_aupr":  safe_macro_aupr(y, score_matrix),
        "micro_auroc": safe_micro_auroc(y, score_matrix),
        "macro_auroc": safe_macro_auroc(y, score_matrix),
    }


# ── Main screen loop ──────────────────────────────────────────────────────────

def screen_family_task(
    feature_type: str,
    family: str,
    task_name: str,
    dataset_dir: Path,
    embedding_dir: Optional[Path],
    embedding_glob: str,
    run_dir: Path,
    cv_mode: str,
    n_splits: int,
    min_positive_count: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    print(f"\n{'='*60}")
    print(f"  {family} | {task_name} | {feature_type}")
    print(f"{'='*60}")

    if feature_type == "function_logits":
        X, cache = load_fl_features(dataset_dir)
    else:
        X, cache = load_ll_features(dataset_dir, embedding_dir, embedding_glob)

    y_raw, label_names_raw = get_task_definition(cache, task_name)
    y, label_names, _ = filter_labels_by_support(
        y_raw, label_names_raw, min_positive_count=min_positive_count
    )
    if y.shape[1] == 0:
        raise ValueError(f"No labels remain for {family} {task_name}")

    cv_splits = build_cv_splits(
        num_samples=len(X), cv_mode=cv_mode,
        n_splits=n_splits, random_seed=random_seed,
    )

    print(f"  X shape      : {X.shape}")
    print(f"  Labels       : {y.shape[1]}")
    print(f"  CV splits    : {len(cv_splits)}")
    print()

    classifiers = build_classifier_list(random_seed)
    rows: List[Dict[str, Any]] = []

    for name, base_clf in classifiers:
        print(f"  [{name}]", end=" ", flush=True)
        try:
            metrics = run_cv_for_classifier(
                name, base_clf, X, y, cv_splits, random_seed
            )
            micro_aupr = metrics["micro_aupr"]
            print(f"micro_aupr={micro_aupr:.4f}" if micro_aupr is not None else "micro_aupr=None")
        except Exception as exc:
            metrics = {k: None for k in ("micro_aupr", "macro_aupr", "micro_auroc", "macro_auroc")}
            print(f"FAILED — {exc}")
            traceback.print_exc()

        rows.append({
            "classifier": name,
            "family": family,
            "task": task_name,
            "feature_type": feature_type,
            **metrics,
        })

    rows.sort(key=lambda r: r["micro_aupr"] if r["micro_aupr"] is not None else -1, reverse=True)

    print(f"\n  {'Rank':<4} {'Classifier':<30} {'Micro AUPR':>10} {'Macro AUPR':>10} {'Micro AUROC':>11}")
    print(f"  {'-'*67}")
    for rank, row in enumerate(rows, 1):
        ma = f"{row['micro_aupr']:.4f}"  if row['micro_aupr']  is not None else "   N/A"
        ma2= f"{row['macro_aupr']:.4f}"  if row['macro_aupr']  is not None else "   N/A"
        ro = f"{row['micro_auroc']:.4f}" if row['micro_auroc'] is not None else "   N/A"
        print(f"  {rank:<4} {row['classifier']:<30} {ma:>10} {ma2:>10} {ro:>11}")

    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def families_from_arg(arg: str) -> List[str]:
    return ["BAHD", "UGT"] if arg.upper() == "BOTH" else [arg.upper()]


def tasks_from_arg(arg: str) -> List[str]:
    return ["donor", "acceptor"] if arg.lower() == "both" else [arg.lower()]


def main() -> None:
    args = parse_args()

    run_name = args.run_name or dt.datetime.now().strftime(f"screen_{args.feature_type}_%Y%m%d_%H%M%S")
    run_dir  = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs   = {"BAHD": Path(args.bahd_dataset_dir), "UGT": Path(args.ugt_dataset_dir)}
    embedding_dirs = {"BAHD": Path(args.bahd_embedding_dir), "UGT": Path(args.ugt_embedding_dir)}

    all_rows: List[Dict[str, Any]] = []

    for family in families_from_arg(args.dataset):
        for task in tasks_from_arg(args.task):
            rows = screen_family_task(
                feature_type      = args.feature_type,
                family            = family,
                task_name         = task,
                dataset_dir       = dataset_dirs[family],
                embedding_dir     = embedding_dirs[family] if args.feature_type == "last_layer" else None,
                embedding_glob    = args.embedding_glob,
                run_dir           = run_dir,
                cv_mode           = args.cv_mode,
                n_splits          = args.n_splits,
                min_positive_count= args.min_positive_count,
                random_seed       = args.random_seed,
            )
            all_rows.extend(rows)

    # Write combined TSV
    import pandas as pd
    results_df = pd.DataFrame(all_rows)
    tsv_path   = run_dir / "screen_results.tsv"
    results_df.to_csv(tsv_path, sep="\t", index=False)

    # Write summary JSON
    write_json(run_dir / "screen_summary.json", {
        "feature_type":       args.feature_type,
        "dataset":            args.dataset,
        "task":               args.task,
        "cv_mode":            args.cv_mode,
        "n_splits":           args.n_splits,
        "min_positive_count": args.min_positive_count,
        "random_seed":        args.random_seed,
        "num_classifiers":    len(build_classifier_list(args.random_seed)),
        "results_path":       str(tsv_path),
    })

    print(f"\n[DONE] Results written to {run_dir}")


if __name__ == "__main__":
    main()
