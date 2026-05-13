#!/usr/bin/env python3
"""
RidgeClassifier baseline for BAHD and UGT datasets.

Uses the same mean-pooled ESM3 function logits as train_lr_baseline.py,
with RidgeClassifier (L2 squared-loss) instead of LogisticRegression.
RidgeClassifier is fit per-label (one-vs-rest) and scored via decision_function.

Outputs are written under:
    results/ridge_baseline/<run_name>/
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    build_cv_splits,
    build_feature_matrix,
    dataset_dir_for_family,
    filter_labels_by_support,
    get_device,
    get_task_definition,
    label_indices_to_names,
    load_dataset_cache,
    write_json,
)


DEFAULT_RIDGE_PARAMS: Dict[str, Any] = {
    "alpha": 1.0,
    "fit_intercept": True,
    "class_weight": "balanced",
}

SCRIPT_DIR = Path(__file__).resolve().parent


def validate_dataset_dir(path: Path, family: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{family} dataset directory not found: {path}")
    if not (path / "labels.pt").exists():
        raise FileNotFoundError(f"{family} labels cache not found: {path / 'labels.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RidgeClassifier baseline on ESM3 function logits.")
    parser.add_argument("--dataset", choices=("BAHD", "UGT", "both"), default="both")
    parser.add_argument("--bahd_dataset_dir", default=str(dataset_dir_for_family("BAHD")))
    parser.add_argument("--ugt_dataset_dir",  default=str(dataset_dir_for_family("UGT")))
    parser.add_argument("--output_dir",
                        default=str(SCRIPT_DIR / "results" / "ridge_baseline"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--cv_mode", choices=("kfold", "loocv"), default="kfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min_positive_count", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength. Higher = more regularization.")
    parser.add_argument("--write_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pickle_models",     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_seed", type=int, default=1)
    return parser.parse_args()


def families_from_arg(dataset_arg: str) -> List[str]:
    return ["BAHD", "UGT"] if dataset_arg.upper() == "BOTH" else [dataset_arg.upper()]


def make_pipeline(params: dict | None = None) -> Pipeline:
    p = {**DEFAULT_RIDGE_PARAMS, **(params or {})}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RidgeClassifier(
            alpha=p["alpha"],
            fit_intercept=p["fit_intercept"],
            class_weight=p["class_weight"],
        )),
    ])


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.sum() == 0:
        return None
    val = float(average_precision_score(y_true, y_score))
    return val if np.isfinite(val) else None


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    val = float(roc_auc_score(y_true, y_score))
    return val if np.isfinite(val) else None


def fit_binary_model(X: np.ndarray, y: np.ndarray,
                     params: dict | None = None) -> Dict[str, Any]:
    classes = np.unique(y)
    if classes.size < 2:
        return {"mode": "constant", "constant": int(classes[0]) if classes.size == 1 else 0,
                "pipeline": None}
    pipeline = make_pipeline(params)
    pipeline.fit(X, y)
    return {"mode": "trained", "constant": None, "pipeline": pipeline}


def score_binary_model(model: Dict[str, Any],
                       X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if model["mode"] == "constant":
        c = int(model["constant"])
        return (np.full(X_test.shape[0], c, dtype=np.uint8),
                np.full(X_test.shape[0], float(c), dtype=np.float32))
    pipeline = model["pipeline"]
    pred  = pipeline.predict(X_test).astype(np.uint8, copy=False)
    score = pipeline.decision_function(X_test).astype(np.float32, copy=False)
    return pred, score


def run_cv_multilabel(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cv_mode: str,
    ridge_params: dict | None = None,
) -> Dict[str, Any]:
    n_samples, n_labels = y.shape
    pred_matrix  = np.zeros((n_samples, n_labels), dtype=np.uint8)
    score_matrix = np.zeros((n_samples, n_labels), dtype=np.float32)

    per_fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        fold_score_buffer = np.zeros((len(test_idx), n_labels), dtype=np.float32)
        for label_idx in range(n_labels):
            model = fit_binary_model(X[train_idx], y[train_idx, label_idx], ridge_params)
            pred, score = score_binary_model(model, X[test_idx])
            pred_matrix[test_idx, label_idx]   = pred
            score_matrix[test_idx, label_idx]  = score
            fold_score_buffer[:, label_idx]    = score

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
    aupr_values, auroc_values = [], []
    positive_counts = y.sum(axis=0).astype(int)

    for label_idx, label_name in enumerate(label_names):
        y_true, y_score = y[:, label_idx], score_matrix[:, label_idx]
        label_aupr  = safe_average_precision(y_true, y_score)
        label_auroc = safe_roc_auc(y_true, y_score)
        if label_aupr  is not None: aupr_values.append(label_aupr)
        if label_auroc is not None: auroc_values.append(label_auroc)
        per_label_metrics.append({
            "label": label_name,
            "positive_count": int(positive_counts[label_idx]),
            "aupr": label_aupr,
            "auroc": label_auroc,
        })

    return {
        "pred_matrix":  pred_matrix,
        "score_matrix": score_matrix,
        "metrics": {
            "num_samples":       int(n_samples),
            "num_labels":        int(n_labels),
            "cv_mode":           str(cv_mode),
            "num_folds":         int(len(cv_splits)),
            "micro_aupr":        safe_average_precision(y.reshape(-1), score_matrix.reshape(-1)),
            "micro_auroc":       safe_roc_auc(y.reshape(-1), score_matrix.reshape(-1)),
            "macro_aupr":        float(np.mean(aupr_values))  if aupr_values  else None,
            "macro_auroc":       float(np.mean(auroc_values)) if auroc_values else None,
            "per_label":         per_label_metrics,
            "per_fold_metrics":  per_fold_metrics,
        },
    }


def fit_full_models(X: np.ndarray, y: np.ndarray,
                    label_names: Sequence[str],
                    ridge_params: dict | None = None) -> Dict[str, Any]:
    return {
        "label_names":  list(label_names),
        "ridge_params": ridge_params or DEFAULT_RIDGE_PARAMS,
        "models": {
            str(label_names[i]): fit_binary_model(X, y[:, i], ridge_params)
            for i in range(y.shape[1])
        },
    }


def write_predictions_tsv(
    path: str | Path,
    fasta_ids: Sequence[str],
    enzyme_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join([
            "fasta_id", "enzyme_id", "true_labels", "predicted_labels",
            "true_vector", "pred_vector", "score_vector",
        ]) + "\n")
        for idx, fasta_id in enumerate(fasta_ids):
            handle.write("\t".join([
                str(fasta_id),
                str(enzyme_ids[idx]),
                ";".join(label_indices_to_names(y_true[idx], label_names)),
                ";".join(label_indices_to_names(y_pred[idx], label_names)),
                ",".join(str(int(v)) for v in y_true[idx].tolist()),
                ",".join(str(int(v)) for v in y_pred[idx].tolist()),
                ",".join(f"{float(v):.6f}" for v in y_score[idx].tolist()),
            ]) + "\n")


def run_family_task(
    family: str,
    task_name: str,
    dataset_dir: str | Path,
    run_dir: Path,
    device: str,
    cv_mode: str,
    n_splits: int,
    random_seed: int,
    min_positive_count: int,
    alpha: float,
    write_predictions: bool,
    pickle_models: bool,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    cache = load_dataset_cache(dataset_dir)
    resolved_device = get_device(device)
    X = build_feature_matrix(dataset_dir, cache, device=resolved_device)
    y_raw, label_names_raw = get_task_definition(cache, task_name)
    y, label_names, kept_counts = filter_labels_by_support(
        y_raw, label_names_raw, min_positive_count=min_positive_count
    )
    if y.shape[1] == 0:
        raise ValueError(f"No labels remain for family={family} task={task_name}")

    ridge_params = {**DEFAULT_RIDGE_PARAMS, "alpha": alpha}

    cv_splits = build_cv_splits(
        num_samples=X.shape[0], cv_mode=cv_mode,
        n_splits=n_splits, random_seed=random_seed,
    )
    result = run_cv_multilabel(X, y, label_names, cv_splits, cv_mode, ridge_params)

    metrics_payload = {
        "family":              family,
        "task":                task_name,
        "dataset_dir":         str(dataset_dir),
        "device":              str(resolved_device),
        "cv_mode":             cv_mode,
        "n_splits":            int(n_splits) if cv_mode == "kfold" else None,
        "random_seed":         int(random_seed),
        "feature_type":        "function_logits_structure_steps10_mean_pool",
        "min_positive_count":  int(min_positive_count),
        "ridge_params":        ridge_params,
        "kept_labels":         list(label_names),
        "kept_label_positive_counts": kept_counts,
        "metrics":             result["metrics"],
    }

    stem = f"{family.lower()}_{task_name}"
    write_json(run_dir / f"{stem}_metrics.json", metrics_payload)

    if write_predictions:
        write_predictions_tsv(
            run_dir / f"{stem}_predictions.tsv",
            cache["fasta_ids"], cache["enzyme_ids"],
            y, result["pred_matrix"], result["score_matrix"], label_names,
        )

    if pickle_models:
        bundle = fit_full_models(X, y, label_names, ridge_params)
        with (run_dir / f"{stem}_models.pkl").open("wb") as f:
            pickle.dump(bundle, f)

    return metrics_payload


def main() -> None:
    args = parse_args()
    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = {
        "BAHD": Path(args.bahd_dataset_dir),
        "UGT":  Path(args.ugt_dataset_dir),
    }
    for family, dataset_dir in dataset_dirs.items():
        if args.dataset in ("both", family):
            validate_dataset_dir(dataset_dir, family)

    all_results: List[Dict[str, Any]] = []
    for family in families_from_arg(args.dataset):
        for task_name in ("donor", "acceptor"):
            all_results.append(run_family_task(
                family=family, task_name=task_name,
                dataset_dir=dataset_dirs[family],
                run_dir=run_dir, device=args.device,
                cv_mode=args.cv_mode, n_splits=args.n_splits,
                random_seed=args.random_seed,
                min_positive_count=args.min_positive_count,
                alpha=args.alpha,
                write_predictions=args.write_predictions,
                pickle_models=args.pickle_models,
            ))

    write_json(run_dir / "run_config.json", {
        "dataset": args.dataset,
        "bahd_dataset_dir": args.bahd_dataset_dir,
        "ugt_dataset_dir":  args.ugt_dataset_dir,
        "output_dir":       str(Path(args.output_dir)),
        "run_name":         run_name,
        "device":           str(get_device(args.device)),
        "cv_mode":          args.cv_mode,
        "n_splits":         args.n_splits if args.cv_mode == "kfold" else None,
        "random_seed":      args.random_seed,
        "min_positive_count": args.min_positive_count,
        "ridge_params":     {**DEFAULT_RIDGE_PARAMS, "alpha": args.alpha},
        "write_predictions": args.write_predictions,
        "pickle_models":     args.pickle_models,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR running {Path(__file__).name}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
