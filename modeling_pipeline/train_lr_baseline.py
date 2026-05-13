#!/usr/bin/env python3
"""
Simple logistic-regression baseline for BAHD and UGT datasets.

The script uses pooled ESM function logits only:
  [1, residues, steps, channels] -> mean over residues -> flattened vector

Supported tasks:
  - BAHD donor type
  - UGT donor compound
  - BAHD/UGT acceptor superclass
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

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


def validate_dataset_dir(path: Path, family: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{family} dataset directory not found: {path}")
    labels_path = path / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"{family} labels cache not found: {labels_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple LOOCV logistic-regression baseline.")
    parser.add_argument(
        "--dataset",
        choices=("BAHD", "UGT", "both"),
        default="both",
        help="Which dataset family to run.",
    )
    parser.add_argument(
        "--bahd_dataset_dir",
        default=str(dataset_dir_for_family("BAHD")),
        help="Path to compact BAHD dataset directory.",
    )
    parser.add_argument(
        "--ugt_dataset_dir",
        default=str(dataset_dir_for_family("UGT")),
        help="Path to compact UGT dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "results" / "lr_baseline"),
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
        help="Drop labels with fewer than this many positives before LOOCV.",
    )
    parser.add_argument(
        "--write_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sequence LOOCV predictions as TSV.",
    )
    parser.add_argument(
        "--pickle_models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit models on the full dataset after LOOCV and pickle them.",
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
        default=0.1,
        help="Regularization strength for logistic regression (inverse of lambda).",
    )
    parser.add_argument(
        "--penalty",
        choices=("l1", "l2"),
        default="l2",
        help="Penalty norm for logistic regression.",
    )
    return parser.parse_args()


def families_from_arg(dataset_arg: str) -> List[str]:
    value = str(dataset_arg).upper()
    if value == "BOTH":
        return ["BAHD", "UGT"]
    return [value]


def make_pipeline(params: dict | None = None) -> Pipeline:
    resolved = {**DEFAULT_LOGISTIC_PARAMS, **(params or {})}
    clf = LogisticRegression(**resolved)
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


def fit_binary_model(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> Dict[str, Any]:
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
    n_samples, n_labels = y.shape
    pred_matrix = np.zeros((n_samples, n_labels), dtype=np.uint8)
    score_matrix = np.zeros((n_samples, n_labels), dtype=np.float32)

    for train_idx, test_idx in cv_splits:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]

        for label_idx in range(n_labels):
            model = fit_binary_model(X_train, y_train[:, label_idx], params=logistic_params)
            pred, score = score_binary_model(model, X_test)
            pred_matrix[test_idx, label_idx] = pred
            score_matrix[test_idx, label_idx] = score

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
                "label": label_name,
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
            "num_samples": int(n_samples),
            "num_labels": int(n_labels),
            "cv_mode": str(cv_mode),
            "num_folds": int(len(cv_splits)),
            "micro_aupr": micro_aupr,
            "micro_auroc": micro_auroc,
            "macro_aupr": float(np.mean(aupr_values)) if aupr_values else None,
            "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
            "per_label": per_label_metrics,
        },
    }


def fit_full_models(X: np.ndarray, y: np.ndarray, label_names: Sequence[str],
                    logistic_params: dict | None = None) -> Dict[str, Any]:
    resolved = {**DEFAULT_LOGISTIC_PARAMS, **(logistic_params or {})}
    models: Dict[str, Any] = {}
    for label_idx, label_name in enumerate(label_names):
        models[str(label_name)] = fit_binary_model(X, y[:, label_idx], params=resolved)
    return {
        "label_names": list(label_names),
        "logistic_params": resolved,
        "models": models,
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
        header = [
            "fasta_id",
            "enzyme_id",
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
                true_labels,
                pred_labels,
                true_vector,
                pred_vector,
                score_vector,
            ]
            handle.write("\t".join(row) + "\n")


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
    write_predictions: bool,
    pickle_models: bool,
    C: float,
    penalty: str,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    cache = load_dataset_cache(dataset_dir)
    resolved_device = get_device(device)
    X = build_feature_matrix(dataset_dir, cache, device=resolved_device)
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

    cv_splits = build_cv_splits(
        num_samples=X.shape[0],
        cv_mode=cv_mode,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    logistic_params = {**DEFAULT_LOGISTIC_PARAMS, "C": C, "penalty": penalty}
    result = run_cv_multilabel(X, y, label_names, cv_splits=cv_splits, cv_mode=cv_mode,
                               logistic_params=logistic_params)
    metrics_payload = {
        "family": family,
        "task": task_name,
        "dataset_dir": str(dataset_dir),
        "device": str(resolved_device),
        "cv_mode": cv_mode,
        "n_splits": int(n_splits) if cv_mode == "kfold" else None,
        "random_seed": int(random_seed),
        "feature_type": "function_logits_structure_steps10_mean_pool",
        "min_positive_count": int(min_positive_count),
        "pooling": "mean",
        "logistic_params": logistic_params,
        "kept_labels": list(label_names),
        "kept_label_positive_counts": kept_counts,
        "metrics": result["metrics"],
    }

    stem = f"{family.lower()}_{task_name}"
    write_json(run_dir / f"{stem}_metrics.json", metrics_payload)

    if write_predictions:
        write_predictions_tsv(
            run_dir / f"{stem}_predictions.tsv",
            cache["fasta_ids"],
            cache["enzyme_ids"],
            y,
            result["pred_matrix"],
            result["score_matrix"],
            label_names,
        )

    if pickle_models:
        model_bundle = fit_full_models(X, y, label_names, logistic_params=logistic_params)
        model_path = run_dir / f"{stem}_models.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model_bundle, handle)

    return metrics_payload


def main() -> None:
    args = parse_args()
    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = {
        "BAHD": Path(args.bahd_dataset_dir),
        "UGT": Path(args.ugt_dataset_dir),
    }
    for family, dataset_dir in dataset_dirs.items():
        if args.dataset in ("both", family):
            validate_dataset_dir(dataset_dir, family)

    all_results: List[Dict[str, Any]] = []
    for family in families_from_arg(args.dataset):
        donor_task = "donor"
        acceptor_task = "acceptor"
        all_results.append(
            run_family_task(
                family=family,
                task_name=donor_task,
                dataset_dir=dataset_dirs[family],
                run_dir=run_dir,
                device=args.device,
                cv_mode=args.cv_mode,
                n_splits=args.n_splits,
                random_seed=args.random_seed,
                min_positive_count=args.min_positive_count,
                write_predictions=args.write_predictions,
                pickle_models=args.pickle_models,
                C=args.C,
                penalty=args.penalty,
            )
        )
        all_results.append(
            run_family_task(
                family=family,
                task_name=acceptor_task,
                dataset_dir=dataset_dirs[family],
                run_dir=run_dir,
                device=args.device,
                cv_mode=args.cv_mode,
                n_splits=args.n_splits,
                random_seed=args.random_seed,
                min_positive_count=args.min_positive_count,
                write_predictions=args.write_predictions,
                pickle_models=args.pickle_models,
                C=args.C,
                penalty=args.penalty,
            )
        )

    write_json(
        run_dir / "run_config.json",
        {
            "dataset": args.dataset,
            "bahd_dataset_dir": args.bahd_dataset_dir,
            "ugt_dataset_dir": args.ugt_dataset_dir,
            "output_dir": str(Path(args.output_dir)),
            "run_name": run_name,
            "device": str(get_device(args.device)),
            "cv_mode": args.cv_mode,
            "n_splits": args.n_splits if args.cv_mode == "kfold" else None,
            "random_seed": args.random_seed,
            "min_positive_count": args.min_positive_count,
            "write_predictions": args.write_predictions,
            "pickle_models": args.pickle_models,
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
