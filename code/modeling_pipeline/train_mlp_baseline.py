#!/usr/bin/env python3
"""
Sequence-aware MLP baseline for BAHD and UGT datasets.

Inputs are the full per-residue function logits:
  [1, residues, 8, 260] -> [residues, 8, 260]

Pooling options:
  - mean: masked mean over residue features
  - attention: learned attention pooling over residues
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import (
    build_cv_splits,
    build_sequence_feature_list,
    dataset_dir_for_family,
    filter_labels_by_support,
    get_device,
    get_task_definition,
    label_indices_to_names,
    load_dataset_cache,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def validate_dataset_dir(path: Path, family: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{family} dataset directory not found: {path}")
    labels_path = path / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"{family} labels cache not found: {labels_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sequence-aware MLP baseline.")
    parser.add_argument("--dataset", choices=("BAHD", "UGT", "both"), default="both")
    parser.add_argument("--bahd_dataset_dir", default=str(dataset_dir_for_family("BAHD")))
    parser.add_argument("--ugt_dataset_dir", default=str(dataset_dir_for_family("UGT")))
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "results" / "mlp_baseline"),
        help="Directory for metrics, predictions, and optional models.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--cv_mode", choices=("kfold", "loocv"), default="kfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min_positive_count", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", choices=("mean", "attention"), default="mean")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=10) #DRH#
    parser.add_argument("--write_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_models", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[np.ndarray], labels: np.ndarray):
        self.sequences = list(sequences)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.sequences[idx], self.labels[idx]


def collate_sequences(batch: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    max_len = max(seq.shape[0] for seq in sequences)
    feat_dim = sequences[0].shape[1]
    x = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    for idx, seq in enumerate(sequences):
        length = seq.shape[0]
        x[idx, :length] = seq
        mask[idx, :length] = True
    y = np.stack(labels, axis=0).astype(np.float32, copy=False)
    return (
        torch.from_numpy(x),
        torch.from_numpy(mask),
        torch.from_numpy(y),
    )


class SequenceMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float, pooling: str) -> None:
        super().__init__()
        self.pooling = pooling
        self.residue_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if pooling == "attention":
            self.attention = nn.Linear(hidden_dim, 1)
        else:
            self.attention = None
        self.output = nn.Linear(hidden_dim, num_labels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.residue_mlp(x)
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "mean":
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        else:
            scores = self.attention(hidden).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (hidden * weights * mask_f).sum(dim=1)
        return self.output(pooled)


def families_from_arg(dataset_arg: str) -> List[str]:
    return ["BAHD", "UGT"] if str(dataset_arg).upper() == "BOTH" else [str(dataset_arg).upper()]


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def fit_sequence_scaler(train_sequences: Sequence[np.ndarray]) -> StandardScaler:
    stacked = np.concatenate(train_sequences, axis=0)
    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


def transform_sequences(sequences: Sequence[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    return [scaler.transform(seq).astype(np.float32, copy=False) for seq in sequences]


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor | None:
    positives = y_train.sum(axis=0).astype(np.float32)
    negatives = float(len(y_train)) - positives
    valid = positives > 0
    if not np.any(valid):
        return None
    weights = np.ones_like(positives, dtype=np.float32)
    weights[valid] = negatives[valid] / positives[valid]
    return torch.from_numpy(weights)


def train_multilabel_model(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    early_stopping_patience: int = 10,
) -> Dict[str, Any]:
    set_random_seed(seed)

    if early_stopping_patience > 0:
        n = len(sequences)
        rng_np = np.random.default_rng(seed)
        val_size = max(1, int(0.1 * n))
        val_idx_set = set(rng_np.choice(n, size=val_size, replace=False).tolist())
        train_idx = [i for i in range(n) if i not in val_idx_set]
        val_idx   = [i for i in range(n) if i in val_idx_set]
        train_seqs   = [sequences[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_seqs     = [sequences[i] for i in val_idx]
        val_labels   = labels[val_idx]
    else:
        train_seqs   = list(sequences)
        train_labels = labels
        val_seqs     = []
        val_labels   = np.empty((0, labels.shape[1]), dtype=np.float32)

    scaler = fit_sequence_scaler(train_seqs)
    scaled_train = transform_sequences(train_seqs, scaler)
    dataset = SequenceDataset(scaled_train, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_sequences,
    )

    if early_stopping_patience > 0:
        scaled_val = transform_sequences(val_seqs, scaler)
        val_dataset = SequenceDataset(scaled_val, val_labels)
        val_loader = DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_sequences
        )

    model = SequenceMLPClassifier(
        input_dim=scaled_train[0].shape[1],
        hidden_dim=hidden_dim,
        num_labels=labels.shape[1],
        dropout=dropout,
        pooling=pooling,
    ).to(device)
    pos_weight = compute_pos_weight(train_labels)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_mask, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        if early_stopping_patience > 0:
            model.eval()
            with torch.no_grad():
                val_x, val_mask, val_y = next(iter(val_loader))
                val_loss = criterion(
                    model(val_x.to(device), val_mask.to(device)),
                    val_y.to(device),
                ).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler_mean": scaler.mean_.astype(np.float32, copy=False),
        "scaler_scale": scaler.scale_.astype(np.float32, copy=False),
    }




def score_multilabel_model(
    trained: Dict[str, Any],
    sequences: Sequence[np.ndarray],
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    pooling: str,
    num_labels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.mean_ = trained["scaler_mean"]
    scaler.scale_ = trained["scaler_scale"]
    scaler.var_ = np.square(trained["scaler_scale"])
    scaler.n_features_in_ = int(len(trained["scaler_mean"]))
    scaled_sequences = transform_sequences(sequences, scaler)

    dataset = SequenceDataset(scaled_sequences, np.zeros((len(scaled_sequences), num_labels), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_sequences)

    model = SequenceMLPClassifier(
        input_dim=scaled_sequences[0].shape[1],
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        dropout=dropout,
        pooling=pooling,
    ).to(device)
    model.load_state_dict(trained["model_state"])
    model.eval()

    with torch.no_grad():
        batch_x, batch_mask, _ = next(iter(loader))
        logits = model(batch_x.to(device), batch_mask.to(device)).detach().cpu().numpy().astype(np.float32, copy=False)
    preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(np.uint8, copy=False)
    return preds, logits


def run_cv_multilabel(
    sequences: Sequence[np.ndarray],
    y: np.ndarray,
    label_names: Sequence[str],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cv_mode: str,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    early_stopping_patience: int = 10,
) -> Dict[str, Any]:
    n_samples, n_labels = y.shape
    pred_matrix = np.zeros((n_samples, n_labels), dtype=np.uint8)
    score_matrix = np.zeros((n_samples, n_labels), dtype=np.float32)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_sequences = [sequences[i] for i in train_idx]
        test_sequences = [sequences[i] for i in test_idx]
        trained = train_multilabel_model(
            sequences=train_sequences,
            labels=y[train_idx],
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_idx,
            early_stopping_patience=early_stopping_patience,
        )
        preds, scores = score_multilabel_model(
            trained=trained,
            sequences=test_sequences,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
            num_labels=n_labels,
        )
        pred_matrix[test_idx] = preds
        score_matrix[test_idx] = scores

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
                "label": str(label_name),
                "positive_count": int(positive_counts[label_idx]),
                "aupr": label_aupr,
                "auroc": label_auroc,
            }
        )

    return {
        "pred_matrix": pred_matrix,
        "score_matrix": score_matrix,
        "metrics": {
            "num_samples": int(n_samples),
            "num_labels": int(n_labels),
            "cv_mode": str(cv_mode),
            "num_folds": int(len(cv_splits)),
            "micro_aupr": safe_average_precision(y.reshape(-1), score_matrix.reshape(-1)),
            "micro_auroc": safe_roc_auc(y.reshape(-1), score_matrix.reshape(-1)),
            "macro_aupr": float(np.mean(aupr_values)) if aupr_values else None,
            "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
            "per_label": per_label_metrics,
        },
    }


def fit_full_model(
    sequences: Sequence[np.ndarray],
    y: np.ndarray,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    trained = train_multilabel_model(
        sequences=sequences,
        labels=y,
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pooling=pooling,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        early_stopping_patience=0,
    )
    trained["model_type"] = "sequence_mlp"
    trained["model_params"] = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "pooling": pooling,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "random_seed": seed,
    }
    return trained


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
        handle.write("\t".join(["fasta_id", "enzyme_id", "true_labels", "predicted_labels", "true_vector", "pred_vector", "score_vector"]) + "\n")
        for idx, fasta_id in enumerate(fasta_ids):
            handle.write(
                "\t".join(
                    [
                        str(fasta_id),
                        str(enzyme_ids[idx]),
                        ";".join(label_indices_to_names(y_true[idx], label_names)),
                        ";".join(label_indices_to_names(y_pred[idx], label_names)),
                        ",".join(str(int(v)) for v in y_true[idx].tolist()),
                        ",".join(str(int(v)) for v in y_pred[idx].tolist()),
                        ",".join(f"{float(v):.6f}" for v in y_score[idx].tolist()),
                    ]
                )
                + "\n"
            )


def run_family_task(
    family: str,
    task_name: str,
    dataset_dir: str | Path,
    run_dir: Path,
    device: str,
    cv_mode: str,
    n_splits: int,
    min_positive_count: int,
    hidden_dim: int,
    dropout: float,
    pooling: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    random_seed: int,
    write_predictions: bool,
    save_models: bool,
    early_stopping_patience: int = 10,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    cache = load_dataset_cache(dataset_dir)
    resolved_device = get_device(device)
    sequences = build_sequence_feature_list(dataset_dir, cache, device=resolved_device, flatten=True)
    y_raw, label_names_raw = get_task_definition(cache, task_name)
    y, label_names, kept_counts = filter_labels_by_support(y_raw, label_names_raw, min_positive_count=min_positive_count)
    if y.shape[1] == 0:
        raise ValueError(f"No labels remain for family={family} task={task_name}")

    cv_splits = build_cv_splits(num_samples=len(sequences), cv_mode=cv_mode, n_splits=n_splits, random_seed=random_seed)
    result = run_cv_multilabel(
        sequences=sequences,
        y=y,
        label_names=label_names,
        cv_splits=cv_splits,
        cv_mode=cv_mode,
        device=resolved_device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pooling=pooling,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        seed=random_seed,
        early_stopping_patience=early_stopping_patience,
    )

    metrics_payload = {
        "family": family,
        "task": task_name,
        "dataset_dir": str(dataset_dir),
        "device": str(resolved_device),
        "cv_mode": cv_mode,
        "n_splits": int(n_splits) if cv_mode == "kfold" else None,
        "feature_type": "function_logits_sequence_flattened",
        "pooling": pooling,
        "min_positive_count": int(min_positive_count),
        "mlp_params": {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "random_seed": random_seed,
            "early_stopping_patience": early_stopping_patience,
        },
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
    if save_models:
        full_model = fit_full_model(
            sequences=sequences,
            y=y,
            device=resolved_device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            seed=random_seed,
        )
        torch.save(full_model, run_dir / f"{stem}_models.pt")
    return metrics_payload


def main() -> None:
    args = parse_args()
    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = {"BAHD": Path(args.bahd_dataset_dir), "UGT": Path(args.ugt_dataset_dir)}
    for family, dataset_dir in dataset_dirs.items():
        if args.dataset in ("both", family):
            validate_dataset_dir(dataset_dir, family)
    all_results: List[Dict[str, Any]] = []
    for family in families_from_arg(args.dataset):
        for task_name in ("donor", "acceptor"):
            all_results.append(
                run_family_task(
                    family=family,
                    task_name=task_name,
                    dataset_dir=dataset_dirs[family],
                    run_dir=run_dir,
                    device=args.device,
                    cv_mode=args.cv_mode,
                    n_splits=args.n_splits,
                    min_positive_count=args.min_positive_count,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    pooling=args.pooling,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    random_seed=args.random_seed,
                    write_predictions=args.write_predictions,
                    save_models=args.save_models,
                    early_stopping_patience=args.early_stopping_patience,
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
            "min_positive_count": args.min_positive_count,
            "pooling": args.pooling,
            "mlp_params": {
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "random_seed": args.random_seed,
            },
            "write_predictions": args.write_predictions,
            "save_models": args.save_models,
            "results_written": [
                {
                    "family": item["family"],
                    "task": item["task"],
                    "metrics_file": f"{item['family'].lower()}_{item['task']}_metrics.json",
                    "predictions_file": (
                        f"{item['family'].lower()}_{item['task']}_predictions.tsv" if args.write_predictions else None
                    ),
                    "model_file": f"{item['family'].lower()}_{item['task']}_models.pt" if args.save_models else None,
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
