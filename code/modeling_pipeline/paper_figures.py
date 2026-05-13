"""Shared plotting and data-loading utilities for BAHD paper figures."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


METRIC_COLS   = ["micro_aupr", "macro_aupr", "micro_auroc", "macro_auroc"]
METRIC_LABELS = ["Micro AUPR", "Macro AUPR", "Micro AUROC", "Macro AUROC"]
METRIC_COLORS = ["#1565C0", "#42A5F5", "#E65100", "#FFA726"]


def load_random_search_results(
    results_dir: Path,
    model_name: str,
    dataset: str = "BAHD",
) -> pd.DataFrame:
    """
    Scan results/<model_name>/ and return one row per run.
    Only includes runs that have both donor and acceptor metrics files.
    """
    model_dir = Path(results_dir) / model_name
    prefix = dataset.lower()
    rows = []

    if not model_dir.exists():
        return pd.DataFrame()

    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        donor_path    = run_dir / f"{prefix}_donor_metrics.json"
        acceptor_path = run_dir / f"{prefix}_acceptor_metrics.json"
        if not donor_path.exists() or not acceptor_path.exists():
            continue

        row = {"run_tag": run_dir.name, "model": model_name}

        for task, path in [("donor", donor_path), ("acceptor", acceptor_path)]:
            with path.open() as f:
                data = json.load(f)
            m = data.get("metrics", {})
            for metric in METRIC_COLS:
                row[f"{task}_{metric}"] = m.get(metric)
            if task == "acceptor":
                row["params"] = (
                    data.get("logistic_params")
                    or data.get("ridge_params")
                    or data.get("mlp_params")
                    or data.get("transformer_params")
                    or data.get("decoder_params")
                    or data.get("svc_params")
                    or data.get("model_params")
                    or {}
                )

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def find_best_run(df: pd.DataFrame, key: str = "acceptor_micro_aupr") -> pd.Series:
    """Return the row with the highest value for the given column."""
    return df.loc[df[key].idxmax()]


def plot_search_summary(
    df: pd.DataFrame,
    metric: str = "acceptor_micro_aupr",
    title: str = "",
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """Bar chart of one metric across all random search runs. Best run highlighted."""
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    best_idx = int(df[metric].idxmax())
    colors = ["#E53935" if i == best_idx else "#1E88E5" for i in range(len(df))]
    run_nums = df["run_tag"].str.extract(r"run(\d+)")[0].fillna("?").tolist()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(df)), df[metric].values, color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"run{r}" for r in run_nums], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_title(title or metric)
    ax.axhline(df[metric].max(), color="#E53935", linestyle="--", linewidth=0.8, alpha=0.5)

    best_val = df[metric].iloc[best_idx]
    ax.legend(
        handles=[mpatches.Patch(color="#E53935",
                                label=f"Best: run{run_nums[best_idx]}  ({best_val:.4f})")],
        fontsize=9, loc="lower right",
    )
    plt.tight_layout()
    return fig


def plot_best_run_metrics(
    donor_vals: dict,
    acceptor_vals: dict,
    title: str = "",
    figsize: tuple = (9, 4),
    ylim: tuple = (0.0, 1.05),
) -> plt.Figure:
    """
    2-panel bar chart (Donor | Acceptor), 4 metrics each.
    donor_vals / acceptor_vals are dicts keyed by METRIC_COLS strings.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, task_label, vals in zip(axes, ["Donor", "Acceptor"], [donor_vals, acceptor_vals]):
        values = [vals.get(m) for m in METRIC_COLS]
        bars = ax.bar(METRIC_LABELS, values, color=METRIC_COLORS)
        ax.set_ylim(ylim)
        ax.set_title(task_label, fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            if val is not None and np.isfinite(float(val)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(float(val) + 0.015, ylim[1] - 0.03),
                    f"{float(val):.3f}",
                    ha="center", va="bottom", fontsize=9,
                )

    plt.tight_layout()
    return fig


def plot_hyperparam_scatter(
    df: pd.DataFrame,
    param_keys: list,
    metric: str = "acceptor_micro_aupr",
    log_scale_params: list = None,
    categorical_params: list = None,
    title: str = "",
    ncols: int = 3,
    subplot_size: tuple = (4.5, 3.5),
) -> plt.Figure:
    """
    One subplot per hyperparameter.
    - Continuous params: scatter + linear trendline (r, p-value in legend).
      Pass param name in log_scale_params to fit on log10(x).
    - Categorical params: box-and-whisker with jittered points overlaid.
      Pass param name in categorical_params.

    Works for all model types (LR, MLP, SFD, TFM) — caller specifies
    which keys exist and how to treat them.
    """
    log_scale_params   = log_scale_params   or []
    categorical_params = categorical_params or []

    param_df = df["params"].apply(pd.Series)
    y        = df[metric].values.astype(float)

    n       = len(param_keys)
    nrows   = (n + ncols - 1) // ncols
    figsize = (subplot_size[0] * ncols, subplot_size[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    rng  = np.random.default_rng(0)

    for ax, key in zip(axes, param_keys):
        if key not in param_df.columns:
            ax.set_visible(False)
            continue

        raw = param_df[key]

        if key in categorical_params:
            categories = sorted(raw.dropna().unique(), key=str)
            groups     = [y[raw == c] for c in categories]
            ax.boxplot(
                groups, patch_artist=True, widths=0.4,
                boxprops=dict(facecolor="#BBDEFB", color="#1565C0"),
                medianprops=dict(color="#E53935", linewidth=2),
                whiskerprops=dict(color="#1565C0"),
                capprops=dict(color="#1565C0"),
                flierprops=dict(marker=""),
            )
            for i, group in enumerate(groups):
                jitter = rng.uniform(-0.08, 0.08, size=len(group))
                ax.scatter(np.full(len(group), i + 1) + jitter, group,
                           alpha=0.7, s=40, zorder=3, color="#1E88E5")
            ax.set_xticks(range(1, len(categories) + 1))
            ax.set_xticklabels(categories, fontsize=9)

        else:
            x_num   = pd.to_numeric(raw, errors="coerce").values.astype(float)
            use_log = key in log_scale_params
            x_fit   = np.log10(x_num) if use_log else x_num

            ax.scatter(x_num, y, alpha=0.7, s=40, zorder=3, color="#1E88E5")
            if use_log:
                ax.set_xscale("log")

            valid = ~np.isnan(x_fit) & ~np.isnan(y)
            if valid.sum() >= 3:
                slope, intercept, r, p, _ = stats.linregress(x_fit[valid], y[valid])
                x_line = np.linspace(x_fit[valid].min(), x_fit[valid].max(), 200)
                y_line = slope * x_line + intercept
                x_plot = 10 ** x_line if use_log else x_line
                ax.plot(x_plot, y_line, color="#E53935", linewidth=1.5, zorder=2,
                        label=f"r={r:.2f}, p={p:.2f}")
                ax.legend(fontsize=8, loc="best")

        ax.set_xlabel(key.replace("_", " "), fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=8)
        ax.set_ylim(bottom=max(0, y.min() - 0.05))
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    return fig


def plot_hyperparam_interaction(
    df: pd.DataFrame,
    x_key: str,
    y_key: str,
    metric: str = "acceptor_micro_aupr",
    x_log: bool = False,
    y_log: bool = False,
    x_categorical: bool = False,
    y_categorical: bool = False,
    title: str = "",
    figsize: tuple = (7, 3.5),
) -> plt.Figure:
    """
    2D scatter: x_key vs y_key, dots colored by metric value.
    Categorical axes get jitter and labelled ticks.
    Log axes are set to log scale (log10 assumed).
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    param_df = df["params"].apply(pd.Series)
    z        = df[metric].values.astype(float)
    rng      = np.random.default_rng(0)

    def resolve_axis(key, is_cat, is_log, param_df):
        raw = param_df[key]
        if is_cat:
            categories = sorted(raw.dropna().unique(), key=str)
            cat_map    = {c: i for i, c in enumerate(categories)}
            vals       = raw.map(cat_map).values.astype(float)
            jitter     = rng.uniform(-0.08, 0.08, size=len(vals))
            return vals + jitter, categories
        else:
            vals = pd.to_numeric(raw, errors="coerce").values.astype(float)
            return vals, None

    x_vals, x_cats = resolve_axis(x_key, x_categorical, x_log, param_df)
    y_vals, y_cats = resolve_axis(y_key, y_categorical, y_log, param_df)

    norm = mcolors.Normalize(vmin=z.min(), vmax=z.max())

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x_vals, y_vals, c=z, cmap="RdYlGn", norm=norm,
                    s=80, edgecolors="grey", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label=metric.replace("_", " ").title())

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    if x_cats is not None:
        ax.set_xticks(range(len(x_cats)))
        ax.set_xticklabels(x_cats, fontsize=9)
    if y_cats is not None:
        ax.set_yticks(range(len(y_cats)))
        ax.set_yticklabels(y_cats, fontsize=9)

    x_label = x_key.replace("_", " ") + (" (log)" if x_log else "")
    y_label = y_key.replace("_", " ") + (" (log)" if y_log else "")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    ax.set_title(title or f"{x_key} vs {y_key} (color = {metric})", fontsize=11)
    plt.tight_layout()
    return fig


FL_COLOR = "#1565C0"
LL_COLOR = "#E65100"


def plot_comparison_best(
    entries: list,
    metric_base: str = "micro_aupr",
    title: str = "Figure 3 — Best Run Comparison",
    figsize: tuple = (11, 4),
) -> plt.Figure:
    """
    Bar chart comparing best-run donor and acceptor metrics across models.

    entries: list of (label, "fl"|"ll", best_run_series_or_None)
    metric_base: e.g. "micro_aupr" — prepended with "donor_" and "acceptor_"
    """
    labels       = [e[0] for e in entries]
    colors       = [FL_COLOR if e[1] == "fl" else LL_COLOR for e in entries]
    donor_col    = f"donor_{metric_base}"
    acceptor_col = f"acceptor_{metric_base}"
    donor_vals   = [e[2][donor_col]    if e[2] is not None else None for e in entries]
    acceptor_vals= [e[2][acceptor_col] if e[2] is not None else None for e in entries]

    x   = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, vals, task in zip(axes, [donor_vals, acceptor_vals], ["Donor", "Acceptor"]):
        bars = ax.bar(x, [v if v is not None else 0 for v in vals],
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val is None:
                bar.set_hatch("///")
                bar.set_alpha(0.3)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        float(val) + 0.008, f"{float(val):.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(task, fontsize=11)
        ax.set_ylabel(metric_base.replace("_", " ").title())

    axes[1].legend(handles=[
        mpatches.Patch(color=FL_COLOR, label="Function Logits (FL)"),
        mpatches.Patch(color=LL_COLOR, label="Last Layer (LL)"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    return fig


def plot_comparison_distribution(
    entries: list,
    metric_base: str = "micro_aupr",
    title: str = "Figure 3 — All Runs Distribution",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Box-and-whisker with jittered points comparing all random search runs across models.

    entries: list of (label, "fl"|"ll", df_or_None)
    metric_base: e.g. "micro_aupr" — prepended with "donor_" and "acceptor_"
    """
    labels = [e[0] for e in entries]
    colors = [FL_COLOR if e[1] == "fl" else LL_COLOR for e in entries]
    rng    = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, task in zip(axes, ["donor", "acceptor"]):
        metric_col = f"{task}_{metric_base}"
        for i, ((_, _, df), color) in enumerate(zip(entries, colors)):
            if df is None or metric_col not in df.columns:
                continue
            group = df[metric_col].dropna().values
            if len(group) == 0:
                continue
            ax.boxplot(
                group, positions=[i + 1], widths=0.35, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.35, color=color),
                medianprops=dict(color=color, linewidth=2),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker=""),
            )
            jitter = rng.uniform(-0.08, 0.08, size=len(group))
            ax.scatter(np.full(len(group), i + 1) + jitter, group,
                       color=color, alpha=0.7, s=30, zorder=3)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(task.capitalize(), fontsize=11)
        ax.set_ylabel(metric_base.replace("_", " ").title())

    axes[1].legend(handles=[
        mpatches.Patch(color=FL_COLOR, label="Function Logits (FL)"),
        mpatches.Patch(color=LL_COLOR, label="Last Layer (LL)"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    return fig


def plot_final_comparison(
    ridge_5fold: dict,
    ridge_loocv: dict,
    lr_5fold: dict,
    lr_loocv: dict,
    title: str = "Figure 4 — Final Model Comparison",
    figsize: tuple = (14, 5),
    ylim: tuple = (0.0, 1.05),
) -> plt.Figure:
    """
    2-panel bar chart (Donor | Acceptor) comparing final Ridge-FL and LR-LL models.

    Each panel has 4 x-positions (Micro AUPR, Macro AUPR, Micro AUROC, Macro AUROC)
    and 4 grouped bars per position:
        Ridge 5-fold  (dark blue)
        Ridge LOOCV   (light blue)
        LR 5-fold     (dark orange)
        LR LOOCV      (light orange)

    5-fold bars include ±1 std error bars computed across per-fold metrics.
    LOOCV bars have no error bars.

    Parameters
    ----------
    ridge_5fold / ridge_loocv / lr_5fold / lr_loocv : dict
        Shape: {"donor": <metrics dict>, "acceptor": <metrics dict>}
        Each metrics dict has keys: micro_aupr, macro_aupr, micro_auroc,
        macro_auroc, per_fold_metrics (list of per-fold dicts, 5-fold only).
    """
    BAR_COLORS = {
        "ridge_5fold": "#1565C0",
        "ridge_loocv": "#42A5F5",
        "lr_5fold":    "#E65100",
        "lr_loocv":    "#FFA726",
    }
    BAR_LABELS = {
        "ridge_5fold": "Ridge 5-fold",
        "ridge_loocv": "Ridge LOOCV",
        "lr_5fold":    "LR 5-fold",
        "lr_loocv":    "LR LOOCV",
    }
    bar_order = ["ridge_5fold", "ridge_loocv", "lr_5fold", "lr_loocv"]
    sources   = {
        "ridge_5fold": ridge_5fold,
        "ridge_loocv": ridge_loocv,
        "lr_5fold":    lr_5fold,
        "lr_loocv":    lr_loocv,
    }

    n_metrics = len(METRIC_COLS)
    n_bars    = len(bar_order)
    width     = 0.18
    x         = np.arange(n_metrics)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, task in zip(axes, ["donor", "acceptor"]):
        for bar_i, key in enumerate(bar_order):
            task_m = sources[key].get(task, {})
            vals   = [task_m.get(m) for m in METRIC_COLS]
            color  = BAR_COLORS[key]

            # per-fold std for 5-fold bars only
            errs = [None] * n_metrics
            if "5fold" in key:
                pfm = task_m.get("per_fold_metrics", [])
                if pfm:
                    for mi, m in enumerate(METRIC_COLS):
                        fold_vals = [f[m] for f in pfm if f.get(m) is not None]
                        if len(fold_vals) >= 2:
                            errs[mi] = float(np.std(fold_vals, ddof=1))

            offset = (bar_i - (n_bars - 1) / 2) * width
            for mi in range(n_metrics):
                val = vals[mi]
                err = errs[mi]
                height = float(val) if val is not None else 0.0
                ax.bar(
                    x[mi] + offset, height,
                    width=width, color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.4,
                )
                if val is not None and err is not None:
                    ax.errorbar(
                        x[mi] + offset, height, yerr=err,
                        fmt="none", ecolor="black",
                        elinewidth=1.2, capsize=3, zorder=5,
                    )
                if val is not None:
                    label_y = height + (err if err else 0) + 0.015
                    ax.text(
                        x[mi] + offset, min(label_y, ylim[1] - 0.02),
                        f"{height:.3f}",
                        ha="center", va="bottom",
                        fontsize=7, rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_LABELS, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(ylim)
        ax.set_title(task.capitalize(), fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].legend(
        handles=[
            mpatches.Patch(color=BAR_COLORS[k], label=BAR_LABELS[k])
            for k in bar_order
        ],
        fontsize=9, loc="lower right",
    )

    plt.tight_layout()
    return fig


def plot_ugt_generalization(
    lr_5fold: dict,
    lr_loocv: dict,
    ugt_metrics: dict,
    title: str = "Figure 5 — LR (LL) Acceptor: BAHD CV vs UGT Held-Out",
    figsize: tuple = (9, 5),
    ylim: tuple = (0.0, 1.05),
) -> plt.Figure:
    """
    Single-panel bar chart comparing LR (LL) acceptor performance across
    three evaluation conditions: BAHD 5-fold CV, BAHD LOOCV, and UGT held-out.

    Parameters
    ----------
    lr_5fold / lr_loocv : dict
        Shape: {"donor": <metrics dict>, "acceptor": <metrics dict>}
        Same format as used in plot_final_comparison.
    ugt_metrics : dict
        Flat metrics dict with keys micro_aupr, macro_aupr, micro_auroc,
        macro_auroc (no donor/acceptor split — UGT is acceptor-only).
    """
    BAR_COLORS = {
        "lr_5fold": "#E65100",
        "lr_loocv": "#FFA726",
        "ugt":      "#6A1B9A",
    }
    BAR_LABELS = {
        "lr_5fold": "LR BAHD 5-fold",
        "lr_loocv": "LR BAHD LOOCV",
        "ugt":      "LR UGT held-out",
    }
    bar_order = ["lr_5fold", "lr_loocv", "ugt"]

    sources = {
        "lr_5fold": lr_5fold.get("acceptor", {}),
        "lr_loocv": lr_loocv.get("acceptor", {}),
        "ugt":      ugt_metrics,
    }

    n_metrics = len(METRIC_COLS)
    n_bars    = len(bar_order)
    width     = 0.22
    x         = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for bar_i, key in enumerate(bar_order):
        task_m = sources[key]
        vals   = [task_m.get(m) for m in METRIC_COLS]
        color  = BAR_COLORS[key]

        errs = [None] * n_metrics
        if key == "lr_5fold":
            pfm = task_m.get("per_fold_metrics", [])
            if pfm:
                for mi, m in enumerate(METRIC_COLS):
                    fold_vals = [f[m] for f in pfm if f.get(m) is not None]
                    if len(fold_vals) >= 2:
                        errs[mi] = float(np.std(fold_vals, ddof=1))

        offset = (bar_i - (n_bars - 1) / 2) * width
        for mi in range(n_metrics):
            val    = vals[mi]
            err    = errs[mi]
            height = float(val) if val is not None else 0.0
            ax.bar(
                x[mi] + offset, height,
                width=width, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4,
            )
            if val is not None and err is not None:
                ax.errorbar(
                    x[mi] + offset, height, yerr=err,
                    fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=3, zorder=5,
                )
            if val is not None:
                label_y = height + (err if err else 0) + 0.015
                ax.text(
                    x[mi] + offset, min(label_y, ylim[1] - 0.02),
                    f"{height:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(ylim)
    ax.set_ylabel("Score")
    ax.set_title("Acceptor", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        handles=[
            mpatches.Patch(color=BAR_COLORS[k], label=BAR_LABELS[k])
            for k in bar_order
        ],
        fontsize=9, loc="upper left",
    )

    plt.tight_layout()
    return fig
