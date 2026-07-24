"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import TARGETS
from .evaluation import ModelResult

sns.set_theme(style="whitegrid")


def plot_parity_grid(results: List[ModelResult], y_true: np.ndarray, groups: np.ndarray, out_path: Path):
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, 2)
    group_codes = pd.Categorical(groups).codes
    for i, result in enumerate(results):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            ax.scatter(y_true[:, j], result.predictions[:, j], c=group_codes, cmap="tab10", alpha=0.7)
            lim = [y_true[:, j].min(), y_true[:, j].max()]
            ax.plot(lim, lim, "k--", lw=1)
            ax.set_xlabel(f"true {target}")
            ax.set_ylabel(f"predicted {target}")
            r2 = result.per_target[target].r2
            ax.set_title(f"{result.name} {target} (R²={r2:.3f})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics_bar(results: List[ModelResult], out_path: Path):
    rows = []
    for r in results:
        for target, m in r.per_target.items():
            rows.append({"model": r.name, "target": target, "metric": "RMSE", "value": m.rmse})
            rows.append({"model": r.name, "target": target, "metric": "MAE", "value": m.mae})
            rows.append({"model": r.name, "target": target, "metric": "R2", "value": m.r2})
    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        sub = plot_df[plot_df["metric"] == metric]
        sns.barplot(data=sub, x="model", y="value", hue="target", ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(importances: pd.DataFrame, out_path: Path, top_n: int = 30, title: str = "Feature importance"):
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["mean"], align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("mean importance")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
