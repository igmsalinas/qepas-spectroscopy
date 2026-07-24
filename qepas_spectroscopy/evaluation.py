"""Evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import TARGETS


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelResult:
    name: str
    per_target: Dict[str, Metrics]
    overall_rmse: float
    predictions: np.ndarray

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "per_target": {k: v.to_dict() for k, v in self.per_target.items()},
            "overall_rmse": self.overall_rmse,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelResult:
    per_target = {}
    for i, target in enumerate(TARGETS):
        per_target[target] = Metrics(
            rmse=float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            mae=float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            r2=float(r2_score(y_true[:, i], y_pred[:, i])),
        )
    overall_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return ModelResult(
        name="",
        per_target=per_target,
        overall_rmse=overall_rmse,
        predictions=y_pred,
    )


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for target, m in r.per_target.items():
            rows.append({
                "model": r.name,
                "target": target,
                "rmse": m.rmse,
                "mae": m.mae,
                "r2": m.r2,
            })
    return pd.DataFrame(rows)


def save_results(results: list[ModelResult], output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = results_table(results)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    return summary
