"""Evaluation value objects and metric computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..core.config import TARGETS


@dataclass(frozen=True, slots=True)
class Metrics:
    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ModelResult:
    name: str
    per_target: dict[str, Metrics]
    overall_rmse: float
    predictions: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "per_target": {
                key: value.to_dict() for key, value in self.per_target.items()
            },
            "overall_rmse": self.overall_rmse,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] = TARGETS,
) -> ModelResult:
    """Compute regression metrics after validating the evaluation boundary."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 2 or predicted.ndim != 2:
        raise ValueError("y_true and y_pred must both be two-dimensional")
    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {actual.shape} and {predicted.shape}"
        )
    if actual.shape[0] == 0:
        raise ValueError("Evaluation arrays cannot be empty")
    if actual.shape[1] != len(target_names):
        raise ValueError(
            f"Expected {len(target_names)} target columns, got {actual.shape[1]}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation arrays must contain only finite values")

    per_target: dict[str, Metrics] = {}
    for index, target in enumerate(target_names):
        per_target[target] = Metrics(
            rmse=float(
                np.sqrt(mean_squared_error(actual[:, index], predicted[:, index]))
            ),
            mae=float(mean_absolute_error(actual[:, index], predicted[:, index])),
            r2=float(r2_score(actual[:, index], predicted[:, index])),
        )
    return ModelResult(
        name="",
        per_target=per_target,
        overall_rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        predictions=predicted,
    )


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = [
        {
            "model": result.name,
            "target": target,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "r2": metrics.r2,
        }
        for result in results
        for target, metrics in result.per_target.items()
    ]
    return pd.DataFrame(
        rows,
        columns=["model", "target", "rmse", "mae", "r2"],
    )
