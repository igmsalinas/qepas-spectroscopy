"""Nested grouped hyperparameter selection for small calibration datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import LeaveOneGroupOut

from ...evaluation import ModelResult, compute_metrics

EstimatorFactory = Callable[[Mapping[str, Any]], BaseEstimator]


def _plain_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True, slots=True)
class NestedSearchFold:
    """Hyperparameter decision made without observing one outer test group."""

    fold: int
    test_group: Any
    inner_nrmse: float
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "test_group": _plain_value(self.test_group),
            "inner_nrmse": self.inner_nrmse,
            "parameters": self.parameters,
        }


@dataclass(slots=True)
class NestedGroupSearchTrainer:
    """Select hyperparameters in inner LOGO folds and evaluate in outer LOGO.

    The selection score is the mean target-wise RMSE divided by the target
    range observed in the outer training partition. The outer test group never
    contributes to normalization, selection, preprocessing, or model fitting.
    """

    name: str
    estimator_factory: EstimatorFactory
    parameter_grid: Sequence[Mapping[str, Any]]
    selections: list[NestedSearchFold] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.parameter_grid:
            raise ValueError("parameter_grid cannot be empty")

    def cross_val_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> ModelResult:
        features = np.asarray(X)
        targets = np.asarray(y)
        group_values = np.asarray(groups)
        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("X and y must be two-dimensional")
        if not (len(features) == len(targets) == len(group_values)):
            raise ValueError("X, y, and groups must have the same sample count")
        if np.unique(group_values).size < 3:
            raise ValueError("Nested grouped search requires at least three groups")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("Nested grouped search requires finite arrays")

        predictions = np.empty(targets.shape, dtype=np.float64)
        self.selections.clear()
        outer = LeaveOneGroupOut()
        for fold, (outer_fit, outer_test) in enumerate(
            outer.split(features, targets, group_values)
        ):
            X_fit = features[outer_fit]
            y_fit = targets[outer_fit]
            fit_groups = group_values[outer_fit]
            target_range = np.ptp(y_fit, axis=0)
            target_range = np.where(target_range > 1e-12, target_range, 1.0)

            scores: list[float] = []
            for raw_parameters in self.parameter_grid:
                parameters = dict(raw_parameters)
                inner_predictions = np.empty(y_fit.shape, dtype=np.float64)
                inner = LeaveOneGroupOut()
                for inner_fit, inner_validation in inner.split(
                    X_fit,
                    y_fit,
                    fit_groups,
                ):
                    estimator = self.estimator_factory(parameters)
                    estimator.fit(X_fit[inner_fit], y_fit[inner_fit])
                    prediction = np.asarray(
                        estimator.predict(X_fit[inner_validation])
                    )
                    if prediction.ndim == 1:
                        prediction = prediction[:, np.newaxis]
                    if prediction.shape != y_fit[inner_validation].shape:
                        raise ValueError(
                            "Nested estimator returned an unexpected prediction shape"
                        )
                    inner_predictions[inner_validation] = prediction

                normalized_errors = (inner_predictions - y_fit) / target_range
                per_target_nrmse = np.sqrt(
                    np.mean(np.square(normalized_errors), axis=0)
                )
                scores.append(float(np.mean(per_target_nrmse)))

            best_index = int(np.argmin(scores))
            best_parameters = dict(self.parameter_grid[best_index])
            best_estimator = self.estimator_factory(best_parameters)
            best_estimator.fit(X_fit, y_fit)
            outer_prediction = np.asarray(
                best_estimator.predict(features[outer_test])
            )
            if outer_prediction.ndim == 1:
                outer_prediction = outer_prediction[:, np.newaxis]
            predictions[outer_test] = outer_prediction
            test_group = np.unique(group_values[outer_test])
            if len(test_group) != 1:
                raise ValueError("Outer fold must contain exactly one group")
            self.selections.append(
                NestedSearchFold(
                    fold=fold,
                    test_group=test_group[0],
                    inner_nrmse=scores[best_index],
                    parameters=best_parameters,
                )
            )

        result = compute_metrics(targets, predictions)
        result.name = self.name
        return result

    def selection_records(self) -> list[dict[str, Any]]:
        return [selection.to_dict() for selection in self.selections]
