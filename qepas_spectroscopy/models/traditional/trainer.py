"""Grouped evaluation service for sklearn-compatible estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from ...evaluation import ModelResult, compute_metrics


@dataclass(slots=True)
class ModelTrainer:
    """Named estimator with grouped evaluation behavior."""

    name: str
    model: BaseEstimator

    def cross_val_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> ModelResult:
        if len(X) != len(y) or len(X) != len(groups):
            raise ValueError("X, y, and groups must have the same number of samples")
        if np.unique(groups).size < 2:
            raise ValueError("Grouped cross-validation requires at least two groups")
        predictions = cross_val_predict(
            self.model,
            X,
            y,
            cv=LeaveOneGroupOut(),
            groups=groups,
            n_jobs=-1,
        )
        result = compute_metrics(y, predictions)
        result.name = self.name
        return result

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        self.model.fit(X, y)
        return self.model
