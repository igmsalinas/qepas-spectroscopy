"""Traditional ML models: Ridge, Random Forest, XGBoost."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV

from ..config import TARGETS
from ..evaluation import ModelResult, compute_metrics


class ModelTrainer:
    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model

    def cross_val_predict(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> ModelResult:
        logo = LeaveOneGroupOut()
        y_pred = cross_val_predict(self.model, X, y, cv=logo, groups=groups, n_jobs=-1)
        result = compute_metrics(y, y_pred)
        result.name = self.name
        return result

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        self.model.fit(X, y)
        return self.model


def build_ridge() -> ModelTrainer:
    return ModelTrainer("Ridge", MultiOutputRegressor(Ridge(alpha=1.0, random_state=42)))


def build_random_forest() -> ModelTrainer:
    return ModelTrainer(
        "RandomForest",
        RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1),
    )


def build_xgboost() -> ModelTrainer:
    return ModelTrainer(
        "XGBoost",
        xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
        ),
    )


def build_gradient_boosting() -> ModelTrainer:
    return ModelTrainer(
        "GradientBoosting",
        MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=42)
        ),
    )


def tune_xgboost(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """Quick grid search for XGBoost hyperparameters using one concentration as validation."""
    logo = LeaveOneGroupOut()
    train_idx, val_idx = next(logo.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    param_grid = {
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.8, 1.0],
    }
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    grid = GridSearchCV(
        base,
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=[(train_idx, val_idx)],
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X, y)
    return {
        "best_params": grid.best_params_,
        "best_score": float(-grid.best_score_),
    }
