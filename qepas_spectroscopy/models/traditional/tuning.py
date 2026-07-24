"""Hyperparameter search for traditional estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from ...core.config import RANDOM_STATE


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Grid search XGBoost with every concentration held out in turn."""
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    if len(splits) < 2:
        raise ValueError("XGBoost tuning requires at least two groups")

    grid = GridSearchCV(
        xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
        ),
        {
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [200, 400],
            "subsample": [0.8, 1.0],
        },
        scoring="neg_root_mean_squared_error",
        cv=splits,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X, y)
    return {
        "best_params": grid.best_params_,
        "best_score": float(-grid.best_score_),
    }
