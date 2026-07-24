"""Factories for traditional multi-output regression estimators."""

from __future__ import annotations

from typing import Any

import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...core.config import RANDOM_STATE
from .nested import NestedGroupSearchTrainer
from .trainer import ModelTrainer


def build_ridge() -> ModelTrainer:
    """Build Ridge with fold-local standardization inside its pipeline."""
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "regressor",
                MultiOutputRegressor(
                    Ridge(alpha=1.0, random_state=RANDOM_STATE)
                ),
            ),
        ]
    )
    return ModelTrainer("Ridge", model)


def build_random_forest() -> ModelTrainer:
    return ModelTrainer(
        "RandomForest",
        RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            # Outer CV owns parallelism; nested -1 values oversubscribe CPUs.
            n_jobs=1,
        ),
    )


def build_xgboost(**overrides: Any) -> ModelTrainer:
    parameters: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    parameters.update(overrides)
    return ModelTrainer("XGBoost", xgb.XGBRegressor(**parameters))


def build_gradient_boosting() -> ModelTrainer:
    return ModelTrainer(
        "GradientBoosting",
        MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            )
        ),
    )


def build_nested_ridge(
    name: str = "RidgeNested",
    alphas: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    ),
) -> NestedGroupSearchTrainer:
    """Build Ridge with alpha selected inside each outer training fold."""

    def estimator(parameters: dict[str, Any]) -> Pipeline:
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "regressor",
                    MultiOutputRegressor(
                        Ridge(
                            alpha=float(parameters["alpha"]),
                            random_state=RANDOM_STATE,
                        )
                    ),
                ),
            ]
        )

    return NestedGroupSearchTrainer(
        name=name,
        estimator_factory=estimator,
        parameter_grid=tuple({"alpha": alpha} for alpha in alphas),
    )


def build_nested_pls(
    name: str = "PLSNested",
    components: tuple[int, ...] = (2, 3, 5, 8, 12),
) -> NestedGroupSearchTrainer:
    """Build multi-output PLS with components selected by nested LOGO."""

    def estimator(parameters: dict[str, Any]) -> PLSRegression:
        return PLSRegression(
            n_components=int(parameters["n_components"]),
            scale=True,
            max_iter=1000,
            tol=1e-8,
        )

    return NestedGroupSearchTrainer(
        name=name,
        estimator_factory=estimator,
        parameter_grid=tuple(
            {"n_components": n_components}
            for n_components in components
        ),
    )
