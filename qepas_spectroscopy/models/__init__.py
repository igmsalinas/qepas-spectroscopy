"""Model factories.

Deep-learning exports are loaded lazily so traditional ML workflows do not pay
TensorFlow startup and GPU-initialization costs.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .traditional import (
    ModelTrainer,
    build_gradient_boosting,
    build_random_forest,
    build_ridge,
    build_xgboost,
    tune_xgboost,
)

_DL_EXPORTS = {"tune_deep_model", "train_deep_model", "ARCHITECTURES"}


def __getattr__(name: str) -> Any:
    if name in _DL_EXPORTS:
        return getattr(import_module(".deep_learning", __name__), name)
    raise AttributeError(name)


__all__ = [
    "ModelTrainer",
    "build_ridge",
    "build_random_forest",
    "build_xgboost",
    "build_gradient_boosting",
    "tune_xgboost",
    "tune_deep_model",
    "train_deep_model",
    "ARCHITECTURES",
]
