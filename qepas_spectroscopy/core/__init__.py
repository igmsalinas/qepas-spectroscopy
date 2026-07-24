"""Shared configuration primitives."""

from .config import (
    DATA_DIR,
    DOCS_DIR,
    EDA_DIR,
    FEATURES_DIR,
    LABELS,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGETS,
    ensure_dirs,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DOCS_DIR",
    "OUTPUT_DIR",
    "MODELS_DIR",
    "FEATURES_DIR",
    "EDA_DIR",
    "NOTEBOOKS_DIR",
    "LABELS",
    "TARGETS",
    "RANDOM_STATE",
    "ensure_dirs",
]
