"""Leakage-safe grouped data splitting policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


def _validated_groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(groups)
    if values.ndim != 1:
        raise ValueError("groups must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("groups cannot be empty")
    return values, np.unique(values)


@dataclass(frozen=True, slots=True)
class ValidationSplit:
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    validation_group: Any


@dataclass(frozen=True, slots=True)
class NestedGroupFold:
    """One outer test fold with an independent inner validation group."""

    fold: int
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    validation_group: Any
    test_group: Any


def validation_group_split(
    groups: np.ndarray,
    validation_group: Any | None = None,
) -> ValidationSplit:
    """Reserve one complete group for validation."""
    values, unique_groups = _validated_groups(groups)
    chosen = unique_groups[-1] if validation_group is None else validation_group
    if not np.any(unique_groups == chosen):
        raise ValueError(f"Validation group {chosen!r} is not present in groups")

    validation_indices = np.flatnonzero(values == chosen)
    fit_indices = np.flatnonzero(values != chosen)
    if fit_indices.size == 0:
        raise ValueError("At least two groups are required for a validation split")
    return ValidationSplit(
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        validation_group=chosen,
    )


def nested_group_folds(groups: np.ndarray) -> Iterator[NestedGroupFold]:
    """Yield disjoint outer test and inner validation group folds."""
    values, unique_groups = _validated_groups(groups)
    if unique_groups.size < 3:
        raise ValueError("Nested grouped validation requires at least three groups")

    for fold, test_group in enumerate(unique_groups):
        validation_group = unique_groups[(fold - 1) % unique_groups.size]
        yield NestedGroupFold(
            fold=fold,
            fit_indices=np.flatnonzero(
                (values != test_group) & (values != validation_group)
            ),
            validation_indices=np.flatnonzero(values == validation_group),
            test_indices=np.flatnonzero(values == test_group),
            validation_group=validation_group,
            test_group=test_group,
        )
