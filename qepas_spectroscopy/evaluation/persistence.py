"""Persistence for evaluation results and out-of-fold predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.config import TARGETS
from .metrics import ModelResult, results_table


def save_results(
    results: list[ModelResult],
    output_dir: str | Path,
) -> pd.DataFrame:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = results_table(results)
    summary.to_csv(directory / "metrics_summary.csv", index=False)
    with (directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump([result.to_dict() for result in results], stream, indent=2)
    return summary


def save_prediction_artifacts(
    results: Sequence[ModelResult],
    y_true: np.ndarray,
    sample_ids: np.ndarray,
    groups: np.ndarray,
    output_dir: str | Path,
    *,
    target_names: Sequence[str] = TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist long-form OOF predictions and per-group error summaries."""
    actual = np.asarray(y_true)
    identifiers = np.asarray(sample_ids)
    group_values = np.asarray(groups)
    if actual.ndim != 2 or actual.shape[1] != len(target_names):
        raise ValueError("Targets do not match the configured target names")
    if not (len(actual) == len(identifiers) == len(group_values)):
        raise ValueError("Targets, sample IDs, and groups must align")

    records: list[dict[str, object]] = []
    for result in results:
        predictions = np.asarray(result.predictions)
        if predictions.shape != actual.shape:
            raise ValueError(
                f"Prediction shape for {result.name} does not match targets"
            )
        for sample_index, sample_id in enumerate(identifiers):
            for target_index, target in enumerate(target_names):
                observed = float(actual[sample_index, target_index])
                predicted = float(predictions[sample_index, target_index])
                records.append(
                    {
                        "model": result.name,
                        "sample_id": str(sample_id),
                        "group": str(group_values[sample_index]),
                        "target": target,
                        "actual": observed,
                        "predicted": predicted,
                        "residual": predicted - observed,
                        "absolute_error": abs(predicted - observed),
                    }
                )

    predictions_frame = pd.DataFrame.from_records(records)
    grouped = predictions_frame.groupby(
        ["model", "target", "group"],
        sort=True,
    )
    group_metrics = grouped.agg(
        samples=("sample_id", "size"),
        rmse=("residual", lambda values: float(np.sqrt(np.mean(values**2)))),
        mae=("absolute_error", "mean"),
        bias=("residual", "mean"),
    ).reset_index()

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(directory / "predictions.csv", index=False)
    group_metrics.to_csv(directory / "group_metrics.csv", index=False)
    return predictions_frame, group_metrics

def save_seed_group_metrics(
    seed_predictions: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    groups: np.ndarray,
    output_dir: str | Path,
    *,
    target_names: Sequence[str] = TARGETS,
) -> pd.DataFrame:
    """Persist per-seed, per-group errors for stochastic ensembles."""
    actual = np.asarray(y_true)
    group_values = np.asarray(groups)
    if actual.ndim != 2 or actual.shape[1] != len(target_names):
        raise ValueError("Targets do not match the configured target names")
    if group_values.ndim != 1 or len(group_values) != len(actual):
        raise ValueError("Targets and groups must align")

    records: list[dict[str, object]] = []
    for model_name, values in seed_predictions.items():
        predictions = np.asarray(values)
        expected = (len(actual), actual.shape[1])
        if predictions.ndim != 3 or predictions.shape[1:] != expected:
            raise ValueError(
                f"Seed predictions for {model_name} must have shape "
                f"(seeds, {expected[0]}, {expected[1]})"
            )
        if not np.isfinite(predictions).all():
            raise ValueError(
                f"Seed predictions for {model_name} must be finite"
            )
        for seed_index, seed_values in enumerate(predictions):
            for group in np.unique(group_values):
                mask = group_values == group
                for target_index, target in enumerate(target_names):
                    residual = (
                        seed_values[mask, target_index]
                        - actual[mask, target_index]
                    )
                    records.append(
                        {
                            "model": model_name,
                            "seed": seed_index + 1,
                            "target": target,
                            "group": str(group),
                            "samples": int(np.sum(mask)),
                            "rmse": float(np.sqrt(np.mean(residual**2))),
                            "mae": float(np.mean(np.abs(residual))),
                            "bias": float(np.mean(residual)),
                        }
                    )

    frame = pd.DataFrame.from_records(
        records,
        columns=[
            "model",
            "seed",
            "target",
            "group",
            "samples",
            "rmse",
            "mae",
            "bias",
        ],
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "deep_seed_group_metrics.csv", index=False)
    return frame

