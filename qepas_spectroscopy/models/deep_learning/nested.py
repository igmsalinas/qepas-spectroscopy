"""Fully nested, multi-seed deep evaluation for small grouped datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ...core.config import RANDOM_STATE
from ...data.augmentation import (
    SpectralAugmentationConfig,
    augment_training_partition,
)
from ...data.normalization import NormalizationBundle
from ...evaluation import ModelResult, compute_metrics
from ...validation import nested_group_folds
from .trainer import (
    _build_tuned_model,
    _make_callbacks,
    _set_seed,
    _steps_per_epoch,
    tune_deep_model,
)


@dataclass(frozen=True, slots=True)
class FullyNestedDeepOutcome:
    results: tuple[ModelResult, ...]
    search: dict[str, Any]
    seed_predictions: dict[str, np.ndarray]


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _metric_payload(result: ModelResult) -> dict[str, Any]:
    return result.to_dict()


def _fold_paths(directory: Path, fold: int) -> tuple[Path, Path]:
    return directory / f"fold_{fold}.npz", directory / f"fold_{fold}.json"


def _load_completed_fold(
    directory: Path,
    fold: int,
    signature: dict[str, Any],
    variant_names: tuple[str, ...],
    seeds: int,
    test_size: int,
    output_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    predictions_path, metadata_path = _fold_paths(directory, fold)
    if not predictions_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return None
        with np.load(predictions_path) as payload:
            predictions = {
                name: np.asarray(payload[name]) for name in variant_names
            }
        expected = (seeds, test_size, output_count)
        if any(value.shape != expected for value in predictions.values()):
            return None
        if any(not np.isfinite(value).all() for value in predictions.values()):
            return None
        return predictions, metadata
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_completed_fold(
    directory: Path,
    fold: int,
    predictions: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    predictions_path, metadata_path = _fold_paths(directory, fold)
    predictions_tmp = directory / f"fold_{fold}.tmp.npz"
    metadata_tmp = directory / f"fold_{fold}.tmp.json"
    with predictions_tmp.open("wb") as stream:
        np.savez_compressed(stream, **predictions)
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    predictions_tmp.replace(predictions_path)
    metadata_tmp.replace(metadata_path)


def train_fully_nested_inception(
    X_signals: np.ndarray,
    X_scalars: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    max_trials: int,
    tuner_epochs: int,
    executions_per_trial: int,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    seeds: int,
    augmentation_ablation: bool,
    tuner_dir: str | Path,
    artifact_dir: str | Path,
    log_dir: str | Path | None = None,
    resume: bool = False,
    preprocessing_id: str = "unspecified",
    augmentation_config: SpectralAugmentationConfig | None = None,
) -> FullyNestedDeepOutcome:
    """Tune inside every outer fold and evaluate paired seed ensembles."""
    signals = np.asarray(X_signals)
    scalars = np.asarray(X_scalars)
    targets = np.asarray(y)
    group_values = np.asarray(groups)
    if signals.ndim != 3 or scalars.ndim != 2 or targets.ndim != 2:
        raise ValueError("deep inputs must have dimensions 3, 2, and 2")
    if not (len(signals) == len(scalars) == len(targets) == len(group_values)):
        raise ValueError("deep inputs must have the same sample count")
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if augmentation_ablation and signals.shape[2] != 3:
        raise ValueError("augmentation ablation requires the three-channel view")

    selected_augmentation = augmentation_config or SpectralAugmentationConfig()
    variants: tuple[tuple[str, bool], ...] = (
        ("InceptionNestedNoAug", False),
    )
    if augmentation_ablation:
        variants += (("InceptionNestedAug", True),)
    variant_names = tuple(name for name, _ in variants)
    seed_predictions = {
        name: np.full(
            (seeds, len(targets), targets.shape[1]),
            np.nan,
            dtype=np.float32,
        )
        for name in variant_names
    }
    fold_records: list[dict[str, Any]] = []
    fold_directory = Path(artifact_dir) / "nested_deep_folds"
    tuner_root = Path(tuner_dir) / "fully_nested_small_inception"
    log_root = Path(log_dir) if log_dir is not None else None
    signature = {
        "protocol_version": 1,
        "preprocessing_id": preprocessing_id,
        "seeds": seeds,
        "augmentation_ablation": augmentation_ablation,
        "augmentation_config": asdict(selected_augmentation),
        "max_trials": max_trials,
        "tuner_epochs": tuner_epochs,
        "executions_per_trial": executions_per_trial,
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
    }

    folds = list(nested_group_folds(group_values))
    for fold in folds:
        completed = (
            _load_completed_fold(
                fold_directory,
                fold.fold,
                signature,
                variant_names,
                seeds,
                len(fold.test_indices),
                targets.shape[1],
            )
            if resume
            else None
        )
        if completed is not None:
            loaded_predictions, loaded_metadata = completed
            for name in variant_names:
                seed_predictions[name][:, fold.test_indices] = loaded_predictions[name]
            fold_records.append(loaded_metadata)
            print(
                f"Nested deep fold {fold.fold + 1}/{len(folds)}: "
                f"loaded completed test group {fold.test_group}"
            )
            continue

        print(
            f"Nested deep fold {fold.fold + 1}/{len(folds)}: "
            f"tune with test={fold.test_group}, validation={fold.validation_group}"
        )
        development_indices = np.sort(
            np.concatenate([fold.fit_indices, fold.validation_indices])
        )
        tuning = tune_deep_model(
            signals[development_indices],
            scalars[development_indices],
            targets[development_indices],
            group_values[development_indices],
            validation_group=fold.validation_group,
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            project_name="small_inception",
            tuner_dir=tuner_root / f"fold_{fold.fold}",
            max_epochs=tuner_epochs,
            batch_size=batch_size,
            early_stopping_patience=max(4, early_stopping_patience // 2),
            allowed_architectures=["inception_spectra"],
            search_profile="small_inception",
            search_verbose=0,
            resume=resume,
            preprocessing_id=preprocessing_id,
        )
        best_params = dict(tuning["best_params"])
        normalizers = NormalizationBundle.fit(
            signals[fold.fit_indices],
            scalars[fold.fit_indices],
            targets[fold.fit_indices],
        )
        validation_inputs = {
            "signals": normalizers.transform_signals(
                signals[fold.validation_indices]
            ),
            "scalars": normalizers.transform_scalars(
                scalars[fold.validation_indices]
            ),
        }
        validation_targets = normalizers.transform_target(
            targets[fold.validation_indices]
        )
        test_inputs = {
            "signals": normalizers.transform_signals(signals[fold.test_indices]),
            "scalars": normalizers.transform_scalars(scalars[fold.test_indices]),
        }
        diagnostics: dict[str, float] | None = None
        for seed_index in range(seeds):
            for name, augmented in variants:
                offset = fold.fold * 100 + seed_index
                tf.keras.backend.clear_session()
                _set_seed(offset)
                if augmented:
                    batch = augment_training_partition(
                        signals[fold.fit_indices],
                        scalars[fold.fit_indices],
                        targets[fold.fit_indices],
                        seed=RANDOM_STATE + offset,
                        config=selected_augmentation,
                    )
                    fit_signals = batch.signals
                    fit_scalars = batch.scalars
                    fit_targets = batch.targets
                    diagnostics = batch.diagnostics
                else:
                    fit_signals = signals[fold.fit_indices]
                    fit_scalars = scalars[fold.fit_indices]
                    fit_targets = targets[fold.fit_indices]

                steps = _steps_per_epoch(len(fit_targets), batch_size)
                model = _build_tuned_model(
                    best_params,
                    (signals.shape[1], signals.shape[2]),
                    scalars.shape[1],
                    targets.shape[1],
                    epochs,
                    steps,
                )
                callbacks = _make_callbacks(
                    early_stopping_patience,
                    (
                        log_root / name / f"fold_{fold.fold}" / f"seed_{seed_index}"
                        if log_root is not None
                        else None
                    ),
                )
                model.fit(
                    {
                        "signals": normalizers.transform_signals(fit_signals),
                        "scalars": normalizers.transform_scalars(fit_scalars),
                    },
                    normalizers.transform_target(fit_targets),
                    validation_data=(validation_inputs, validation_targets),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0,
                )
                normalized_prediction = model.predict(test_inputs, verbose=0)
                seed_predictions[name][seed_index, fold.test_indices] = (
                    normalizers.inverse_transform_target(normalized_prediction)
                )
                print(
                    f"  {name} seed {seed_index + 1}/{seeds} completed"
                )

        record = {
            "signature": signature,
            "fold": fold.fold,
            "test_group": _plain(fold.test_group),
            "validation_group": _plain(fold.validation_group),
            "fit_groups": [
                _plain(value) for value in np.unique(group_values[fold.fit_indices])
            ],
            "best_params": best_params,
            "trials": tuning.get("trials", []),
            "augmentation_diagnostics": diagnostics,
        }
        fold_records.append(record)
        _save_completed_fold(
            fold_directory,
            fold.fold,
            {
                name: seed_predictions[name][:, fold.test_indices]
                for name in variant_names
            },
            record,
        )

    results: list[ModelResult] = []
    seed_metrics: dict[str, list[dict[str, Any]]] = {}
    for name in variant_names:
        predictions = seed_predictions[name]
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"Incomplete seed predictions for {name}")
        ensemble_prediction = np.mean(predictions, axis=0)
        result = compute_metrics(targets, ensemble_prediction)
        result.name = name
        results.append(result)
        seed_metrics[name] = []
        for seed_index in range(seeds):
            seed_result = compute_metrics(targets, predictions[seed_index])
            seed_result.name = f"{name}Seed{seed_index + 1}"
            seed_metrics[name].append(_metric_payload(seed_result))

    search = {
        "protocol": "fully_nested_grouped_small_inception",
        "independent_outer_estimate": True,
        "selection_metric": "normalized_target_validation_mae",
        "seeds": seeds,
        "ensemble": "mean_prediction",
        "augmentation_ablation": augmentation_ablation,
        "augmentation_config": asdict(selected_augmentation),
        "augmentation_hyperparameters_retuned": False,
        "outer_folds": fold_records,
        "seed_metrics": seed_metrics,
    }
    return FullyNestedDeepOutcome(
        results=tuple(results),
        search=search,
        seed_predictions=seed_predictions,
    )
