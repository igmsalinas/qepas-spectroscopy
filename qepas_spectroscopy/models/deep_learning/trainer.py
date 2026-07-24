"""Leakage-safe training utilities for deep-learning models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras import callbacks

from ...core.config import RANDOM_STATE
from ...evaluation import ModelResult, compute_metrics
from ...validation import nested_group_folds, validation_group_split
from .architectures import build_simple_cnn
from .hypermodel import build_qepas_hypermodel
from .small_data import build_small_inception_hypermodel
from ...data.normalization import NormalizationBundle

CHECKPOINT_FORMAT_VERSION = 4
TUNER_PROTOCOL_VERSION = 7


@dataclass(frozen=True, slots=True)
class LoadedFoldCheckpoint:
    """A model and the exact preprocessing state used to train it."""

    model: tf.keras.Model
    normalizers: NormalizationBundle


def _set_seed(offset: int = 0) -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE + offset)


def _make_callbacks(
    patience: int,
    log_dir: Path | None = None,
) -> list[callbacks.Callback]:
    """Build callbacks without conflicting plateau and scheduled LR policies."""
    result: list[callbacks.Callback] = [
        callbacks.EarlyStopping(
            monitor="val_mae",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    if log_dir is not None:
        result.append(
            callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0)
        )
    return result


def _tuner_exists(
    tuner_dir: Path,
    project_name: str,
    expected_protocol: dict[str, Any] | None = None,
) -> bool:
    project_dir = tuner_dir / project_name
    protocol_path = project_dir / "qepas_protocol.json"
    if not (project_dir / "oracle.json").is_file() or not any(
        path.is_dir() for path in project_dir.glob("trial_*")
    ):
        return False
    try:
        with protocol_path.open(encoding="utf-8") as stream:
            protocol = json.load(stream)
    except (OSError, ValueError):
        return False
    if protocol.get("version") != TUNER_PROTOCOL_VERSION:
        return False
    return expected_protocol is None or protocol == expected_protocol


def _write_tuner_protocol(
    tuner_dir: Path,
    project_name: str,
    protocol: dict[str, Any],
) -> None:
    path = tuner_dir / project_name / "qepas_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)


def _steps_per_epoch(sample_count: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return max(1, math.ceil(sample_count / batch_size))


def tune_deep_model(
    X_signals: np.ndarray,
    X_scalars: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    validation_group: Any | None = None,
    max_trials: int = 20,
    executions_per_trial: int = 1,
    project_name: str = "qepas_advanced_tuning",
    tuner_dir: str | Path = "outputs/keras_tuner_advanced",
    max_epochs: int = 100,
    batch_size: int = 16,
    early_stopping_patience: int = 12,
    allowed_architectures: list[str] | None = None,
    search_profile: str = "broad",
    search_verbose: int = 1,
    resume: bool = False,
    preprocessing_id: str = "unspecified",
) -> dict[str, Any]:
    """Tune on one group holdout with fit-only preprocessing.

    This is a development model-selection split, not an unbiased outer
    performance estimate. Callers that reuse the selected hyperparameters
    across outer folds must report that limitation."""
    _set_seed()
    split = validation_group_split(groups, validation_group)
    normalizers = NormalizationBundle.fit(
        X_signals[split.fit_indices],
        X_scalars[split.fit_indices],
        y[split.fit_indices],
    )
    X_sig_norm = normalizers.transform_signals(X_signals)
    X_sc_norm = normalizers.transform_scalars(X_scalars)
    y_norm = normalizers.transform_target(y)

    tuner_path = Path(tuner_dir)
    tuner_path.mkdir(parents=True, exist_ok=True)
    steps = _steps_per_epoch(len(split.fit_indices), batch_size)
    if search_profile == "broad":
        hypermodel_fn = build_qepas_hypermodel(
            input_shape=(X_sig_norm.shape[1], X_sig_norm.shape[2]),
            scalar_dim=X_sc_norm.shape[1],
            num_outputs=y_norm.shape[1],
            max_epochs=max_epochs,
            steps_per_epoch=steps,
            allowed_architectures=allowed_architectures,
        )
    elif search_profile == "small_inception":
        allowed = set(allowed_architectures or ["inception_spectra"])
        if allowed != {"inception_spectra"}:
            raise ValueError(
                "small_inception search only supports inception_spectra"
            )
        hypermodel_fn = build_small_inception_hypermodel(
            input_shape=(X_sig_norm.shape[1], X_sig_norm.shape[2]),
            scalar_dim=X_sc_norm.shape[1],
            num_outputs=y_norm.shape[1],
            max_epochs=max_epochs,
            steps_per_epoch=steps,
        )
    else:
        raise ValueError(f"Unknown deep search profile: {search_profile}")

    protocol = {
        "version": TUNER_PROTOCOL_VERSION,
        "input_shape": list(X_sig_norm.shape[1:]),
        "scalar_dim": int(X_sc_norm.shape[1]),
        "num_outputs": int(y_norm.shape[1]),
        "max_trials": max_trials,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "executions_per_trial": executions_per_trial,
        "random_state": RANDOM_STATE,
        "objective": "val_mae",
        "validation_group": _json_value(split.validation_group),
        "search_profile": search_profile,
        "allowed_architectures": allowed_architectures,
        "preprocessing_id": preprocessing_id,
    }
    can_resume = resume and _tuner_exists(
        tuner_path,
        project_name,
        expected_protocol=protocol,
    )
    if can_resume:
        print(f"Resuming existing KerasTuner project at {tuner_path / project_name}")

    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel_fn,
        objective=kt.Objective("val_mae", direction="min"),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        seed=RANDOM_STATE,
        directory=str(tuner_path),
        project_name=project_name,
        overwrite=not can_resume,
    )
    _write_tuner_protocol(tuner_path, project_name, protocol)
    tuner.search(
        {
            "signals": X_sig_norm[split.fit_indices],
            "scalars": X_sc_norm[split.fit_indices],
        },
        y_norm[split.fit_indices],
        validation_data=(
            {
                "signals": X_sig_norm[split.validation_indices],
                "scalars": X_sc_norm[split.validation_indices],
            },
            y_norm[split.validation_indices],
        ),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=_make_callbacks(early_stopping_patience),
        verbose=search_verbose,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_params = {key: best_hps.get(key) for key in best_hps.values}
    validation_group = split.validation_group
    if isinstance(validation_group, np.generic):
        validation_group = validation_group.item()
    fit_groups = [
        _json_value(group)
        for group in np.unique(groups[split.fit_indices]).tolist()
    ]
    trial_summaries = []
    for trial in tuner.oracle.trials.values():
        trial_summaries.append(
            {
                "trial_id": trial.trial_id,
                "status": str(trial.status),
                "score": (
                    float(trial.score) if trial.score is not None else None
                ),
                "best_step": trial.best_step,
                "parameters": dict(trial.hyperparameters.values),
            }
        )
    trial_summaries.sort(
        key=lambda trial: (
            trial["score"] is None,
            trial["score"] if trial["score"] is not None else float("inf"),
        )
    )
    return {
        "best_params": best_params,
        "normalizers": normalizers,
        "tuner": tuner,
        "trials": trial_summaries,
        "validation_group": validation_group,
        "selection_protocol": {
            "name": "single_group_holdout",
            "objective": "val_mae",
            "fit_groups": fit_groups,
            "validation_group": validation_group,
            "independent_outer_estimate": False,
        },
        "max_epochs": max_epochs,
        "steps_per_epoch": steps,
        "preprocessing_id": preprocessing_id,
    }


def _fold_checkpoint_paths(
    log_dir: Path | None,
    fold: int,
    *,
    create: bool = False,
) -> tuple[Path | None, Path | None]:
    if log_dir is None:
        return None, None
    directory = Path(log_dir) / "fold_checkpoints"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return (
        directory / f"fold_{fold}.weights.h5",
        directory / f"fold_{fold}.json",
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hyperparameters_from_values(values: dict[str, Any]) -> kt.HyperParameters:
    hyperparameters = kt.HyperParameters()
    hyperparameters.values.update(values)
    return hyperparameters


def _build_tuned_model(
    best_params: dict[str, Any],
    input_shape: tuple[int, int],
    scalar_dim: int,
    num_outputs: int,
    epochs: int,
    steps_per_epoch: int,
) -> tf.keras.Model:
    architecture = best_params.get("architecture")
    allowed = [architecture] if isinstance(architecture, str) else None
    hypermodel = build_qepas_hypermodel(
        input_shape=input_shape,
        scalar_dim=scalar_dim,
        num_outputs=num_outputs,
        max_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        allowed_architectures=allowed,
    )
    return hypermodel(_hyperparameters_from_values(best_params))


def _build_model_from_spec(
    model_spec: dict[str, Any],
    input_shape: tuple[int, int],
    scalar_dim: int,
    num_outputs: int,
) -> tf.keras.Model:
    if model_spec["kind"] == "simple_cnn":
        model = build_simple_cnn(
            input_shape=input_shape,
            scalar_dim=scalar_dim,
            num_outputs=num_outputs,
        )
        return model
    if model_spec["kind"] == "tuned":
        return _build_tuned_model(
            best_params=model_spec["best_params"],
            input_shape=input_shape,
            scalar_dim=scalar_dim,
            num_outputs=num_outputs,
            epochs=int(model_spec["epochs"]),
            steps_per_epoch=int(model_spec["steps_per_epoch"]),
        )
    raise ValueError(f"Unknown checkpoint model kind: {model_spec['kind']}")


def _save_fold_checkpoint(
    model: tf.keras.Model,
    normalizers: NormalizationBundle,
    model_spec: dict[str, Any],
    log_dir: Path | None,
    fold: int,
    test_group: Any,
    preprocessing_id: str,
) -> None:
    weights_path, config_path = _fold_checkpoint_paths(
        log_dir,
        fold,
        create=True,
    )
    if weights_path is None or config_path is None:
        return
    model.save_weights(weights_path)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "test_group": _json_value(test_group),
        "preprocessing_id": preprocessing_id,
        "normalizers": normalizers.to_dict(),
        "model": model_spec,
    }
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"Saved fold {fold} checkpoint to {weights_path}")


def _load_fold_checkpoint(
    log_dir: Path | None,
    fold: int,
    expected_test_group: Any,
    input_shape: tuple[int, int],
    scalar_dim: int,
    num_outputs: int,
    expected_preprocessing_id: str,
) -> LoadedFoldCheckpoint | None:
    weights_path, config_path = _fold_checkpoint_paths(log_dir, fold)
    if weights_path is None or config_path is None:
        return None
    if not weights_path.is_file() or not config_path.is_file():
        return None
    try:
        with config_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            print(f"Ignoring legacy checkpoint for fold {fold}")
            return None
        if payload.get("test_group") != _json_value(expected_test_group):
            print(f"Ignoring checkpoint for fold {fold}: test group changed")
            return None
        if payload.get("preprocessing_id") != expected_preprocessing_id:
            print(f"Ignoring checkpoint for fold {fold}: preprocessing changed")
            return None

        normalizers = NormalizationBundle.from_dict(payload["normalizers"])
        model = _build_model_from_spec(
            payload["model"],
            input_shape=input_shape,
            scalar_dim=scalar_dim,
            num_outputs=num_outputs,
        )
        model.load_weights(weights_path)
        print(f"Loaded fold {fold} checkpoint from {weights_path}")
        return LoadedFoldCheckpoint(model=model, normalizers=normalizers)
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"Could not load checkpoint for fold {fold}: {error}")
        return None


def train_deep_model(
    X_signals: np.ndarray,
    X_scalars: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tuner_result: dict[str, Any] | None = None,
    epochs: int = 150,
    batch_size: int = 16,
    early_stopping_patience: int = 20,
    log_dir: Path | None = None,
    model_name: str = "DeepLearning",
    resume: bool = False,
    preprocessing_id: str = "unspecified",
) -> ModelResult:
    """Evaluate a deep model with group-disjoint early-stopping folds.

    Every outer test fold receives a new normalizer fitted only on the inner fit
    groups. Early stopping observes an independent validation group, never the
    outer test group. If ``tuner_result`` was selected globally, architecture
    selection is not nested inside each outer fold; the resulting predictions
    are therefore a development estimate. Checkpoints persist preprocessing
    alongside model weights.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    if not (len(X_signals) == len(X_scalars) == len(y) == len(groups)):
        raise ValueError("All deep-learning inputs must have the same sample count")

    input_shape = (X_signals.shape[1], X_signals.shape[2])
    scalar_dim = X_scalars.shape[1]
    num_outputs = y.shape[1]
    predictions = np.zeros_like(y)
    folds = list(nested_group_folds(groups))

    for fold in folds:
        _set_seed(fold.fold)
        tf.keras.backend.clear_session()
        if resume:
            loaded = _load_fold_checkpoint(
                log_dir,
                fold.fold,
                fold.test_group,
                input_shape,
                scalar_dim,
                num_outputs,
                preprocessing_id,
            )
            if loaded is not None:
                print(
                    f"Deep model fold {fold.fold + 1}/{len(folds)}: "
                    f"loading checkpoint for group {fold.test_group}"
                )
                prediction = loaded.model.predict(
                    {
                        "signals": loaded.normalizers.transform_signals(
                            X_signals[fold.test_indices]
                        ),
                        "scalars": loaded.normalizers.transform_scalars(
                            X_scalars[fold.test_indices]
                        ),
                    },
                    verbose=0,
                )
                predictions[fold.test_indices] = (
                    loaded.normalizers.inverse_transform_target(prediction)
                )
                continue

        print(
            f"Deep model fold {fold.fold + 1}/{len(folds)}: "
            f"test={fold.test_group}, validation={fold.validation_group}"
        )
        normalizers = NormalizationBundle.fit(
            X_signals[fold.fit_indices],
            X_scalars[fold.fit_indices],
            y[fold.fit_indices],
        )
        steps = _steps_per_epoch(len(fold.fit_indices), batch_size)
        if tuner_result is None:
            model = build_simple_cnn(
                input_shape=input_shape,
                scalar_dim=scalar_dim,
                num_outputs=num_outputs,
            )
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            model_spec: dict[str, Any] = {"kind": "simple_cnn"}
        else:
            best_params = dict(tuner_result["best_params"])
            model = _build_tuned_model(
                best_params,
                input_shape,
                scalar_dim,
                num_outputs,
                epochs,
                steps,
            )
            model_spec = {
                "kind": "tuned",
                "best_params": best_params,
                "epochs": epochs,
                "steps_per_epoch": steps,
            }

        fit_inputs = {
            "signals": normalizers.transform_signals(
                X_signals[fold.fit_indices]
            ),
            "scalars": normalizers.transform_scalars(
                X_scalars[fold.fit_indices]
            ),
        }
        validation_inputs = {
            "signals": normalizers.transform_signals(
                X_signals[fold.validation_indices]
            ),
            "scalars": normalizers.transform_scalars(
                X_scalars[fold.validation_indices]
            ),
        }
        model.fit(
            fit_inputs,
            normalizers.transform_target(y[fold.fit_indices]),
            validation_data=(
                validation_inputs,
                normalizers.transform_target(y[fold.validation_indices]),
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=_make_callbacks(
                patience=early_stopping_patience,
                log_dir=(
                    Path(log_dir) / f"fold_{fold.fold}"
                    if log_dir is not None
                    else None
                ),
            ),
            verbose=0,
        )
        _save_fold_checkpoint(
            model,
            normalizers,
            model_spec,
            log_dir,
            fold.fold,
            fold.test_group,
            preprocessing_id,
        )
        prediction = model.predict(
            {
                "signals": normalizers.transform_signals(
                    X_signals[fold.test_indices]
                ),
                "scalars": normalizers.transform_scalars(
                    X_scalars[fold.test_indices]
                ),
            },
            verbose=0,
        )
        predictions[fold.test_indices] = normalizers.inverse_transform_target(
            prediction
        )

    result = compute_metrics(y, predictions)
    result.name = model_name
    return result


def save_tuner_results(
    tuner_result: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": tuner_result["best_params"],
        "normalizers": tuner_result["normalizers"].to_dict(),
        "validation_group": tuner_result.get("validation_group"),
        "selection_protocol": tuner_result.get("selection_protocol"),
        "preprocessing_id": tuner_result.get("preprocessing_id"),
        "trials": tuner_result.get("trials", []),
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
