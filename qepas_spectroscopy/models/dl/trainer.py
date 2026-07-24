"""Training utilities for deep-learning models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from keras import callbacks
import keras_tuner as kt

from ...config import RANDOM_STATE
from ...evaluation import ModelResult, compute_metrics
from .hypermodel import build_qepas_hypermodel
from .normalizers import NormalizationBundle


def _set_seed():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def _make_callbacks(patience: int, log_dir: Path | None = None):
    """Build callbacks. Avoids ReduceLROnPlateau because LR schedules are used."""
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if log_dir is not None:
        cb.append(callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0))
    return cb


def _tuner_exists(tuner_dir: Path, project_name: str) -> bool:
    project_dir = tuner_dir / project_name
    return (project_dir / "oracle.json").exists() and (project_dir / "trial_0").exists()


def tune_deep_model(
    X_signals: np.ndarray,
    X_scalars: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_trials: int = 20,
    executions_per_trial: int = 1,
    project_name: str = "qepas_advanced_tuning",
    tuner_dir: str | Path = "outputs/keras_tuner_advanced",
    max_epochs: int = 100,
    batch_size: int = 16,
    early_stopping_patience: int = 12,
    allowed_architectures: list[str] | None = None,
    resume: bool = False,
) -> Dict[str, any]:
    """Run KerasTuner BayesianOptimization on the rich hypermodel.

    If resume=True and a previous tuner project exists, continue the search.
    """
    _set_seed()

    tuner_dir = Path(tuner_dir)
    tuner_dir.mkdir(parents=True, exist_ok=True)

    normalizers = NormalizationBundle.fit(X_signals, X_scalars, y)
    X_sig_norm = normalizers.transform_signals(X_signals)
    X_sc_norm = normalizers.transform_scalars(X_scalars)
    y_norm = normalizers.transform_target(y)

    unique_groups = np.unique(groups)
    val_group = unique_groups[-1]
    train_mask = groups != val_group
    val_mask = groups == val_group

    hypermodel_fn = build_qepas_hypermodel(
        input_shape=(X_sig_norm.shape[1], X_sig_norm.shape[2]),
        scalar_dim=X_sc_norm.shape[1],
        num_outputs=y_norm.shape[1],
        max_epochs=max_epochs,
        allowed_architectures=allowed_architectures,
    )

    can_resume = resume and _tuner_exists(tuner_dir, project_name)
    if can_resume:
        print(f"Resuming existing KerasTuner project at {tuner_dir / project_name}")

    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel_fn,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        seed=RANDOM_STATE,
        directory=str(tuner_dir),
        project_name=project_name,
        overwrite=not can_resume,
    )

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    tuner.search(
        {"signals": X_sig_norm[train_mask], "scalars": X_sc_norm[train_mask]},
        y_norm[train_mask],
        validation_data=(
            {"signals": X_sig_norm[val_mask], "scalars": X_sc_norm[val_mask]},
            y_norm[val_mask],
        ),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_params = {k: best_hps.get(k) for k in best_hps.values.keys()}

    return {
        "best_params": best_params,
        "normalizers": normalizers,
        "tuner": tuner,
    }


def _fold_checkpoint_paths(log_dir: Path | None, fold: int) -> tuple[Path | None, Path | None]:
    """Return (weights_path, config_path) for a fold checkpoint."""
    if log_dir is None:
        return None, None
    path = log_dir / "fold_checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    weights_path = path / f"fold_{fold}.weights.h5"
    config_path = path / f"fold_{fold}_hps.json"
    return weights_path, config_path


def _save_fold_checkpoint(
    model: tf.keras.Model,
    tuner: kt.Tuner,
    log_dir: Path | None,
    fold: int,
):
    weights_path, config_path = _fold_checkpoint_paths(log_dir, fold)
    if weights_path is None or config_path is None:
        return
    model.save_weights(weights_path)
    hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    with open(config_path, "w") as f:
        json.dump({k: hps.get(k) for k in hps.values.keys()}, f, indent=2)
    print(f"Saved fold {fold} weights to {weights_path}")


def _load_fold_checkpoint(
    log_dir: Path | None,
    fold: int,
    input_shape: tuple[int, int],
    scalar_dim: int,
    num_outputs: int,
) -> tf.keras.Model | None:
    weights_path, config_path = _fold_checkpoint_paths(log_dir, fold)
    if weights_path is None or config_path is None:
        return None
    if not weights_path.exists() or not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            best_params = json.load(f)
        # Reconstruct hyperparameters object and build model
        hp = kt.HyperParameters()
        for k, v in best_params.items():
            hp.values[k] = v
        model = build_qepas_hypermodel(
            input_shape=input_shape,
            scalar_dim=scalar_dim,
            num_outputs=num_outputs,
        )(hp)
        model.load_weights(weights_path)
        print(f"Loaded fold {fold} weights from {weights_path}")
        return model
    except Exception as e:
        print(f"Could not load checkpoint for fold {fold}: {e}")
        return None


def train_deep_model(
    X_signals: np.ndarray,
    X_scalars: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tuner_result: Dict[str, any] | None = None,
    epochs: int = 150,
    batch_size: int = 16,
    early_stopping_patience: int = 20,
    log_dir: Path | None = None,
    model_name: str = "DeepLearning",
    resume: bool = False,
) -> ModelResult:
    """Train a deep model with leave-one-concentration-out CV.

    If tuner_result is provided, uses the best hyperparameters. Otherwise trains
    a default simple CNN.

    If resume=True, attempts to load already-trained fold checkpoints from
    log_dir/fold_checkpoints so previously completed folds are skipped.
    """
    _set_seed()

    if tuner_result is not None:
        normalizers = tuner_result["normalizers"]
    else:
        normalizers = NormalizationBundle.fit(X_signals, X_scalars, y)

    X_sig_norm = normalizers.transform_signals(X_signals)
    X_sc_norm = normalizers.transform_scalars(X_scalars)
    y_norm = normalizers.transform_target(y)

    input_shape = (X_sig_norm.shape[1], X_sig_norm.shape[2])
    scalar_dim = X_sc_norm.shape[1]
    num_outputs = y_norm.shape[1]

    unique_groups = np.unique(groups)
    predictions = np.zeros_like(y)
    fold_histories = []

    for fold, g in enumerate(unique_groups):
        if resume:
            loaded_model = _load_fold_checkpoint(
                log_dir, fold, input_shape, scalar_dim, num_outputs
            )
            if loaded_model is not None:
                print(f"Deep model fold {fold + 1}/{len(unique_groups)}: loading checkpoint for group {g}")
                val_mask = groups == g
                val_idx = np.where(val_mask)[0]
                pred_norm = loaded_model.predict(
                    {"signals": X_sig_norm[val_idx], "scalars": X_sc_norm[val_idx]},
                    verbose=0,
                )
                predictions[val_idx] = normalizers.inverse_transform_target(pred_norm)
                fold_histories.append(None)
                continue

        print(f"Deep model fold {fold + 1}/{len(unique_groups)}: leaving out group {g}")
        train_mask = groups != g
        val_mask = groups == g
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if tuner_result is not None:
            model = tuner_result["tuner"].hypermodel.build(
                tuner_result["tuner"].get_best_hyperparameters(num_trials=1)[0]
            )
        else:
            from .architectures import build_simple_cnn
            model = build_simple_cnn(
                input_shape=input_shape,
                scalar_dim=scalar_dim,
                num_outputs=num_outputs,
            )
            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["mae"],
            )

        fold_log_dir = log_dir / f"fold_{fold}" if log_dir else None
        cb = _make_callbacks(
            patience=early_stopping_patience,
            log_dir=fold_log_dir,
        )

        history = model.fit(
            {"signals": X_sig_norm[train_idx], "scalars": X_sc_norm[train_idx]},
            y_norm[train_idx],
            validation_data=(
                {"signals": X_sig_norm[val_idx], "scalars": X_sc_norm[val_idx]},
                y_norm[val_idx],
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=0,
        )
        fold_histories.append(history.history)

        _save_fold_checkpoint(model, tuner_result["tuner"], log_dir, fold)

        pred_norm = model.predict(
            {"signals": X_sig_norm[val_idx], "scalars": X_sc_norm[val_idx]},
            verbose=0,
        )
        predictions[val_idx] = normalizers.inverse_transform_target(pred_norm)

    result = compute_metrics(y, predictions)
    result.name = model_name
    return result


def save_tuner_results(tuner_result: Dict[str, any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": tuner_result["best_params"],
        "normalizers": tuner_result["normalizers"].to_dict(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
