"""Typer adapter for the training application service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from ..application import (
    DEEP_ARCHITECTURE_NAMES,
    TrainingOptions,
    TrainingPaths,
    TrainingPipeline,
    create_experiment_id,
)
from ..core.config import OUTPUT_DIR

app = typer.Typer()


@app.command()
def train(
    experiment_label: str = typer.Option(
        "experiment",
        help="Human-readable label appended to a generated run ID",
    ),
    run_id: str | None = typer.Option(
        None,
        help="Exact run ID to create or resume",
    ),
    experiments_dir: Path = typer.Option(
        OUTPUT_DIR / "experiments",
        help="Root directory containing isolated experiment runs",
    ),
    traditional_suite: Literal["baseline", "spectral", "all"] = typer.Option(
        "baseline",
        help="Traditional models: fixed baselines, nested Ridge/PLS, or both",
    ),
    deep_tuner_trials: int = typer.Option(
        20, min=1, help="Number of KerasTuner trials"
    ),
    deep_tuner_epochs: int = typer.Option(
        100, min=1, help="Maximum epochs for each tuner trial"
    ),
    deep_tuner_executions: int = typer.Option(
        1,
        min=1,
        help="Repeated model initializations averaged per tuner trial",
    ),
    deep_protocol: Literal[
        "development", "nested-small-data"
    ] = typer.Option(
        "development",
        help="Deep model-selection protocol",
    ),
    deep_seeds: int = typer.Option(
        1,
        min=1,
        help="Independent seeds per fully nested outer fold",
    ),
    deep_augmentation_ablation: bool = typer.Option(
        False,
        help="Pair unaugmented and fit-only augmented seed ensembles",
    ),
    deep_epochs: int = typer.Option(
        150, min=1, help="Maximum epochs for each deep model fold"
    ),
    signal_length: int = typer.Option(
        4096, min=1, help="Fixed resampled signal length"
    ),
    resampling_method: Literal["polyphase", "linear"] = typer.Option(
        "polyphase",
        help="Signal resampling: polyphase (anti-aliased) or linear",
    ),
    include_cartesian_signals: bool = typer.Option(
        False,
        help="Also feed redundant X/Y channels to deep models",
    ),
    deep_batch_size: int = typer.Option(
        16, min=1, help="Batch size for deep models"
    ),
    deep_early_stopping_patience: int = typer.Option(
        20, min=0, help="Early-stopping patience"
    ),
    skip_deep: bool = typer.Option(False, help="Skip deep-learning models"),
    skip_traditional: bool = typer.Option(
        False,
        help="Skip traditional models and run only deep learning",
    ),
    skip_xgb_tune: bool = typer.Option(
        False, help="Skip XGBoost hyperparameter tuning"
    ),
    resume: bool = typer.Option(
        False, help="Resume compatible tuner and fold checkpoints"
    ),
    architectures: list[str] | None = typer.Option(
        None,
        help=(
            f"Architectures to search. Choices: "
            f"{', '.join(DEEP_ARCHITECTURE_NAMES)}. "
            "Default: fast CNN-based set."
        ),
    ),
    tensorboard_dir: Path | None = typer.Option(
        None,
        help="Override the run-local TensorBoard/checkpoint directory",
    ),
) -> None:
    """Train, evaluate, and persist one isolated experiment run."""
    if resume and run_id is None:
        raise typer.BadParameter("--resume requires an explicit --run-id")
    experiment_id = run_id or create_experiment_id(experiment_label)
    paths = TrainingPaths.for_experiment(
        experiment_id,
        experiments_root=experiments_dir,
        resume=resume,
    )
    typer.echo(f"Experiment ID: {experiment_id}")
    options = TrainingOptions(
        deep_tuner_trials=deep_tuner_trials,
        deep_tuner_epochs=deep_tuner_epochs,
        deep_tuner_executions=deep_tuner_executions,
        deep_protocol=deep_protocol,
        deep_seeds=deep_seeds,
        deep_augmentation_ablation=deep_augmentation_ablation,
        deep_epochs=deep_epochs,
        signal_length=signal_length,
        resampling_method=resampling_method,
        include_cartesian_signals=include_cartesian_signals,
        deep_batch_size=deep_batch_size,
        deep_early_stopping_patience=deep_early_stopping_patience,
        skip_deep=skip_deep,
        skip_traditional=skip_traditional,
        skip_xgb_tune=skip_xgb_tune,
        resume=resume,
        architectures=tuple(architectures) if architectures else None,
        tensorboard_dir=tensorboard_dir,
        traditional_suite=traditional_suite,
    )
    TrainingPipeline(
        paths=paths,
        reporter=typer.echo,
    ).run(options)


if __name__ == "__main__":
    app()
