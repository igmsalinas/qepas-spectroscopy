"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.preprocessing import StandardScaler

from .config import FEATURES_DIR, MODELS_DIR, TARGETS, ensure_dirs
from .data import iter_scans, load_raw_arrays
from .evaluation import save_results
from .features import build_features, build_raw_signal_features
from .models import (
    build_ridge,
    build_random_forest,
    build_xgboost,
    build_gradient_boosting,
    tune_xgboost,
)
from .models.dl import tune_deep_model, train_deep_model, save_tuner_results, ARCHITECTURES
from .plotting import plot_parity_grid, plot_feature_importance, plot_metrics_bar

app = typer.Typer()


def _build_engineered_dataset():
    records = []
    for scan in iter_scans():
        arrays = load_raw_arrays(scan.path)
        feats = build_features(arrays)
        feats["time"] = scan.time
        feats["N"] = scan.n
        feats["13CO2"] = scan.label_13co2
        feats["12CO2"] = scan.label_12co2
        records.append(feats)
    return pd.DataFrame(records)


def _build_raw_dataset(signal_length: int = 4096):
    signal_list, scalar_list, y_list, groups_list = [], [], [], []
    for scan in iter_scans():
        arrays = load_raw_arrays(scan.path)
        raw = build_raw_signal_features(arrays, length=signal_length)
        signal_list.append(raw["signals"])
        scalar_list.append(raw["scalars"])
        y_list.append(scan.labels)
        groups_list.append(scan.time)
    return (
        np.stack(signal_list),
        np.stack(scalar_list),
        np.stack(y_list),
        np.array(groups_list),
    )


@app.command()
def train(
    deep_tuner_trials: int = typer.Option(20, help="Number of KerasTuner trials"),
    deep_epochs: int = typer.Option(150, help="Max epochs for deep model per fold"),
    signal_length: int = typer.Option(4096, help="Downsampled signal length"),
    deep_batch_size: int = typer.Option(16, help="Batch size for deep model"),
    deep_early_stopping_patience: int = typer.Option(20, help="Early stopping patience"),
    skip_deep: bool = typer.Option(False, help="Skip deep learning model"),
    skip_xgb_tune: bool = typer.Option(False, help="Skip XGBoost hyperparameter tuning"),
    resume: bool = typer.Option(False, help="Resume previous deep tuner/CV training if available"),
    architectures: list[str] | None = typer.Option(
        None,
        help=f"Architectures to search. Choices: {', '.join(ARCHITECTURES.keys())}. Default: fast CNN-based set.",
    ),
    tensorboard_dir: str | None = typer.Option("outputs/tensorboard", help="TensorBoard log directory (also used for fold checkpoints)"),
):
    """Train all models and save results."""
    ensure_dirs()

    # --- Engineered features ---
    typer.echo("Building engineered feature dataset...")
    df = _build_engineered_dataset()
    df.to_csv(FEATURES_DIR / "feature_table.csv", index=False)

    feature_cols = [c for c in df.columns if c not in TARGETS + ["time", "N"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGETS].values.astype(np.float32)
    groups = df["time"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pd.Series(scaler.mean_, index=feature_cols).to_csv(FEATURES_DIR / "scaler_mean.csv")
    pd.Series(scaler.scale_, index=feature_cols).to_csv(FEATURES_DIR / "scaler_scale.csv")

    results = []

    for trainer in [build_ridge(), build_random_forest(), build_xgboost(), build_gradient_boosting()]:
        typer.echo(f"Training {trainer.name}...")
        X_in = X_scaled if trainer.name in {"Ridge"} else X
        result = trainer.cross_val_predict(X_in, y, groups)
        results.append(result)
        typer.echo(json.dumps(result.to_dict(), indent=2))

    if not skip_xgb_tune:
        typer.echo("Tuning XGBoost hyperparameters...")
        xgb_tuned = tune_xgboost(X, y, groups)
        typer.echo(f"Best XGBoost params: {xgb_tuned['best_params']} (val RMSE={xgb_tuned['best_score']:.4f})")
        with open(MODELS_DIR / "xgb_tuning.json", "w") as f:
            json.dump(xgb_tuned, f, indent=2, default=str)

    # --- Deep learning ---
    if not skip_deep:
        typer.echo("Building raw signal dataset...")
        X_sig, X_sc, y_raw, groups_raw = _build_raw_dataset(signal_length=signal_length)
        np.savez(
            FEATURES_DIR / "raw_dataset.npz",
            signals=X_sig,
            scalars=X_sc,
            y=y_raw,
            groups=groups_raw,
        )

        tb_dir = Path(tensorboard_dir) if tensorboard_dir else None

        typer.echo("Tuning deep model with KerasTuner...")
        tuner_result = tune_deep_model(
            X_sig,
            X_sc,
            y_raw,
            groups_raw,
            max_trials=deep_tuner_trials,
            project_name="qepas_advanced_tuning",
            tuner_dir=Path("outputs/keras_tuner_advanced"),
            max_epochs=100,
            batch_size=deep_batch_size,
            early_stopping_patience=max(8, deep_early_stopping_patience // 2),
            allowed_architectures=architectures,
            resume=resume,
        )
        save_tuner_results(tuner_result, MODELS_DIR / "deep_tuner_best.json")
        typer.echo(f"Best deep params: {tuner_result['best_params']}")

        typer.echo("Training deep model with leave-one-concentration-out CV...")
        deep_result = train_deep_model(
            X_sig,
            X_sc,
            y_raw,
            groups_raw,
            tuner_result=tuner_result,
            epochs=deep_epochs,
            batch_size=deep_batch_size,
            early_stopping_patience=deep_early_stopping_patience,
            log_dir=tb_dir,
            model_name="DeepLearning",
            resume=resume,
        )
        results.append(deep_result)
        typer.echo(json.dumps(deep_result.to_dict(), indent=2))

    # --- Save and plot ---
    summary = save_results(results, MODELS_DIR)
    typer.echo("\n=== Summary ===")
    typer.echo(summary.to_string(index=False))

    plot_metrics_bar(results, MODELS_DIR / "metrics_comparison.png")

    # For parity grid, collect y and groups depending on whether deep model was run
    if not skip_deep:
        y_plot = y_raw
        groups_plot = groups_raw
    else:
        y_plot = y
        groups_plot = groups
    plot_parity_grid(results, y_plot, groups_plot, MODELS_DIR / "parity_grid.png")

    # Feature importance for tree-based models
    for trainer in [build_random_forest(), build_xgboost()]:
        model = trainer.fit(X, y)
        importances = model.feature_importances_
        if importances.ndim == 1:
            importances = np.column_stack([importances, importances])
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "13CO2": importances[:, 0],
            "12CO2": importances[:, 1],
        })
        imp_df["mean"] = imp_df[["13CO2", "12CO2"]].mean(axis=1)
        imp_df = imp_df.sort_values("mean", ascending=False)
        imp_df.to_csv(MODELS_DIR / f"{trainer.name.lower()}_importance.csv", index=False)
        plot_feature_importance(imp_df, MODELS_DIR / f"{trainer.name.lower()}_importance.png", title=f"{trainer.name} feature importance")

    typer.echo(f"All outputs saved to {MODELS_DIR}")


if __name__ == "__main__":
    app()
