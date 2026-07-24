"""Application service that orchestrates a complete training experiment."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..core.config import FEATURES_DIR, MODELS_DIR, OUTPUT_DIR
from ..data import DatasetPipeline, DatasetViews, EngineeredDataset, RawSignalDataset
from ..features import (
    COMPACT_SPECTROSCOPY_FEATURES,
    RawSignalPreprocessingConfig,
    feature_indices,
)
from ..evaluation import (
    ModelResult,
    save_prediction_artifacts,
    save_results,
    save_seed_group_metrics,
)
from ..models.traditional import (
    ModelTrainer,
    build_gradient_boosting,
    build_nested_pls,
    build_nested_ridge,
    build_random_forest,
    build_ridge,
    build_xgboost,
    tune_xgboost,
)
from ..validation import nested_group_folds, validation_group_split
from .experiments import validate_experiment_id
from ..visualization import (
    plot_feature_importance,
    plot_metrics_bar,
    plot_parity_grid,
)

Reporter = Callable[[str], None]
DEEP_ARCHITECTURE_NAMES = (
    "simple_cnn",
    "resnet1d",
    "tcn",
    "lstm",
    "multiscale_cnn",
    "transformer1d",
    "inception_spectra",
    "dilated_resnet1d",
)


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """All writable locations used by one experiment run."""

    features: Path = FEATURES_DIR
    models: Path = MODELS_DIR
    tuner: Path = OUTPUT_DIR / "keras_tuner_advanced"
    tensorboard: Path = OUTPUT_DIR / "tensorboard"
    run_dir: Path | None = None
    experiment_id: str | None = None

    @classmethod
    def for_experiment(
        cls,
        experiment_id: str,
        *,
        experiments_root: str | Path = OUTPUT_DIR / "experiments",
        resume: bool = False,
    ) -> "TrainingPaths":
        validated_id = validate_experiment_id(experiment_id)
        base = Path(experiments_root) / validated_id
        if resume and not base.is_dir():
            raise FileNotFoundError(f"Experiment does not exist: {base}")
        if not resume and base.exists():
            raise FileExistsError(f"Experiment already exists: {base}")
        return cls(
            features=base / "features",
            models=base / "models",
            tuner=base / "tuner",
            tensorboard=base / "tensorboard",
            run_dir=base,
            experiment_id=validated_id,
        )

    def ensure(self) -> None:
        directories = [self.features, self.models, self.tuner]
        if self.run_dir is not None:
            directories.append(self.run_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    """Framework-independent training command parameters."""

    deep_tuner_trials: int = 20
    deep_tuner_epochs: int = 100
    deep_tuner_executions: int = 1
    deep_protocol: Literal["development", "nested-small-data"] = "development"
    deep_seeds: int = 1
    deep_augmentation_ablation: bool = False
    deep_epochs: int = 150
    signal_length: int = 4096
    resampling_method: Literal["polyphase", "linear"] = "polyphase"
    include_cartesian_signals: bool = False
    deep_batch_size: int = 16
    deep_early_stopping_patience: int = 20
    skip_deep: bool = False
    skip_traditional: bool = False
    skip_xgb_tune: bool = False
    resume: bool = False
    architectures: tuple[str, ...] | None = None
    tensorboard_dir: Path | None = None
    traditional_suite: Literal["baseline", "spectral", "all"] = "baseline"

    def __post_init__(self) -> None:
        positive = {
            "deep_tuner_trials": self.deep_tuner_trials,
            "deep_tuner_epochs": self.deep_tuner_epochs,
            "deep_tuner_executions": self.deep_tuner_executions,
            "deep_seeds": self.deep_seeds,
            "deep_epochs": self.deep_epochs,
            "signal_length": self.signal_length,
            "deep_batch_size": self.deep_batch_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Training options must be positive: {invalid}")
        if self.deep_early_stopping_patience < 0:
            raise ValueError("deep_early_stopping_patience cannot be negative")
        if self.skip_deep and self.skip_traditional:
            raise ValueError("At least one model family must be enabled")
        if self.resampling_method not in {"polyphase", "linear"}:
            raise ValueError(
                f"Unknown resampling method: {self.resampling_method}"
            )
        if self.traditional_suite not in {"baseline", "spectral", "all"}:
            raise ValueError(
                f"Unknown traditional suite: {self.traditional_suite}"
            )
        if self.deep_protocol not in {"development", "nested-small-data"}:
            raise ValueError(f"Unknown deep protocol: {self.deep_protocol}")
        if self.deep_augmentation_ablation and self.include_cartesian_signals:
            raise ValueError(
                "Deep augmentation requires the three-channel signal view"
            )
        if self.deep_protocol == "nested-small-data" and self.architectures:
            if set(self.architectures) != {"inception_spectra"}:
                raise ValueError(
                    "nested-small-data only supports inception_spectra"
                )
        if self.architectures:
            unknown = set(self.architectures) - set(DEEP_ARCHITECTURE_NAMES)
            if unknown:
                raise ValueError(
                    f"Unknown deep architectures: {sorted(unknown)}"
                )

    @property
    def raw_preprocessing(self) -> RawSignalPreprocessingConfig:
        return RawSignalPreprocessingConfig(
            length=self.signal_length,
            resampling=self.resampling_method,
            include_cartesian=self.include_cartesian_signals,
        )


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """In-memory outcome returned by the training application service."""

    results: tuple[ModelResult, ...]
    summary: pd.DataFrame
    output_dir: Path
    experiment_id: str | None = None


@dataclass(slots=True)
class TrainingPipeline:
    """Coordinate data assembly, training, evaluation, and artifacts."""

    dataset_pipeline: DatasetPipeline = field(default_factory=DatasetPipeline)
    paths: TrainingPaths = field(default_factory=TrainingPaths)
    reporter: Reporter = print

    def run(self, options: TrainingOptions) -> TrainingRun:
        self.paths.ensure()
        started_at = datetime.now(timezone.utc).isoformat()
        self._write_manifest("running", options, started_at=started_at)
        try:
            raw: RawSignalDataset | None = None
            if options.skip_deep:
                engineered = self._build_engineered_dataset()
            else:
                views = self._build_dataset_views(options)
                engineered = views.engineered
                raw = views.raw

            self._persist_split_assignments(
                engineered,
                include_traditional=not options.skip_traditional,
                include_deep=raw is not None,
                deep_protocol=options.deep_protocol,
            )
            results = (
                []
                if options.skip_traditional
                else self._train_traditional(engineered, options)
            )
            if (
                not options.skip_traditional
                and not options.skip_xgb_tune
                and options.traditional_suite in {"baseline", "all"}
            ):
                self._tune_xgboost(engineered)
            if raw is not None:
                results.extend(self._train_deep(raw, options))

            summary = save_results(results, self.paths.models)
            save_prediction_artifacts(
                results,
                engineered.targets,
                engineered.sample_ids,
                engineered.groups,
                self.paths.models,
            )
            self.reporter("\n=== Summary ===")
            self.reporter(summary.to_string(index=False))
            self._write_plots_and_importances(results, engineered)
            completed_at = datetime.now(timezone.utc).isoformat()
            self._write_manifest(
                "completed",
                options,
                started_at=started_at,
                completed_at=completed_at,
                summary=summary,
                sample_count=len(engineered.features),
                group_count=len(np.unique(engineered.groups)),
            )
            output_dir = self.paths.run_dir or self.paths.models
            self.reporter(f"All outputs saved to {output_dir}")
            return TrainingRun(
                results=tuple(results),
                summary=summary,
                output_dir=output_dir,
                experiment_id=self.paths.experiment_id,
            )
        except Exception as error:
            self._write_manifest(
                "failed",
                options,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(error).__name__}: {error}",
            )
            raise

    def _write_manifest(
        self,
        status: str,
        options: TrainingOptions,
        *,
        started_at: str,
        completed_at: str | None = None,
        summary: pd.DataFrame | None = None,
        sample_count: int | None = None,
        group_count: int | None = None,
        error: str | None = None,
    ) -> None:
        if self.paths.run_dir is None:
            return
        payload: dict[str, object] = {
            "schema_version": 1,
            "experiment_id": self.paths.experiment_id,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "options": asdict(options),
            "sample_count": sample_count,
            "group_count": group_count,
        }
        if summary is not None:
            payload["metrics"] = summary.to_dict(orient="records")
        if error is not None:
            payload["error"] = error
        target = self.paths.run_dir / "run_manifest.json"
        temporary = self.paths.run_dir / "run_manifest.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
        temporary.replace(target)

    def _persist_split_assignments(
        self,
        dataset: EngineeredDataset,
        *,
        include_traditional: bool,
        include_deep: bool,
        deep_protocol: str = "development",
    ) -> None:
        records: list[dict[str, object]] = []
        groups = np.asarray(dataset.groups)
        sample_ids = np.asarray(dataset.sample_ids)
        traditional_groups = np.unique(groups) if include_traditional else ()
        for fold, test_group in enumerate(traditional_groups):
            for index, sample_id in enumerate(sample_ids):
                records.append(
                    {
                        "protocol": "traditional_logo",
                        "fold": fold,
                        "sample_id": str(sample_id),
                        "group": str(groups[index]),
                        "role": (
                            "test" if groups[index] == test_group else "fit"
                        ),
                    }
                )

        if include_deep and deep_protocol == "development":
            tuning_split = validation_group_split(groups)
            fit_set = set(tuning_split.fit_indices.tolist())
            for index, sample_id in enumerate(sample_ids):
                records.append(
                    {
                        "protocol": "deep_tuner_holdout",
                        "fold": 0,
                        "sample_id": str(sample_id),
                        "group": str(groups[index]),
                        "role": "fit" if index in fit_set else "validation",
                    }
                )
        if include_deep:
            for fold in nested_group_folds(groups):
                validation_set = set(fold.validation_indices.tolist())
                test_set = set(fold.test_indices.tolist())
                for index, sample_id in enumerate(sample_ids):
                    role = "fit"
                    if index in validation_set:
                        role = "validation"
                    elif index in test_set:
                        role = "test"
                    records.append(
                        {
                            "protocol": "deep_nested_logo",
                            "fold": fold.fold,
                            "sample_id": str(sample_id),
                            "group": str(groups[index]),
                            "role": role,
                        }
                    )

        destination = self.paths.run_dir or self.paths.models
        pd.DataFrame.from_records(records).to_csv(
            destination / "splits.csv",
            index=False,
        )

    def _build_engineered_dataset(self) -> EngineeredDataset:
        self.reporter("Extracting and preprocessing engineered features...")
        dataset = self.dataset_pipeline.build_engineered()
        self._persist_engineered(dataset)
        return dataset

    def _build_dataset_views(self, options: TrainingOptions) -> DatasetViews:
        self.reporter(
            "Extracting engineered and raw views in one scan pass..."
        )
        config = options.raw_preprocessing
        views = self.dataset_pipeline.build_views(config)
        self._assert_aligned(views.engineered, views.raw)
        self._persist_engineered(views.engineered)
        self._persist_raw(views.raw)
        return views

    def _persist_engineered(self, dataset: EngineeredDataset) -> None:
        dataset.frame.to_csv(
            self.paths.features / "feature_table.csv",
            index=False,
        )
        # These full-data statistics are inference artifacts only. Evaluation
        # fits Ridge scaling independently inside every training fold.
        scaler = StandardScaler().fit(dataset.features)
        pd.Series(scaler.mean_, index=dataset.feature_names).to_csv(
            self.paths.features / "scaler_mean.csv"
        )
        pd.Series(scaler.scale_, index=dataset.feature_names).to_csv(
            self.paths.features / "scaler_scale.csv"
        )

    def _train_traditional(
        self,
        dataset: EngineeredDataset,
        options: TrainingOptions,
    ) -> list[ModelResult]:
        results: list[ModelResult] = []
        if options.traditional_suite in {"baseline", "all"}:
            trainers = [
                build_ridge(),
                build_random_forest(),
                build_xgboost(),
                build_gradient_boosting(),
            ]
            for trainer in trainers:
                self.reporter(f"Training {trainer.name}...")
                result = trainer.cross_val_predict(
                    dataset.features,
                    dataset.targets,
                    dataset.groups,
                )
                results.append(result)
                self.reporter(json.dumps(result.to_dict(), indent=2))

        if options.traditional_suite in {"spectral", "all"}:
            compact = feature_indices(
                dataset.feature_names,
                COMPACT_SPECTROSCOPY_FEATURES,
            )
            searches = (
                (
                    build_nested_ridge("RidgeNestedFull"),
                    dataset.features,
                    "full",
                ),
                (
                    build_nested_ridge("RidgeNestedCompact"),
                    dataset.features[:, compact],
                    "compact_spectroscopy",
                ),
                (
                    build_nested_pls("PLSNestedFull"),
                    dataset.features,
                    "full",
                ),
                (
                    build_nested_pls("PLSNestedCompact"),
                    dataset.features[:, compact],
                    "compact_spectroscopy",
                ),
            )
            tuning: dict[str, object] = {
                "protocol": "nested_leave_one_group_out",
                "selection_metric": "mean_target_range_normalized_rmse",
                "models": {},
            }
            model_tuning = tuning["models"]
            if not isinstance(model_tuning, dict):
                raise RuntimeError("Invalid nested tuning artifact state")
            for trainer, features, feature_set in searches:
                self.reporter(f"Training {trainer.name} with nested grouped tuning...")
                result = trainer.cross_val_predict(
                    features,
                    dataset.targets,
                    dataset.groups,
                )
                results.append(result)
                model_tuning[trainer.name] = {
                    "feature_set": feature_set,
                    "feature_count": int(features.shape[1]),
                    "outer_folds": trainer.selection_records(),
                }
                self.reporter(json.dumps(result.to_dict(), indent=2))
            with (self.paths.models / "nested_traditional_search.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(tuning, stream, indent=2)

        return results

    def _tune_xgboost(self, dataset: EngineeredDataset) -> None:
        self.reporter("Tuning XGBoost hyperparameters...")
        result = tune_xgboost(
            dataset.features,
            dataset.targets,
            dataset.groups,
        )
        self.reporter(
            f"Best XGBoost params: {result['best_params']} "
            f"(grouped CV RMSE={result['best_score']:.4f})"
        )
        with (self.paths.models / "xgb_tuning.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(result, stream, indent=2, default=str)

    def _persist_raw(self, dataset: RawSignalDataset) -> None:
        np.savez_compressed(
            self.paths.features / "raw_dataset.npz",
            signals=dataset.signals,
            scalars=dataset.scalars,
            y=dataset.targets,
            groups=dataset.groups,
            sample_ids=dataset.sample_ids,
            signal_names=np.asarray(dataset.signal_names),
            scalar_names=np.asarray(dataset.scalar_names),
            preprocessing_id=np.asarray(dataset.preprocessing_id),
        )
        manifest = {
            "version": 1,
            "preprocessing_id": dataset.preprocessing_id,
            "sample_count": len(dataset.signals),
            "signal_length": dataset.signals.shape[1],
            "signal_names": list(dataset.signal_names),
            "scalar_names": list(dataset.scalar_names),
        }
        with (self.paths.features / "preprocessing_manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2)

    def _train_deep(
        self,
        dataset: RawSignalDataset,
        options: TrainingOptions,
    ) -> list[ModelResult]:
        # Delayed import keeps traditional workflows and CLI help lightweight.
        from ..models.deep_learning import (
            save_tuner_results,
            train_deep_model,
            train_fully_nested_inception,
            tune_deep_model,
        )

        if options.deep_protocol == "nested-small-data":
            self.reporter(
                "Running fully nested small-Inception tuning and seed ablation..."
            )
            outcome = train_fully_nested_inception(
                dataset.signals,
                dataset.scalars,
                dataset.targets,
                dataset.groups,
                max_trials=options.deep_tuner_trials,
                tuner_epochs=options.deep_tuner_epochs,
                executions_per_trial=options.deep_tuner_executions,
                epochs=options.deep_epochs,
                batch_size=options.deep_batch_size,
                early_stopping_patience=options.deep_early_stopping_patience,
                seeds=options.deep_seeds,
                augmentation_ablation=options.deep_augmentation_ablation,
                tuner_dir=self.paths.tuner,
                artifact_dir=self.paths.models,
                log_dir=options.tensorboard_dir or self.paths.tensorboard,
                resume=options.resume,
                preprocessing_id=dataset.preprocessing_id,
            )
            with (self.paths.models / "deep_nested_search.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(outcome.search, stream, indent=2, default=str)
            np.savez_compressed(
                self.paths.models / "deep_seed_predictions.npz",
                **outcome.seed_predictions,
            )
            rows: list[dict[str, object]] = []
            for model_name, seed_records in outcome.search["seed_metrics"].items():
                for seed_index, seed_record in enumerate(seed_records):
                    for target, metrics in seed_record["per_target"].items():
                        rows.append(
                            {
                                "model": model_name,
                                "seed": seed_index + 1,
                                "target": target,
                                **metrics,
                            }
                        )
            pd.DataFrame.from_records(rows).to_csv(
                self.paths.models / "deep_seed_metrics.csv",
                index=False,
            )
            save_seed_group_metrics(
                outcome.seed_predictions,
                dataset.targets,
                dataset.groups,
                self.paths.models,
            )
            for result in outcome.results:
                self.reporter(json.dumps(result.to_dict(), indent=2))
            return list(outcome.results)

        self.reporter("Tuning deep model with KerasTuner...")
        tuner_result = tune_deep_model(
            dataset.signals,
            dataset.scalars,
            dataset.targets,
            dataset.groups,
            max_trials=options.deep_tuner_trials,
            executions_per_trial=options.deep_tuner_executions,
            project_name="qepas_advanced_tuning",
            tuner_dir=self.paths.tuner,
            max_epochs=options.deep_tuner_epochs,
            batch_size=options.deep_batch_size,
            early_stopping_patience=max(
                8, options.deep_early_stopping_patience // 2
            ),
            allowed_architectures=(
                list(options.architectures) if options.architectures else None
            ),
            resume=options.resume,
            preprocessing_id=dataset.preprocessing_id,
        )
        save_tuner_results(
            tuner_result,
            self.paths.models / "deep_tuner_best.json",
        )
        self.reporter(f"Best deep params: {tuner_result['best_params']}")

        self.reporter("Training deep model with nested grouped evaluation...")
        result = train_deep_model(
            dataset.signals,
            dataset.scalars,
            dataset.targets,
            dataset.groups,
            tuner_result=tuner_result,
            epochs=options.deep_epochs,
            batch_size=options.deep_batch_size,
            early_stopping_patience=options.deep_early_stopping_patience,
            log_dir=options.tensorboard_dir or self.paths.tensorboard,
            model_name="DeepLearning",
            resume=options.resume,
            preprocessing_id=dataset.preprocessing_id,
        )
        self.reporter(json.dumps(result.to_dict(), indent=2))
        return [result]

    @staticmethod
    def _assert_aligned(
        engineered: EngineeredDataset,
        raw: RawSignalDataset,
    ) -> None:
        if not np.array_equal(engineered.sample_ids, raw.sample_ids):
            raise ValueError("Engineered and raw sample ordering does not match")
        if not np.array_equal(engineered.targets, raw.targets):
            raise ValueError("Engineered and raw target ordering does not match")
        if not np.array_equal(engineered.groups, raw.groups):
            raise ValueError("Engineered and raw group ordering does not match")

    def _write_plots_and_importances(
        self,
        results: list[ModelResult],
        dataset: EngineeredDataset,
    ) -> None:
        plot_metrics_bar(
            results,
            self.paths.models / "metrics_comparison.png",
        )
        plot_parity_grid(
            results,
            dataset.targets,
            dataset.groups,
            self.paths.models / "parity_grid.png",
        )
        evaluated = {result.name for result in results}
        for trainer in (build_random_forest(), build_xgboost()):
            if trainer.name in evaluated:
                self._write_feature_importance(trainer, dataset)

    def _write_feature_importance(
        self,
        trainer: ModelTrainer,
        dataset: EngineeredDataset,
    ) -> None:
        model = trainer.fit(dataset.features, dataset.targets)
        importances = np.asarray(model.feature_importances_)
        if importances.ndim != 1 or len(importances) != len(
            dataset.feature_names
        ):
            raise ValueError(
                f"Unexpected {trainer.name} feature importance shape: "
                f"{importances.shape}"
            )
        table = pd.DataFrame(
            {"feature": dataset.feature_names, "mean": importances}
        ).sort_values("mean", ascending=False)
        stem = trainer.name.lower()
        table.to_csv(
            self.paths.models / f"{stem}_importance.csv",
            index=False,
        )
        plot_feature_importance(
            table,
            self.paths.models / f"{stem}_importance.png",
            title=f"{trainer.name} feature importance",
        )
