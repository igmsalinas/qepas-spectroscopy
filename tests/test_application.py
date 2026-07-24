from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qepas_spectroscopy.application import (
    TrainingOptions,
    TrainingPaths,
    TrainingPipeline,
)
from qepas_spectroscopy.data import EngineeredDataset
from qepas_spectroscopy.evaluation import Metrics, ModelResult


class _DatasetPipelineStub:
    def __init__(self, engineered: EngineeredDataset) -> None:
        self.engineered = engineered
        self.raw_requested = False

    def build_engineered(self) -> EngineeredDataset:
        return self.engineered

    def build_raw(self, signal_length: int):
        self.raw_requested = True
        raise AssertionError("raw data should not be requested")


class TrainingApplicationTests(unittest.TestCase):
    def test_training_options_validate_at_application_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            TrainingOptions(signal_length=0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            TrainingOptions(architectures=("not-a-model",))
        with self.assertRaisesRegex(ValueError, "resampling"):
            TrainingOptions(resampling_method="nearest")
        with self.assertRaisesRegex(ValueError, "traditional suite"):
            TrainingOptions(traditional_suite="unknown")
        with self.assertRaisesRegex(ValueError, "model family"):
            TrainingOptions(skip_deep=True, skip_traditional=True)
        with self.assertRaisesRegex(ValueError, "deep protocol"):
            TrainingOptions(deep_protocol="not-a-protocol")
        with self.assertRaisesRegex(ValueError, "positive"):
            TrainingOptions(deep_seeds=0)
        with self.assertRaisesRegex(ValueError, "only supports"):
            TrainingOptions(
                deep_protocol="nested-small-data",
                architectures=("simple_cnn",),
            )
        with self.assertRaisesRegex(ValueError, "three-channel"):
            TrainingOptions(
                deep_augmentation_ablation=True,
                include_cartesian_signals=True,
            )

        valid = TrainingOptions(
            deep_protocol="nested-small-data",
            architectures=("inception_spectra",),
            deep_seeds=5,
            deep_augmentation_ablation=True,
        )
        self.assertEqual(valid.deep_seeds, 5)

    def test_skip_options_run_only_the_traditional_application_path(self) -> None:
        frame = pd.DataFrame(
            {
                "feature": [0.0, 1.0],
                "sample_id": ["a/0", "b/0"],
                "folder": ["folder-a", "folder-b"],
                "time": ["a", "b"],
                "N": [0, 0],
                "13CO2": [0.0, 1.0],
                "12CO2": [0.0, 10.0],
            }
        )
        dataset = EngineeredDataset(frame, ("feature",))
        data_pipeline = _DatasetPipelineStub(dataset)
        predictions = dataset.targets.copy()
        metrics = {
            "13CO2": Metrics(0.0, 0.0, 1.0),
            "12CO2": Metrics(0.0, 0.0, 1.0),
        }
        result = ModelResult("Stub", metrics, 0.0, predictions)
        messages: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = TrainingPaths(
                features=root / "features",
                models=root / "models",
                tuner=root / "tuner",
                tensorboard=root / "tensorboard",
            )
            pipeline = TrainingPipeline(
                dataset_pipeline=data_pipeline,
                paths=paths,
                reporter=messages.append,
            )
            with (
                patch.object(
                    TrainingPipeline,
                    "_train_traditional",
                    return_value=[result],
                ),
                patch.object(TrainingPipeline, "_write_plots_and_importances"),
            ):
                run = pipeline.run(
                    TrainingOptions(skip_deep=True, skip_xgb_tune=True)
                )

            self.assertTrue((paths.features / "feature_table.csv").is_file())
            self.assertTrue((paths.models / "metrics.json").is_file())
            self.assertTrue((paths.models / "predictions.csv").is_file())
            self.assertTrue((paths.models / "group_metrics.csv").is_file())
            self.assertTrue((paths.models / "splits.csv").is_file())
            self.assertEqual(run.results, (result,))
            self.assertFalse(data_pipeline.raw_requested)
            self.assertTrue(any("All outputs" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
