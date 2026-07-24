from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from qepas_spectroscopy.application import (
    TrainingPaths,
    create_experiment_id,
    validate_experiment_id,
)


class ExperimentInfrastructureTests(unittest.TestCase):
    def test_generated_ids_are_stable_and_filesystem_safe(self) -> None:
        instant = datetime(2026, 7, 24, 12, 30, 1, 123456, tzinfo=timezone.utc)

        experiment_id = create_experiment_id(
            "Ridge baseline #1",
            now=instant,
        )

        self.assertEqual(
            experiment_id,
            "20260724T123001123456Z-ridge-baseline-1",
        )
        self.assertEqual(validate_experiment_id(experiment_id), experiment_id)
        with self.assertRaisesRegex(ValueError, "only"):
            validate_experiment_id("../escape")

    def test_training_paths_isolate_runs_and_reject_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = TrainingPaths.for_experiment(
                "run-001",
                experiments_root=root,
            )
            paths.ensure()

            self.assertEqual(paths.run_dir, root / "run-001")
            self.assertTrue(paths.features.is_dir())
            with self.assertRaises(FileExistsError):
                TrainingPaths.for_experiment(
                    "run-001",
                    experiments_root=root,
                )
            resumed = TrainingPaths.for_experiment(
                "run-001",
                experiments_root=root,
                resume=True,
            )
            self.assertEqual(resumed.run_dir, paths.run_dir)


if __name__ == "__main__":
    unittest.main()
