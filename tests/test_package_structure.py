from __future__ import annotations

import unittest

from qepas_spectroscopy.application import TrainingPipeline
from qepas_spectroscopy.core import config
from qepas_spectroscopy.data import DatasetPipeline, NormalizationBundle
from qepas_spectroscopy.evaluation import compute_metrics
from qepas_spectroscopy.interfaces.cli import app
from qepas_spectroscopy.models.traditional import build_ridge
from qepas_spectroscopy.validation import nested_group_folds
from qepas_spectroscopy.visualization import plot_parity_grid


class PackageStructureTests(unittest.TestCase):
    def test_canonical_packages_expose_required_public_api(self) -> None:
        public_objects = (
            TrainingPipeline,
            DatasetPipeline,
            compute_metrics,
            app,
            NormalizationBundle,
            build_ridge,
            nested_group_folds,
            plot_parity_grid,
        )
        self.assertTrue(all(public_objects))

    def test_core_config_resolves_repository_root(self) -> None:
        self.assertTrue((config.PROJECT_ROOT / "pyproject.toml").is_file())
        self.assertTrue((config.PROJECT_ROOT / "qepas_spectroscopy").is_dir())


if __name__ == "__main__":
    unittest.main()
