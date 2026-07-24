from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from qepas_spectroscopy.data import DatasetPipeline, RawSignalDataset, Scan
from qepas_spectroscopy.features import RawSignalPreprocessingConfig


def _scan_data(offset: float) -> dict[str, np.ndarray]:
    signal = np.linspace(offset, offset + 1.0, num=32)
    return {
        "modulo_remuestreado": signal,
        "fase_remuestreada": signal - 0.5,
        "X_remuestreada": signal * 2,
        "Y_remuestreada": signal * -1,
        "vector_med_presion": np.array(offset + 1),
        "vector_cons_presion": np.array(offset + 2),
        "vector_med_flujo": np.array(offset + 3),
        "vector_cons_flujo": np.array(offset + 4),
        "vector_temp_Vflujo": np.array(offset + 5),
    }


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scans = [
            Scan("folder-a", "10:00:00", 0, Path("/scan/a"), 1.0, 10.0),
            Scan("folder-b", "11:00:00", 0, Path("/scan/b"), 2.0, 20.0),
        ]
        self.arrays = {
            Path("/scan/a"): _scan_data(0.0),
            Path("/scan/b"): _scan_data(1.0),
        }
        self.load_count = 0

        def load(path: str | Path) -> dict[str, np.ndarray]:
            self.load_count += 1
            return self.arrays[Path(path)]

        self.pipeline = DatasetPipeline(
            scan_source=lambda: iter(self.scans),
            array_loader=load,
        )

    def test_builds_engineered_dataset_with_traceable_samples(self) -> None:
        dataset = self.pipeline.build_engineered()

        self.assertEqual(dataset.features.shape[0], 2)
        self.assertEqual(dataset.targets.shape, (2, 2))
        self.assertEqual(dataset.groups.tolist(), ["10:00:00", "11:00:00"])
        self.assertEqual(
            dataset.sample_ids.tolist(),
            ["10:00:00/scan-000", "11:00:00/scan-000"],
        )
        self.assertNotIn("13CO2", dataset.feature_names)

    def test_builds_both_aligned_views_in_one_io_pass(self) -> None:
        views = self.pipeline.build_views(
            RawSignalPreprocessingConfig(length=8, resampling="linear")
        )

        self.assertEqual(views.raw.signals.shape, (2, 8, 3))
        self.assertEqual(views.raw.scalars.shape, (2, 5))
        self.assertIn("resampling=linear", views.raw.preprocessing_id)
        np.testing.assert_array_equal(
            views.engineered.sample_ids,
            views.raw.sample_ids,
        )
        self.assertEqual(self.load_count, len(self.scans))

    def test_raw_dataset_rejects_misaligned_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of samples"):
            RawSignalDataset(
                signals=np.zeros((2, 8, 3)),
                scalars=np.zeros((1, 5)),
                targets=np.zeros((2, 2)),
                groups=np.array(["a", "b"]),
                sample_ids=np.array(["a/0", "b/0"]),
                signal_names=("modulus", "phase_sin", "phase_cos"),
                preprocessing_id="test-pipeline-v1",
            )


if __name__ == "__main__":
    unittest.main()
