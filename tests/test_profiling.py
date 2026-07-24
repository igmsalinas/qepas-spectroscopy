from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qepas_spectroscopy.data import DatasetProfiler, Scan


def _arrays(scale: float) -> dict[str, np.ndarray]:
    phase = np.linspace(-4.0, 4.0, num=32)
    modulus = scale * np.linspace(1.0, 2.0, num=32)
    return {
        "modulo_remuestreado": modulus,
        "fase_remuestreada": phase,
        "X_remuestreada": modulus * np.sin(phase),
        "Y_remuestreada": modulus * np.cos(phase),
        "vector_med_presion": np.array(0.9),
        "vector_cons_presion": np.array(0.9),
        "vector_med_flujo": np.array(110.0),
        "vector_cons_flujo": np.array(1000.0),
        "vector_temp_Vflujo": np.array(30.3),
    }


class DatasetProfilerTests(unittest.TestCase):
    def test_profiles_signal_relationships_and_persists_artifacts(self) -> None:
        scans = [
            Scan("a", "10:00:00", 0, Path("/a"), 0.0, 0.0),
            Scan("b", "11:00:00", 0, Path("/b"), 1.0, 10.0),
        ]
        arrays = {Path("/a"): _arrays(1.0), Path("/b"): _arrays(2.0)}
        profiler = DatasetProfiler(
            scan_source=lambda: iter(scans),
            array_loader=lambda path: arrays[Path(path)],
        )

        profile = profiler.profile()

        self.assertEqual(profile.summary["scan_count"], 2)
        self.assertEqual(profile.summary["group_count"], 2)
        self.assertAlmostEqual(
            profile.summary["modulus_identity_relative_mae"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            profile.summary["phase_xy_circular_mae_radians"],
            0.0,
            places=12,
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = profile.save(tmp)
            self.assertTrue(all(path.is_file() for path in artifacts))


if __name__ == "__main__":
    unittest.main()
