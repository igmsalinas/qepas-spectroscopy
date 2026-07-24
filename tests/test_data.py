from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qepas_spectroscopy.data import (
    MissingScanDataError,
    extract_time,
    iter_scans,
    load_raw_arrays,
)


class DataLoadingTests(unittest.TestCase):
    def test_extract_time_from_measurement_folder(self) -> None:
        folder = "MEDICIONES_REALIZADAS_a_fecha_de_lunes_13_abril_2026_13_43_08"

        self.assertEqual(extract_time(folder), "13:43:08")
        self.assertIsNone(extract_time("unrelated"))

    def test_iter_scans_accepts_injected_labels_and_scan_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            folder = base_dir / "MEDICIONES_TEST_10_20_30"
            for index in range(3):
                (folder / f"modulo_fase_X_Y_barrido_N_{index}").mkdir(
                    parents=True,
                    exist_ok=True,
                )

            labels = {"10:20:30": {"13CO2": 1.5, "12CO2": 2.5}}
            scans = list(
                iter_scans(base_dir, labels=labels, scans_per_group=2)
            )
            all_scans = list(
                iter_scans(base_dir, labels=labels, scans_per_group=None)
            )

        self.assertEqual([scan.n for scan in scans], [0, 1])
        self.assertEqual([scan.n for scan in all_scans], [0, 1, 2])
        self.assertTrue(all(scan.path.name.startswith("modulo_fase") for scan in scans))
        np.testing.assert_array_equal(scans[0].labels, np.array([1.5, 2.5], dtype=np.float32))

    def test_load_raw_arrays_reports_all_missing_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan_dir = Path(tmp)
            np.save(scan_dir / "modulo_remuestreado.npy", np.arange(4))

            with self.assertRaises(MissingScanDataError) as raised:
                load_raw_arrays(scan_dir)

        self.assertIn("fase_remuestreada.npy", str(raised.exception))
        self.assertIn(str(scan_dir), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
