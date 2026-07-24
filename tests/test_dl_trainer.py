from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qepas_spectroscopy.models.deep_learning.trainer import (
    _make_callbacks,
    _tuner_exists,
    save_tuner_results,
)


class DeepTrainerInfrastructureTests(unittest.TestCase):
    def test_early_stopping_uses_cross_loss_comparable_metric(self) -> None:
        callback = _make_callbacks(patience=2)[0]

        self.assertEqual(callback.monitor, "val_mae")

    def test_tuner_resume_detects_zero_padded_trial_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tuner_dir = Path(tmp)
            project = tuner_dir / "experiment"
            (project / "trial_00").mkdir(parents=True)
            (project / "oracle.json").write_text("{}", encoding="utf-8")
            (project / "qepas_protocol.json").write_text(
                '{"version": 7}', encoding="utf-8"
            )

            self.assertTrue(_tuner_exists(tuner_dir, "experiment"))

    def test_tuner_resume_rejects_changed_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tuner_dir = Path(tmp)
            project = tuner_dir / "experiment"
            (project / "trial_00").mkdir(parents=True)
            (project / "oracle.json").write_text("{}", encoding="utf-8")
            (project / "qepas_protocol.json").write_text(
                '{"version": 7, "preprocessing_id": "pipeline-a"}',
                encoding="utf-8",
            )

            self.assertFalse(
                _tuner_exists(
                    tuner_dir,
                    "experiment",
                    expected_protocol={
                        "version": 7,
                        "preprocessing_id": "pipeline-b",
                    },
                )
            )

    def test_tuner_resume_rejects_legacy_validation_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tuner_dir = Path(tmp)
            project = tuner_dir / "experiment"
            (project / "trial_00").mkdir(parents=True)
            (project / "oracle.json").write_text("{}", encoding="utf-8")

            self.assertFalse(_tuner_exists(tuner_dir, "experiment"))

    def test_tuner_resume_requires_oracle_and_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tuner_dir = Path(tmp)
            project = tuner_dir / "experiment"
            project.mkdir()
            (project / "oracle.json").write_text("{}", encoding="utf-8")
            (project / "qepas_protocol.json").write_text(
                '{"version": 7}', encoding="utf-8"
            )

            self.assertFalse(_tuner_exists(tuner_dir, "experiment"))


    def test_saved_tuner_artifact_records_selection_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "tuner.json"
            normalizers = type(
                "Normalizers",
                (),
                {"to_dict": lambda self: {"signal": {}}},
            )()
            protocol = {
                "name": "single_group_holdout",
                "independent_outer_estimate": False,
            }
            save_tuner_results(
                {
                    "best_params": {"architecture": "simple_cnn"},
                    "normalizers": normalizers,
                    "validation_group": "group-7",
                    "selection_protocol": protocol,
                    "preprocessing_id": "raw-v1",
                    "trials": [],
                },
                target,
            )

            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["selection_protocol"], protocol)


if __name__ == "__main__":
    unittest.main()
