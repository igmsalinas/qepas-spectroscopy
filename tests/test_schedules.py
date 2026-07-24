from __future__ import annotations

import unittest

from qepas_spectroscopy.models.deep_learning.schedules import (
    make_lr_schedule,
)


class LearningRateScheduleTests(unittest.TestCase):
    def test_warmup_epochs_are_converted_to_optimizer_steps(self) -> None:
        schedule = make_lr_schedule(
            schedule_name="constant",
            initial_lr=0.01,
            warmup_epochs=2,
            total_epochs=10,
            steps_per_epoch=5,
        )

        self.assertAlmostEqual(float(schedule(0)), 0.0)
        self.assertAlmostEqual(float(schedule(5)), 0.005)
        self.assertAlmostEqual(float(schedule(10)), 0.01)

    def test_invalid_epoch_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_epochs"):
            make_lr_schedule("constant", 0.01, 0, 0)
        with self.assertRaisesRegex(ValueError, "steps_per_epoch"):
            make_lr_schedule("constant", 0.01, 0, 10, steps_per_epoch=0)


if __name__ == "__main__":
    unittest.main()
