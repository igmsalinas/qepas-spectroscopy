from __future__ import annotations

import unittest

import numpy as np

from qepas_spectroscopy.validation import (
    nested_group_folds,
    validation_group_split,
)


class GroupSplittingTests(unittest.TestCase):
    def test_nested_folds_keep_fit_validation_and_test_groups_disjoint(self) -> None:
        groups = np.repeat(["a", "b", "c", "d"], 2)

        folds = list(nested_group_folds(groups))

        self.assertEqual(len(folds), 4)
        for fold in folds:
            fit_groups = set(groups[fold.fit_indices])
            validation_groups = set(groups[fold.validation_indices])
            test_groups = set(groups[fold.test_indices])
            self.assertTrue(fit_groups.isdisjoint(validation_groups))
            self.assertTrue(fit_groups.isdisjoint(test_groups))
            self.assertTrue(validation_groups.isdisjoint(test_groups))
            self.assertEqual(test_groups, {fold.test_group})
            self.assertEqual(validation_groups, {fold.validation_group})

    def test_validation_split_rejects_unknown_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            validation_group_split(np.array(["a", "b"]), validation_group="c")

    def test_nested_folds_require_three_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            list(nested_group_folds(np.array(["a", "b"])))


if __name__ == "__main__":
    unittest.main()
