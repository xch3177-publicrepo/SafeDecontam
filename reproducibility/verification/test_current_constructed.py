"""Synthetic boundary checks; these rows are not empirical paper evidence."""
import math
import unittest

import numpy as np

from verify_current_constructed import evaluate, threshold_from_scores


class ConstructedVerifierTests(unittest.TestCase):
    def test_threshold_prefers_same_recall_with_less_damage(self):
        rows = [{"family": "test", "group_id": str(i), "decision_label": "retain"}
                for i in range(106)]
        scores = [0.1] * 105 + [0.8]
        rows += [{"family": "test", "group_id": str(i), "decision_label": "remove"}
                 for i in (0, 1)]
        scores += [0.9, 1.0]
        # 0.8 has the same recall but damages a clean group; 0.9 is preferred.
        self.assertEqual(threshold_from_scores(rows, np.array(scores), .05), .9)

    def test_saturated_probabilities_force_abstention(self):
        rows = [{"family": "test", "group_id": str(i), "decision_label": "retain"}
                for i in range(106)]
        rows.append({"family": "test", "group_id": "0", "decision_label": "remove"})
        threshold = threshold_from_scores(rows, np.ones(107), .05)
        self.assertEqual(threshold, math.inf)
        self.assertEqual(evaluate(rows, np.ones(107), threshold)["CR"], 0.0)

    def test_group_damage_counts_group_once(self):
        rows = [
            {"family": "test", "group_id": "A", "decision_label": "retain"},
            {"family": "test", "group_id": "A", "decision_label": "retain"},
            {"family": "test", "group_id": "B", "decision_label": "retain"},
            {"family": "test", "group_id": "A", "decision_label": "remove"},
        ]
        values = evaluate(rows, np.array([.9, .8, .1, .9]), .7)
        self.assertEqual(values["removed_clean_pairs"], 2)
        self.assertEqual(values["damaged_groups"], 1)
        self.assertEqual(values["total_groups"], 2)
        self.assertEqual(values["GCDR"], 50.0)
        self.assertAlmostEqual(values["CDR"], 200 / 3)


if __name__ == "__main__":
    unittest.main()
