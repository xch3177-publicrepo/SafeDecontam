"""Deterministic mathematical boundary checks; no experimental data are used."""

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scipy.stats import binom

from calibrate import calibrate, upper_bound


def synthetic_spec(n, families=("SYNTHETIC_A",)):
    return {
        "data_status": "SYNTHETIC test fixture; not paper data",
        "requested_families": list(families),
        "shared_score_scale": "SYNTHETIC_float64_v1",
        "alpha": 0.05,
        "epsilon": 0.05,
        "calibration_maxima": {h: [i / max(n, 1) for i in range(n)] for h in families},
    }


class CalibrationTests(unittest.TestCase):
    def test_one_family_minimum_sample_boundary(self):
        self.assertEqual(calibrate(synthetic_spec(58))["status"], "abstain")
        enough = calibrate(synthetic_spec(59))
        self.assertEqual(enough["status"], "calibrated")
        self.assertEqual(enough["families"][0]["allowed_exceedances_k"], 0)
        # Closed-form zero-failure inversion independently checks Beta quantile.
        self.assertAlmostEqual(upper_bound(59, 0, 0.05), 1 - 0.05 ** (1 / 59), places=13)

    def test_three_family_bonferroni_boundary(self):
        names = ("SYNTHETIC_A", "SYNTHETIC_B", "SYNTHETIC_C")
        self.assertEqual(calibrate(synthetic_spec(79, names))["status"], "abstain")
        result = calibrate(synthetic_spec(80, names))
        self.assertEqual(result["status"], "calibrated")
        self.assertEqual(result["H"], 3)
        self.assertTrue(all(r["risk_upper_bound"] <= 0.05 for r in result["families"]))

    def test_106_groups_allow_one_but_not_two_exceedances(self):
        result = calibrate(synthetic_spec(106))["families"][0]
        self.assertEqual(result["allowed_exceedances_k"], 1)
        self.assertEqual(result["order_statistic_rank_1_based"], 105)
        self.assertLessEqual(result["risk_upper_bound"], 0.05)
        self.assertGreater(upper_bound(106, 2, 0.05), 0.05)
        # A separate binomial-tail identity checks exact inversion, including
        # the k=1 off-by-one convention used by the order-statistic cutoff.
        self.assertAlmostEqual(binom.cdf(1, 106, result["risk_upper_bound"]), 0.05, places=12)

    def test_ties_remain_retained_and_do_not_spend_more_than_k(self):
        spec = synthetic_spec(106)
        spec["calibration_maxima"]["SYNTHETIC_A"] = [0.5] * 106
        result = calibrate(spec)
        self.assertEqual(result["shared_cutoff"], math.nextafter(0.5, math.inf))
        self.assertFalse(0.5 >= result["shared_cutoff"])
        self.assertEqual(result["families"][0]["observed_exceedances_at_family_cutoff"], 0)
        self.assertEqual(result["families"][0]["allowed_exceedances_k"], 1)

    def test_shared_cutoff_is_maximum_and_never_increases_family_deletion(self):
        spec = synthetic_spec(80, ("SYNTHETIC_A", "SYNTHETIC_B", "SYNTHETIC_C"))
        spec["calibration_maxima"]["SYNTHETIC_A"] = [0.2] * 80
        spec["calibration_maxima"]["SYNTHETIC_B"] = [0.4] * 80
        spec["calibration_maxima"]["SYNTHETIC_C"] = [0.6] * 80
        result = calibrate(spec)
        self.assertEqual(result["shared_cutoff"], math.nextafter(0.6, math.inf))
        for family in result["families"]:
            for score in (0.1, 0.3, 0.5, 0.7):
                self.assertLessEqual(score >= result["shared_cutoff"], score >= family["family_cutoff"])

    def test_missing_or_empty_requested_family_is_not_dropped(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                spec = synthetic_spec(80, ("SYNTHETIC_A", "SYNTHETIC_B", "SYNTHETIC_C"))
                if missing:
                    del spec["calibration_maxima"]["SYNTHETIC_C"]
                else:
                    spec["calibration_maxima"]["SYNTHETIC_C"] = []
                result = calibrate(spec)
                self.assertEqual(result["H"], 3)
                self.assertEqual(result["status"], "abstain")
                self.assertIsNone(result["shared_cutoff"])
                self.assertEqual(result["families"][0]["family_alpha"], 0.05 / 3)

    def test_nonfinite_scores_and_cutoff_overflow_abstain(self):
        for value in (math.nan, math.inf, -math.inf, sys.float_info.max):
            with self.subTest(value=value):
                spec = synthetic_spec(59)
                spec["calibration_maxima"]["SYNTHETIC_A"][0] = value
                result = calibrate(spec)
                self.assertEqual(result["status"], "abstain")
                self.assertIsNone(result["shared_cutoff"])
                json.dumps(result, allow_nan=False)

    def test_bad_configuration_is_rejected(self):
        for edit in (
            {"epsilon": 0}, {"alpha": 1}, {"epsilon": math.nan},
            {"alpha": 1e-100}, {"requested_families": []},
            {"requested_families": ["SYNTHETIC_A", "SYNTHETIC_A"]},
            {"shared_score_scale": ""}, {"epsilon": True},
        ):
            with self.subTest(edit=edit):
                spec = copy.deepcopy(synthetic_spec(59))
                spec.update(edit)
                with self.assertRaises(ValueError):
                    calibrate(spec)
        for k in (-1, 59):
            with self.assertRaises(ValueError):
                upper_bound(59, k, 0.05)

    def test_cli_json_has_input_hash_and_roundtrips_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.json"
            path.write_text(json.dumps(synthetic_spec(59)), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("calibrate.py")), str(path)],
                capture_output=True, text=True, check=True,
            )
        result = json.loads(proc.stdout)
        self.assertEqual(len(result["input_sha256"]), 64)
        self.assertEqual(result["shared_cutoff"], math.nextafter(58 / 59, math.inf))


if __name__ == "__main__":
    unittest.main()
