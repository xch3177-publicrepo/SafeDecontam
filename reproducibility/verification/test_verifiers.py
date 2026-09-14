"""Synthetic self-checks; these fixtures are not research evidence."""

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from verify_audit import cp_interval, cp_upper, pairwise_auc

ROOT = Path(__file__).resolve().parent


class VerifierTests(unittest.TestCase):
    def test_binomial_zero_event_boundary_and_interval_symmetry(self):
        for n in (30, 59, 71, 80, 106):
            self.assertAlmostEqual(cp_upper(0, n), 1 - 0.05**(1 / n), places=14)
        lo, hi = cp_interval(67, 82)
        reflected_lo, reflected_hi = cp_interval(15, 82)
        self.assertAlmostEqual(lo, 1 - reflected_hi, places=14)
        self.assertAlmostEqual(hi, 1 - reflected_lo, places=14)

    def test_auc_ties_and_reversed_ranking(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(pairwise_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(pairwise_auc(labels, np.array([0.8, 0.9, 0.1, 0.2])), 0.0)
        self.assertEqual(pairwise_auc(labels, np.ones(4)), 0.5)

    def test_fresh_human_labels_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            predictions = directory / "synthetic_predictions.jsonl"
            final_path = directory / "synthetic_final.csv"
            a_path, b_path = directory / "synthetic_a.csv", directory / "synthetic_b.csv"
            report_path = directory / "report.json"
            labels = ["input", "clean", "uncertain", "input_and_label", "clean"]
            scores = [0.9, 0.7, 0.1, 0.8, 0.2]
            predictions.write_text("".join(json.dumps({
                "source_id": f"SYNTHETIC-{i}", "score": score, "threshold": 0.5,
                "removed": score >= 0.5, "silver_label": int(labels[i] not in ("clean", "uncertain")),
            }) + "\n" for i, score in enumerate(scores)))
            def write_final(duplicate=False):
                with final_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["item_id", "human_label", "full_page_accessible"])
                    for i, label in enumerate(labels):
                        item_id = f"SYNTHETIC-{0 if duplicate and i == 4 else i}"
                        writer.writerow([item_id, label, ["yes", "yes", "unknown", "no", "no"][i]])
            write_final()
            for path, source in ((a_path, labels), (b_path, ["clean"] + labels[1:])):
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["item_id", "item_label"])
                    writer.writerows((f"SYNTHETIC-{i}", label) for i, label in enumerate(source))
            command = [sys.executable, str(ROOT / "verify_human_labels.py"),
                       "--predictions", str(predictions), "--final-labels", str(final_path),
                       "--annotator-a", str(a_path), "--annotator-b", str(b_path),
                       "--output", str(report_path)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["all_items"]["true_positive_n"], 2)
            self.assertEqual(report["all_items"]["false_positive_n"], 1)
            self.assertEqual(report["all_items"]["uncertain_excluded_n"], 1)
            self.assertAlmostEqual(report["all_items"]["precision_excluding_uncertain"], 2/3)
            self.assertEqual(report["annotator_agreement"]["four_class"]["agreement_n"], 4)
            self.assertEqual(report["annotator_agreement"]["binary_both_certain"]["kappa"], 0.5)
            self.assertEqual(report["access_counts"], {"no": 2, "unknown": 1, "yes": 2})
            write_final(duplicate=True)
            failed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(json.loads(report_path.read_text())["overall_status"], "ERROR")
            write_final()
            final_path.write_text(final_path.read_text().replace("SYNTHETIC-4", "SYNTHETIC-MISSING"))
            failed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn("ID sets differ", json.loads(report_path.read_text())["error"])


if __name__ == "__main__":
    unittest.main()
