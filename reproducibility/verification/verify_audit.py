#!/usr/bin/env python3
"""Independently check released audit outputs without fitting any model.

This program does not import experiment code. It distinguishes row-level
replay, aggregate arithmetic, and unavailable evidence in its JSON report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


def binomial_cdf(k: int, n: int, probability: float) -> float:
    """Direct finite sum; sufficient for the small sample sizes in this audit."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return math.fsum(
        math.comb(n, j) * probability**j * (1.0 - probability)**(n - j)
        for j in range(k + 1)
    )


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper endpoint by binomial-CDF inversion."""
    if not 0 <= k <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError("Require n > 0, 0 <= k <= n, 0 < alpha < 1")
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cp_interval(k: int, n: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if k == 0 else 1.0 - cp_upper(n - k, n, alpha / 2)
    return [lower, cp_upper(k, n, alpha / 2)]


def allowed_exceedances(n: int, epsilon: float, alpha: float) -> int | None:
    feasible = [k for k in range(n) if cp_upper(k, n, alpha) <= epsilon]
    return max(feasible) if feasible else None


def minimum_zero_exceedance_groups(epsilon: float, alpha: float, families: int) -> int:
    # Bonferroni allocation alpha/families. Verify the boundary against the
    # exact zero-event expression, avoiding floating-point ceil edge cases.
    n = math.ceil(math.log(alpha / families) / math.log1p(-epsilon))
    while 1 - (alpha / families)**(1 / n) > epsilon:
        n += 1
    while n > 1 and 1 - (alpha / families)**(1 / (n - 1)) <= epsilon:
        n -= 1
    return n


def replay_metrics(labels: np.ndarray, removed: np.ndarray) -> dict:
    positive, clean = labels == 1, labels == 0
    tp = int(np.count_nonzero(removed & positive))
    fp = int(np.count_nonzero(removed & clean))
    pos_n, clean_n = int(positive.sum()), int(clean.sum())
    return {
        "positive_n": pos_n,
        "retain_n": clean_n,
        "true_positive_n": tp,
        "false_positive_n": fp,
        "false_negative_n": pos_n - tp,
        "true_negative_n": clean_n - fp,
        "removed_n": tp + fp,
        "cr": tp / pos_n,
        "cdr": fp / clean_n,
        "gcdr": fp / clean_n,
    }


def pairwise_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    # Probability a positive outranks a negative, with half credit for ties.
    positive = scores[labels == 1]
    clean = scores[labels == 0]
    pair_comparisons = positive[:, None] - clean[None, :]
    return float((np.count_nonzero(pair_comparisons > 0)
                  + 0.5 * np.count_nonzero(pair_comparisons == 0))
                 / pair_comparisons.size)


def bootstrap_intervals(labels: np.ndarray, removed: np.ndarray,
                        source_ids: list[str], reps: int, seed: int) -> dict:
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Expected exactly one row per source group")
    rng = np.random.default_rng(seed)
    samples = []
    # Preserve input group order: the archived experiment sampled that order.
    # Each group has one row, so integer index sampling equals source-ID sampling.
    for _ in range(reps):
        idx = rng.choice(len(source_ids), len(source_ids), replace=True)
        values = replay_metrics(labels[idx], removed[idx])
        samples.append([values["cr"], values["cdr"], values["gcdr"]])
    endpoints = np.quantile(np.asarray(samples), [0.025, 0.975], axis=0,
                            method="linear")
    return {key: endpoints[:, i].tolist()
            for i, key in enumerate(("cr", "cdr", "gcdr"))}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    audit_dir = args.audit_dir.resolve()
    result_path = audit_dir / "results.json"
    predictions_path = audit_dir / "locked_test_predictions.jsonl"
    expected = json.loads(result_path.read_text())
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()
            if line.strip()]
    human = json.loads(args.human_aggregates.read_text())
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PENDING",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "inputs": {p.name: {"sha256": file_hash(p)}
                   for p in (result_path, predictions_path, args.human_aggregates)},
        "checks": [],
        "unavailable_evidence": [],
        "recomputed": {},
    }

    def check(name: str, actual, target, evidence: str, tolerance=1e-12):
        if isinstance(actual, (float, int)) and isinstance(target, (float, int)):
            passed = math.isclose(actual, target, rel_tol=0, abs_tol=tolerance)
        elif isinstance(actual, list) and isinstance(target, list):
            passed = len(actual) == len(target) and all(
                math.isclose(a, b, rel_tol=0, abs_tol=tolerance)
                for a, b in zip(actual, target)
            )
        else:
            passed = actual == target
        report["checks"].append({"name": name, "status": "PASS" if passed else "FAIL",
                                 "recomputed": actual, "recorded": target,
                                 "evidence": evidence})

    source_ids = [row["source_id"] for row in rows]
    labels = np.asarray([row["silver_label"] for row in rows], dtype=int)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    threshold = float(expected["calibration"]["threshold"])
    removed = scores >= threshold
    recorded_removed = np.asarray([row["removed"] for row in rows], dtype=bool)
    row_evidence = "recomputed_from_locked_test_prediction_rows"
    aggregate_evidence = "arithmetic_from_reported_aggregate_counts_not_row_replay"
    check("one_row_per_source_group", len(set(source_ids)), len(rows), row_evidence)
    check("binary_silver_labels", bool(np.all(np.isin(labels, [0, 1]))), True, row_evidence)
    check("finite_scores", bool(np.all(np.isfinite(scores))), True, row_evidence)
    check("all_prediction_thresholds_equal_reported_threshold",
          all(row["threshold"] == threshold for row in rows), True, row_evidence)
    check("removal_flags_follow_score_ge_threshold",
          bool(np.array_equal(removed, recorded_removed)), True, row_evidence)
    check("category_and_binary_label_consistency",
          all(int(row["silver_category"] != "clean") == row["silver_label"]
              for row in rows), True, row_evidence)
    check("locked_test_row_count", len(rows), expected["split_counts"]["test"], row_evidence)
    metrics = replay_metrics(labels, removed)
    for key in ("positive_n", "retain_n", "false_negative_n", "false_positive_n",
                "cr", "cdr", "gcdr"):
        check(f"locked_test.{key}", metrics[key], expected["locked_test"][key], row_evidence)
    auc = pairwise_auc(labels, scores)
    check("locked_test.auroc_diagnostic", auc,
          expected["locked_test"]["auroc_diagnostic"], row_evidence)
    bootstrap = bootstrap_intervals(labels, removed, source_ids,
                                    args.bootstrap_reps, args.bootstrap_seed)
    for key, endpoints in bootstrap.items():
        check(f"locked_test.bootstrap_95_ci.{key}", endpoints,
              expected["locked_test"]["cluster_bootstrap_95_ci"][key], row_evidence)
    categories = {}
    for category, values in expected["locked_test"]["recall_by_category"].items():
        idx = np.asarray([row["silver_category"] == category for row in rows])
        categories[category] = {"n": int(idx.sum()), "removed": int(removed[idx].sum()),
                                "recall": float(removed[idx].mean())}
        for key, value in categories[category].items():
            check(f"locked_test.category.{category}.{key}", value, values[key], row_evidence)
    report["recomputed"]["locked_test"] = {
        **metrics, "auroc": auc, "recall_by_category": categories,
        "bootstrap_95_ci": bootstrap,
        "bootstrap": {"replicates": args.bootstrap_reps, "seed": args.bootstrap_seed,
                      "unit": "source group (one item per row)",
                      "quantile_method": "linear", "resampling": "unstratified"},
    }

    validation = {}
    for name, values in expected["validation_model_selection"].items():
        numerator = values["positive_n"] - values["false_negative_n"]
        validation[name] = {"removed_contaminated_n": numerator,
                            "positive_n": values["positive_n"],
                            "recall": numerator / values["positive_n"]}
        check(f"validation.{name}.positive_n", values["positive_n"], 63, aggregate_evidence)
        check(f"validation.{name}.cr", numerator / values["positive_n"], values["cr"],
              aggregate_evidence)
        check(f"validation.{name}.cdr", values["false_positive_n"] / values["retain_n"],
              values["cdr"], aggregate_evidence)
        check(f"validation.{name}.gcdr", values["false_positive_n"] / values["retain_n"],
              values["gcdr"], aggregate_evidence)
    selected = max(validation, key=lambda name: (
        validation[name]["recall"],
        -expected["validation_model_selection"][name]["cdr"]))
    check("validation.selected_scorer", selected, expected["selected_scorer"], aggregate_evidence)
    report["recomputed"]["validation_aggregate_arithmetic"] = validation

    calibration = expected["calibration"]
    n, epsilon, alpha = calibration["clean_groups"], calibration["epsilon"], calibration["alpha"]
    k = allowed_exceedances(n, epsilon, alpha)
    check("calibration.allowed_exceedances", k, calibration["allowed_exceedances"],
          aggregate_evidence)
    check("calibration.upper_bound", cp_upper(k, n, alpha), calibration["upper_bound"],
          aggregate_evidence)
    check("calibration.reported_clean_groups", n, 106, aggregate_evidence)
    min_groups = {str(h): minimum_zero_exceedance_groups(epsilon, alpha, h) for h in (1, 3)}
    check("minimum_groups.one_family", min_groups["1"], 59, "exact_binomial_arithmetic")
    check("minimum_groups.three_families", min_groups["3"], 80, "exact_binomial_arithmetic")
    report["recomputed"]["calibration_aggregate_arithmetic"] = {
        "reported_clean_groups": n, "epsilon": epsilon, "alpha": alpha,
        "bounds_by_exceedance_count": {str(j): cp_upper(j, n, alpha) for j in (0, 1, 2)},
        "largest_allowed_exceedance_count": k,
        "minimum_zero_exceedance_groups_by_family_count": min_groups,
        "threshold": {"recorded": threshold,
                      "derivation_status": "NOT_REPLAYED_NO_CALIBRATION_SCORES"},
    }
    for budget, values in expected.get("budget_sensitivity", {}).items():
        budget_k = allowed_exceedances(n, float(budget), alpha)
        check(f"budget.{budget}.abstained", budget_k is None, values["abstained"], aggregate_evidence)
        if budget_k is None:
            continue
        check(f"budget.{budget}.allowed_exceedances", budget_k, values["allowed_exceedances"],
              aggregate_evidence)
        check(f"budget.{budget}.upper_bound", cp_upper(budget_k, n, alpha),
              values["upper_bound"], aggregate_evidence)
        budget_metrics = replay_metrics(labels, scores >= values["threshold"])
        for key in ("cr", "cdr", "gcdr", "false_negative_n", "false_positive_n"):
            check(f"budget.{budget}.{key}", budget_metrics[key], values[key], row_evidence)
    check("split_counts_sum", sum(expected["split_counts"].values()), expected["item_count"],
          aggregate_evidence)
    check("category_counts_sum", sum(expected["category_counts"].values()), expected["item_count"],
          aggregate_evidence)

    author_evidence = "author_confirmed_aggregate_inputs_not_independent_label_replay"
    h, full = human["locked_test_human_audit"], human["full_page_subsample"]
    human_stats = {
        "evidence": author_evidence,
        "contaminated_recall": h["contaminated_removed"] / h["contaminated_total"],
        "contaminated_recall_two_sided_95_cp": cp_interval(h["contaminated_removed"], h["contaminated_total"]),
        "clean_deletion_rate": h["clean_removed"] / h["clean_total"],
        "clean_deletion_one_sided_95_upper": cp_upper(h["clean_removed"], h["clean_total"]),
        "removed_item_precision": h["contaminated_removed"] / h["removed_total"],
        "removed_item_precision_two_sided_95_cp": cp_interval(h["contaminated_removed"], h["removed_total"]),
        "full_page_recall": full["contaminated_removed"] / full["contaminated_total"],
        "full_page_recall_two_sided_95_cp": cp_interval(full["contaminated_removed"], full["contaminated_total"]),
        "full_page_clean_deletion_one_sided_95_upper": cp_upper(full["clean_removed"], full["clean_total"]),
    }
    check("human.total_items", h["contaminated_total"] + h["clean_total"] + h["uncertain_total"],
          len(rows), author_evidence)
    check("human.removed_total_consistency", h["contaminated_removed"] + h["clean_removed"]
          + h["uncertain_removed"], h["removed_total"], author_evidence)
    check("human.removed_total_matches_frozen_rule", h["removed_total"], metrics["removed_n"],
          "aggregate_consistency_with_row_replayed_removal_total_only")
    report["recomputed"]["human_aggregate_arithmetic"] = human_stats
    report["unavailable_evidence"] = [
        {"claim": "validation AUROCs and score-derived thresholds",
         "status": "NOT_REPLAYED", "reason": "Validation score/label rows are not in these inputs."},
        {"claim": "calibration clean-group count and order-statistic threshold",
         "status": "NOT_REPLAYED", "reason": "No calibration score rows; bounds are checked conditional on the reported n=106."},
        {"claim": "human-label per-item accuracy, human AUROC, agreement, and kappa",
         "status": "NOT_REPLAYED", "reason": "No item-level human labels or annotator records supplied; aggregate arithmetic does not establish label validity."},
        {"claim": "historical constructed pairs, transfer, web pilot, and other benchmark runs",
         "status": "OUT_OF_SCOPE", "reason": "This verifier is restricted to the supplied MMLU audit files and stated aggregate inputs."},
        {"claim": "independent clean draws and calibration population representativeness",
         "status": "NOT_STATISTICALLY_VERIFIABLE_FROM_OUTPUTS",
         "reason": "Sampling assumptions require study-design evidence; a numeric replay cannot establish them."},
    ]
    failed = [item["name"] for item in report["checks"] if item["status"] == "FAIL"]
    report["overall_status"] = "FAIL" if failed else "PASS_WITH_LIMITATIONS"
    report["failed_checks"] = failed
    report["checks_passed"] = len(report["checks"]) - len(failed)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True,
                        help="Directory containing results.json and locked_test_predictions.jsonl")
    parser.add_argument("--human-aggregates", type=Path,
                        default=Path(__file__).with_name("author_confirmed_aggregates.json"))
    parser.add_argument("--output", type=Path, default=Path("audit_verification_report.json"))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        report = {"schema_version": 1, "overall_status": "ERROR",
                  "error_type": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['overall_status']}: {args.output}")
    if report.get("failed_checks"):
        print("Failed checks: " + ", ".join(report["failed_checks"]))
    return 0 if report["overall_status"] == "PASS_WITH_LIMITATIONS" else 1


if __name__ == "__main__":
    sys.exit(main())
