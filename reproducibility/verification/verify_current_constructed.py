#!/usr/bin/env python3
"""Replay the complete constructed run from source rows, frozen scores, and models.

No experiment function is imported. This verifies saved outputs and fitted
model inference, not the training optimizer, feature extraction, or labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from verify_audit import binomial_cdf, cp_upper, file_hash, pairwise_auc

FAMILIES = ["arithmetic", "mcq", "math"]
MAIN_FAMILIES = ["arithmetic", "mcq"]
VARIANTS = ["ngram4", "shingle3", "bm25n", "fusion", "no_constraints", "core", "mlp", "mlp_margin"]
SEEDS = [42579] + [int(hashlib.sha256(f"SafeDecontam-real-v2-{i}".encode()).hexdigest()[:8], 16) % 100000
                   for i in range(1, 10)]


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


@lru_cache(maxsize=None)
def bound(k, n, alpha):
    return cp_upper(k, n, alpha) if n else 1.0


@lru_cache(maxsize=None)
def maximum_feasible_count(n, epsilon, alpha):
    # U_CP(k;n,alpha) <= epsilon iff BinomialCDF(k;n,epsilon) <= alpha.
    # The CDF is monotone in k, allowing an independent integer search.
    lo, hi = -1, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binomial_cdf(mid, n, epsilon) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def subset(rows, scores, families):
    idx = [i for i, row in enumerate(rows) if row["family"] in families]
    return [rows[i] for i in idx], scores[idx]


def group_maxima(rows, scores):
    maxima = {}
    for row, score in zip(rows, scores):
        if row["decision_label"] == "retain":
            key = (row["family"], row["group_id"])
            maxima[key] = max(maxima.get(key, -math.inf), float(score))
    return maxima


def threshold_from_scores(rows, scores, epsilon):
    families = sorted({row["family"] for row in rows})
    maxima = group_maxima(rows, scores)
    candidates = np.append(np.unique(scores), math.inf)
    feasible = np.ones(len(candidates), dtype=bool)
    damage = np.zeros(len(candidates), dtype=int)
    for family in families:
        values = np.sort([value for (fam, _), value in maxima.items() if fam == family])
        allowed = maximum_feasible_count(len(values), epsilon, 0.05 / len(families))
        deleted = len(values) - np.searchsorted(values, candidates, side="left")
        feasible &= deleted <= allowed
        damage += deleted
    positive = np.sort([score for row, score in zip(rows, scores) if row["decision_label"] == "remove"])
    recalled = len(positive) - np.searchsorted(positive, candidates, side="left")
    eligible = np.flatnonzero(feasible)
    if not len(eligible):
        return math.inf
    # Maximize calibration recall, then minimize group damage, then maximize
    # the threshold. Inclusive score >= threshold matches the archived rule.
    recalled_max = recalled[eligible].max()
    eligible = eligible[recalled[eligible] == recalled_max]
    damage_min = damage[eligible].min()
    eligible = eligible[damage[eligible] == damage_min]
    return float(candidates[eligible[-1]])


def evaluate(rows, scores, threshold):
    positive = np.asarray([row["decision_label"] == "remove" for row in rows])
    removed = scores >= threshold
    positives, clean = int(positive.sum()), int((~positive).sum())
    tp, fp = int((removed & positive).sum()), int((removed & ~positive).sum())
    maxima = group_maxima(rows, scores)
    families = sorted({fam for fam, _ in maxima})
    damaged = sum(value >= threshold for value in maxima.values())
    upper = {}
    for family in families:
        values = [value for (fam, _), value in maxima.items() if fam == family]
        upper[family] = 100 * bound(sum(value >= threshold for value in values), len(values), .05 / len(families))
    return {
        "CR": 100 * tp / positives if positives else None,
        "CDR": 100 * fp / clean if clean else None,
        "GCDR": 100 * damaged / len(maxima) if maxima else None,
        "CRR": 100 * (1 - fp / clean) if clean else None,
        "FRPC": fp / tp if tp else None,
        "damaged_groups": damaged, "total_groups": len(maxima),
        "removed_clean_pairs": fp, "total_clean_pairs": clean,
        "removed_contam": tp, "total_contam": positives,
        "U95_worst": max(upper.values()) if upper else None,
        "U95_by_family": upper,
    }


def error_breakdown(rows, scores, threshold):
    missed, false_removed = Counter(), Counter()
    for row, score in zip(rows, scores):
        key = "|".join(row[column] for column in ("family", "relation", "candidate_source"))
        if row["decision_label"] == "remove" and score < threshold:
            missed[key] += 1
        if row["decision_label"] == "retain" and score >= threshold:
            false_removed[key] += 1
    return {"n_miss": sum(missed.values()), "n_fp": sum(false_removed.values()),
            "missed_by": dict(sorted(missed.items())), "false_removed_by": dict(sorted(false_removed.items()))}


def bootstrap_recall(rows, scores, threshold):
    groups = defaultdict(list)
    for row, score in zip(rows, scores):
        if row["decision_label"] == "remove":
            groups[row["group_id"]].append(score >= threshold)
    keys = sorted(groups)
    positives = np.asarray([len(groups[key]) for key in keys])
    removed = np.asarray([sum(groups[key]) for key in keys])
    random = np.random.RandomState(0)
    values = []
    for _ in range(2000):
        indices = random.choice(len(keys), len(keys), replace=True)
        values.append(removed[indices].sum() / positives[indices].sum())
    return (100 * np.quantile(values, [.025, .975], method="linear")).tolist()


def sigmoid(values):
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def run(args):
    directory = args.directory.resolve()
    manifest = read_json(directory / "run_manifest.json")
    recorded = read_json(directory / "results.json")
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PENDING", "checks": [], "failed_checks": [],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "input_hashes": {}, "recomputed": {"primary": {}, "sensitivity": {"mlp": [], "mlp_margin": []},
                                           "sensitivity_math_margin": []},
        "scope": "Independent row/score/threshold/model-inference replay; no training or feature-extraction reproduction.",
    }

    def remember(path):
        report["input_hashes"][str(path.relative_to(directory))] = file_hash(path)

    def check(name, actual, expected, tolerance=1e-10):
        actual = finite_json(actual)
        if isinstance(actual, dict) and isinstance(expected, dict):
            check(name + ".keys", sorted(actual), sorted(expected))
            for key in sorted(actual.keys() & expected.keys()):
                check(name + "." + str(key), actual[key], expected[key], tolerance)
            return
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            passed = math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)
        elif isinstance(actual, list) and isinstance(expected, list) and all(isinstance(v, (int, float)) for v in actual + expected):
            passed = len(actual) == len(expected) and all(math.isclose(a, b, rel_tol=0, abs_tol=tolerance) for a, b in zip(actual, expected))
        else:
            passed = actual == expected
        result = {"name": name, "status": "PASS" if passed else "FAIL"}
        if not passed:
            result.update({"recomputed": actual, "recorded": expected})
            report["failed_checks"].append(name)
        report["checks"].append(result)

    def arrays_match(name, actual, expected, tolerance=1e-9):
        shape_ok = actual.shape == expected.shape
        check(name + ".shape", list(actual.shape), list(expected.shape))
        if shape_ok:
            difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
            check(name + ".max_abs_error_within_tolerance", difference <= tolerance, True)

    for path in (directory / "run_manifest.json", directory / "results.json"):
        remember(path)
    check("run_complete", manifest.get("status"), "complete")
    check("manifest.seed_list", manifest["seed_list"], SEEDS)
    check("result.seed_list", recorded["seed_list"], SEEDS)
    for name, digest in manifest["source_sha256"].items():
        path = directory / "source" / name
        check("source_hash." + name, file_hash(path), digest)
        remember(path)
    source_rows = {}
    expected_sources = set()
    if args.input_dir:
        for name, metadata in manifest["inputs"].items():
            path = args.input_dir / name
            check("source_input_hash." + name, file_hash(path), metadata["sha256"])
            report["input_hashes"]["canonical_inputs/" + name] = file_hash(path)
            source_rows[name] = read_csv(path)
            for i, row in enumerate(source_rows[name]):
                if row.get("benchmark_text") and row.get("candidate_text", "").strip():
                    expected_sources.add((name, i))
        check("source_usable_pairs", len(expected_sources), manifest["usable_pairs"])

    for seed in SEEDS:
        seed_dir = directory / "seeds" / str(seed)
        frames = {part: read_csv(seed_dir / f"frame_{part}.csv") for part in ("train", "cal", "test")}
        seed_result = read_json(seed_dir / "results.json")
        feature_metadata = read_json(seed_dir / "features.json")
        score_file = seed_dir / "scores.npz"
        scores = np.load(score_file, allow_pickle=False)
        features = np.load(seed_dir / "features.npz", allow_pickle=False)
        for path in [seed_dir / "results.json", seed_dir / "features.json", seed_dir / "features.npz", score_file]:
            remember(path)
        check(f"seed.{seed}.identity", seed_result["seed"], seed)
        observed_sources, ownership = [], defaultdict(set)
        source_mismatch_count = 0
        observed_groups = {part: defaultdict(set) for part in frames}
        for part, rows in frames.items():
            prefix = f"seed.{seed}.{part}"
            frame_path = seed_dir / f"frame_{part}.csv"
            remember(frame_path)
            check(prefix + ".frame_hash", file_hash(frame_path), feature_metadata["frames"][part]["sha256"])
            check(prefix + ".row_count", len(rows), feature_metadata["frames"][part]["rows"])
            check(prefix + ".feature_shape", list(features[part].shape), [len(rows), len(feature_metadata["columns"])])
            check(prefix + ".finite_features", bool(np.isfinite(features[part]).all()), True)
            check(prefix + ".declared_split", all(row["part"] == part for row in rows), True)
            check(prefix + ".decision_labels", sorted({row["decision_label"] for row in rows}), ["remove", "retain"])
            for row in rows:
                key = (row["family"], row["group_id"])
                ownership[key].add(part)
                observed_groups[part][row["family"]].add(row["group_id"])
                source = (row["source_csv"], int(row["source_csv_row"]))
                observed_sources.append(source)
                if source_rows:
                    if source[0] not in source_rows or source[1] >= len(source_rows[source[0]]) or source[1] < 0:
                        source_mismatch_count += 1
                    else:
                        original = source_rows[source[0]][source[1]]
                        for column in ("family", "group_id", "decision_label", "relation", "candidate_source"):
                            if original[column] != row[column]:
                                source_mismatch_count += 1
        check(f"seed.{seed}.source_rows_unique", len(set(observed_sources)), len(observed_sources))
        check(f"seed.{seed}.no_cross_split_groups", all(len(parts) == 1 for parts in ownership.values()), True)
        if source_rows:
            check(f"seed.{seed}.input_rows_covered", set(observed_sources) == expected_sources, True)
            check(f"seed.{seed}.source_metadata_mismatches", source_mismatch_count, 0)
        for family in FAMILIES:
            groups = sorted({group for fam, group in ownership if fam == family})
            random = np.random.RandomState(seed)
            shuffled = np.asarray(groups)
            random.shuffle(shuffled)
            a, b = int(round(.4 * len(groups))), int(round(.3 * len(groups)))
            expected_groups = {"train": set(shuffled[:a]), "cal": set(shuffled[a:a+b]), "test": set(shuffled[a+b:])}
            for part in frames:
                check(f"seed.{seed}.split_ownership.{family}.{part}", observed_groups[part][family] == expected_groups[part], True)
            counts = [len(observed_groups[part][family]) for part in ("train", "cal", "test")]
            check(f"seed.{seed}.split_counts.{family}", counts, seed_result["splits"][family])
        names = VARIANTS if seed == SEEDS[0] else ["mlp", "mlp_margin"]
        check(f"seed.{seed}.variant_names", sorted(seed_result["variants"]), sorted(names))
        expected_score_keys = sorted(name + "_" + part for name in names for part in ("cal", "test"))
        check(f"seed.{seed}.score_keys", sorted(scores.files), expected_score_keys)
        for name in names:
            for part in ("cal", "test"):
                values = scores[name + "_" + part]
                check(f"seed.{seed}.{name}.{part}.score_count", list(values.shape), [len(frames[part])])
                check(f"seed.{seed}.{name}.{part}.finite_scores", bool(np.isfinite(values).all()), True)

        # Independently evaluate each saved fitted model from its weight arrays.
        all_columns = feature_metadata["columns"]
        for model_name in (["mlp", "core", "no_constraints"] if seed == SEEDS[0] else ["mlp"]):
            model_path = seed_dir / (model_name + "_model.npz")
            meta_path = seed_dir / (model_name + "_model.json")
            parameters = np.load(model_path, allow_pickle=False)
            meta = read_json(meta_path)
            remember(model_path)
            remember(meta_path)
            selected_columns = [all_columns.index(column) for column in meta["feature_columns"]]
            check(f"seed.{seed}.{model_name}.positive_class_order", meta["classes"], [0, 1])
            for part in ("cal", "test"):
                matrix = (features[part][:, selected_columns] - parameters["scaler_mean"]) / parameters["scaler_scale"]
                if model_name == "mlp":
                    hidden = np.maximum(0, matrix @ parameters["coef_0"] + parameters["intercept_0"])
                    margin = (hidden @ parameters["coef_1"] + parameters["intercept_1"]).ravel()
                    arrays_match(f"seed.{seed}.inference.mlp_margin.{part}", margin, scores["mlp_margin_" + part])
                else:
                    margin = (matrix @ parameters["coef"].T + parameters["intercept"]).ravel()
                arrays_match(f"seed.{seed}.inference.{model_name}.{part}", sigmoid(margin), scores[model_name + "_" + part])
        if seed == SEEDS[0]:
            for part in ("cal", "test"):
                for name in ("ngram4", "shingle3", "bm25n"):
                    arrays_match(f"seed.{seed}.inference.{name}.{part}", features[part][:, all_columns.index(name)], scores[name + "_" + part])
                fusion = features[part][:, [all_columns.index(column) for column in ("wcos", "ccos", "lsa", "bm25n")]].mean(axis=1)
                arrays_match(f"seed.{seed}.inference.fusion.{part}", fusion, scores["fusion_" + part])

        def check_evaluation(name, result, families, epsilon, prefix):
            cal_rows, cal_scores = subset(frames["cal"], scores[name + "_cal"], families)
            test_rows, test_scores = subset(frames["test"], scores[name + "_test"], families)
            threshold = threshold_from_scores(cal_rows, cal_scores, epsilon)
            actual = {"tau": threshold, "eps": epsilon, "families": families,
                      "cal": evaluate(cal_rows, cal_scores, threshold),
                      "test": evaluate(test_rows, test_scores, threshold)}
            positive = np.asarray([row["decision_label"] == "remove" for row in test_rows], dtype=int)
            actual["AUROC"] = pairwise_auc(positive, test_scores)
            for family in families:
                rr, ss = subset(test_rows, test_scores, [family])
                actual["test_" + family] = evaluate(rr, ss, threshold)
            if "test_CR_ci95" in result:
                actual["test_CR_ci95"] = bootstrap_recall(test_rows, test_scores, threshold)
            if "breakdown" in result:
                actual["breakdown"] = error_breakdown(test_rows, test_scores, threshold)
            check(prefix, actual, result)
            return finite_json(actual)

        primary_evaluations = {}
        for name in names:
            budgets = ["eps5", "eps10"] if seed == SEEDS[0] else ["eps5"]
            check(f"seed.{seed}.{name}.budgets", sorted(seed_result["variants"][name]), sorted(budgets))
            primary_evaluations[name] = {}
            for budget in budgets:
                evaluation = check_evaluation(name, seed_result["variants"][name][budget], MAIN_FAMILIES,
                                              .05 if budget == "eps5" else .1, f"seed.{seed}.{name}.{budget}")
                primary_evaluations[name][budget] = evaluation
            if name in ("mlp", "mlp_margin"):
                values = primary_evaluations[name]["eps5"]
                report["recomputed"]["sensitivity"][name].append({"seed": seed, "tau": values["tau"],
                    **{key: values["test"][key] for key in ("CR", "CDR", "GCDR", "damaged_groups")}})
        math_result = check_evaluation("mlp_margin", seed_result["math_margin_local"], ["math"], .05,
                                       f"seed.{seed}.math_margin_local")
        report["recomputed"]["sensitivity_math_margin"].append({"seed": seed,
            "CR": math_result["test"]["CR"], "GCDR": math_result["test"]["GCDR"]})
        if seed == SEEDS[0]:
            report["recomputed"]["primary"]["variants"] = primary_evaluations
            locals_ = {}
            for family in FAMILIES:
                locals_[family] = check_evaluation("mlp_margin", seed_result["margin_family_local"][family], [family], .05,
                                                   f"seed.{seed}.margin_local.{family}")
            report["recomputed"]["primary"]["margin_family_local"] = locals_
            transfer = {"taus": {family: locals_[family]["tau"] for family in FAMILIES},
                        "matrix": {}, "scorer": "MLP margin"}
            for source in FAMILIES:
                tau = locals_[source]["tau"]
                for target in FAMILIES:
                    rr, ss = subset(frames["test"], scores["mlp_margin_test"], [target])
                    metric = evaluate(rr, ss, math.inf if tau is None else tau)
                    metric["regret"] = max(0., locals_[target]["test"]["CR"] - metric["CR"])
                    transfer["matrix"][source + "->" + target] = metric
            check("primary.margin_transfer", transfer, seed_result["margin_transfer"])
            report["recomputed"]["primary"]["margin_transfer"] = transfer
            check("primary.top_level_matches_seed_record", recorded["primary"], seed_result)
        scores.close()
        features.close()

    for name in ("mlp", "mlp_margin"):
        for actual, reference in zip(report["recomputed"]["sensitivity"][name], recorded["sensitivity"][name]):
            check(f"sensitivity.{name}.{actual['seed']}", actual, reference)
        check(f"sensitivity.{name}.length", len(recorded["sensitivity"][name]), len(SEEDS))
    for actual, reference in zip(report["recomputed"]["sensitivity_math_margin"], recorded["sensitivity_math_margin"]):
        check(f"sensitivity.math.{actual['seed']}", actual, reference)
    check("sensitivity.math.length", len(recorded["sensitivity_math_margin"]), len(SEEDS))
    summary = {}
    for name, values in report["recomputed"]["sensitivity"].items():
        recalls = np.asarray([row["CR"] for row in values])
        summary[name] = {"mean": float(recalls.mean()), "population_sd": float(recalls.std(ddof=0)),
                         "min": float(recalls.min()), "max": float(recalls.max()),
                         "zero_recall_splits": int(np.count_nonzero(recalls == 0))}
    check("sensitivity_summary", summary, recorded["sensitivity_summary"])
    report["recomputed"]["sensitivity_summary"] = summary
    recalls = np.asarray([row["CR"] for row in report["recomputed"]["sensitivity_math_margin"]])
    math_summary = {"mean": float(recalls.mean()), "population_sd": float(recalls.std(ddof=0)),
                    "min": float(recalls.min()), "max": float(recalls.max())}
    check("sensitivity_math_summary", math_summary, recorded["sensitivity_math_margin_summary"])
    report["recomputed"]["sensitivity_math_margin_summary"] = math_summary
    report["limitations"] = [
        "Scorer training and raw-text feature extraction are not rerun; saved fitted-model inference is checked against saved features and scores.",
        "Label correctness, benchmark derivation, and population sampling assumptions are not established by arithmetic replay.",
        "The historical protocol selected thresholds using calibration positive recall and damage tie-breaking; successful replay does not upgrade this exploratory pipeline to a valid prespecified confirmatory guarantee.",
        "Exact input CSV hash and row mapping checked." if args.input_dir else "Source CSVs not provided; frame-to-source correspondence was not checked.",
    ]
    report["overall_status"] = "FAIL" if report["failed_checks"] else "PASS"
    report["checks_passed"] = sum(item["status"] == "PASS" for item in report["checks"])
    report["checks_failed"] = len(report["failed_checks"])
    return finite_json(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the completed run_manifest.json, results.json, and seeds/")
    parser.add_argument("--input-dir", type=Path, help="Canonical pair CSV directory; enables input hashes and source-row correspondence checks")
    parser.add_argument("--output", type=Path, default=Path("constructed_verification_report.json"))
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        report = {"overall_status": "ERROR", "error_type": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['overall_status']}: {args.output}")
    if report.get("failed_checks"):
        print("Failed checks: " + ", ".join(report["failed_checks"]))
    if report.get("error"):
        print(report["error"])
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
