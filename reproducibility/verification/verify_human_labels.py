#!/usr/bin/env python3
"""Recompute an audit from supplied human CSV rows; never substitute paper counts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

from verify_audit import cp_interval, cp_upper, file_hash, pairwise_auc

VALID_LABELS = {"clean", "input", "input_and_label", "uncertain"}
VALID_ACCESS = {"yes", "no", "unknown"}


def read_csv(path: Path, id_column: str, label_column: str,
             access_column: str | None = None) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, label_column} | ({access_column} if access_column else set())
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        records = {}
        for row in reader:
            item_id = row[id_column].strip()
            if not item_id or item_id in records:
                raise ValueError(f"{path.name}: empty or duplicate item ID")
            label = row[label_column].strip()
            if label not in VALID_LABELS:
                raise ValueError(f"{path.name}: invalid label {label!r}")
            parsed = {"label": label}
            if access_column:
                access = row[access_column].strip().lower()
                if access not in VALID_ACCESS:
                    raise ValueError(f"{path.name}: access must be yes, no, or unknown")
                parsed["access"] = access
            records[item_id] = parsed
    if not records:
        raise ValueError(f"{path.name}: no records")
    return records


def kappa(a: list, b: list) -> dict:
    if not a:
        return {"n": 0, "agreement_n": 0, "agreement": None, "kappa": None,
                "undefined_reason": "empty subset"}
    n = len(a)
    agreement_n = sum(x == y for x, y in zip(a, b))
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b[label]
                   for label in set(a) | set(b)) / n**2
    value = None if expected == 1 else (agreement_n / n - expected) / (1 - expected)
    return {"n": n, "agreement_n": agreement_n, "agreement": agreement_n / n,
            "kappa": value,
            "undefined_reason": "constant identical labels" if value is None else None}


def metrics(rows: list[dict]) -> dict:
    certain = [row for row in rows if row["label"] != "uncertain"]
    labels = np.asarray([int(row["label"] != "clean") for row in certain], dtype=int)
    removed = np.asarray([row["removed"] for row in certain], dtype=bool)
    scores = np.asarray([row["score"] for row in certain], dtype=float)
    positive, clean = labels == 1, labels == 0
    tp = int((removed & positive).sum())
    fp = int((removed & clean).sum())
    positive_n, clean_n = int(positive.sum()), int(clean.sum())
    return {
        "total_n": len(rows), "uncertain_excluded_n": len(rows) - len(certain),
        "certain_n": len(certain), "contaminated_n": positive_n, "clean_n": clean_n,
        "true_positive_n": tp, "false_negative_n": positive_n - tp,
        "false_positive_n": fp, "true_negative_n": clean_n - fp,
        "removed_among_uncertain_n": sum(row["removed"] for row in rows if row["label"] == "uncertain"),
        "recall": tp / positive_n if positive_n else None,
        "recall_two_sided_95_cp": cp_interval(tp, positive_n) if positive_n else None,
        "clean_deletion_rate": fp / clean_n if clean_n else None,
        "clean_deletion_one_sided_95_upper": cp_upper(fp, clean_n) if clean_n else None,
        "precision_excluding_uncertain": tp / (tp + fp) if tp + fp else None,
        "precision_excluding_uncertain_two_sided_95_cp": cp_interval(tp, tp + fp) if tp + fp else None,
        "auroc": pairwise_auc(labels, scores) if positive_n and clean_n else None,
        "null_metric_meaning": "Undefined due to an empty denominator or absent class.",
    }


def run(args: argparse.Namespace) -> dict:
    if args.final_labels is None:
        return {
            "overall_status": "UNAVAILABLE",
            "reason": "No item-level final human labels supplied. No human results were computed.",
            "substitute_aggregate_counts_used": False,
        }
    raw = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    predictions = {}
    for row in raw:
        item_id = row["source_id"]
        if not item_id or item_id in predictions:
            raise ValueError("Prediction rows have an empty or duplicate source ID")
        if not (math.isfinite(float(row["score"])) and math.isfinite(float(row["threshold"]))):
            raise ValueError("Prediction rows contain non-finite scores or thresholds")
        if not isinstance(row["removed"], bool):
            raise ValueError("Prediction removed flags must be JSON booleans")
        if row["removed"] != (row["score"] >= row["threshold"]):
            raise ValueError("Prediction removal flag conflicts with its frozen threshold")
        if row.get("silver_label") not in (0, 1):
            raise ValueError("Prediction silver_label must be 0 or 1")
        predictions[item_id] = row
    final = read_csv(args.final_labels, args.id_column, args.label_column, args.access_column)
    if set(final) != set(predictions):
        raise ValueError("Final-label and prediction ID sets differ; no intersection-only evaluation is allowed")
    ids = sorted(final)
    rows = [{**final[item_id], **{key: predictions[item_id][key]
                                for key in ("score", "removed", "silver_label")}}
            for item_id in ids]
    certain = [row for row in rows if row["label"] != "uncertain"]
    report = {
        "overall_status": "RECOMPUTED_FROM_SUPPLIED_LABEL_ROWS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "substitute_aggregate_counts_used": False,
        "inputs": {path.name: {"sha256": file_hash(path)}
                   for path in (args.predictions, args.final_labels)},
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "access_counts": {value: sum(row["access"] == value for row in rows)
                          for value in sorted(VALID_ACCESS)},
        "all_items": metrics(rows),
        "access_subgroups": {value: metrics([row for row in rows if row["access"] == value])
                             for value in sorted(VALID_ACCESS)},
        "silver_vs_final_human_binary_agreement": kappa(
            [row["silver_label"] for row in certain],
            [int(row["label"] != "clean") for row in certain]),
        "annotator_agreement": {"status": "UNAVAILABLE", "reason": "No pair of annotator CSVs supplied."},
        "limitations": [
            "Calculations use supplied labels; they do not establish annotator independence, correctness, blinding, or adjudication provenance.",
            "The final CSV defines full-page access explicitly; yes/no/unknown groups are reported separately.",
            "Uncertain labels are excluded from recall, clean-deletion, precision, and AUROC; their removal count is reported separately.",
        ],
    }
    if bool(args.annotator_a) != bool(args.annotator_b):
        raise ValueError("Provide both --annotator-a and --annotator-b, or neither")
    if args.annotator_a:
        a = read_csv(args.annotator_a, args.id_column, args.annotator_label_column)
        b = read_csv(args.annotator_b, args.id_column, args.annotator_label_column)
        if set(a) != set(predictions) or set(b) != set(predictions):
            raise ValueError("Annotator and prediction ID sets differ")
        labels_a = [a[item_id]["label"] for item_id in ids]
        labels_b = [b[item_id]["label"] for item_id in ids]
        both_certain = [(x, y) for x, y in zip(labels_a, labels_b)
                        if x != "uncertain" and y != "uncertain"]
        report["annotator_agreement"] = {
            "status": "RECOMPUTED_FROM_SUPPLIED_LABEL_ROWS",
            "four_class": kappa(labels_a, labels_b),
            "binary_both_certain": kappa([int(x != "clean") for x, _ in both_certain],
                                          [int(y != "clean") for _, y in both_certain]),
            "four_class_confusion_counts": {
                x: {y: sum(aa == x and bb == y for aa, bb in zip(labels_a, labels_b))
                    for y in sorted(VALID_LABELS)} for x in sorted(VALID_LABELS)},
        }
        for path in (args.annotator_a, args.annotator_b):
            report["inputs"][path.name] = {"sha256": file_hash(path)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--final-labels", type=Path)
    parser.add_argument("--id-column", default="item_id")
    parser.add_argument("--label-column", default="human_label")
    parser.add_argument("--access-column", default="full_page_accessible")
    parser.add_argument("--annotator-a", type=Path)
    parser.add_argument("--annotator-b", type=Path)
    parser.add_argument("--annotator-label-column", default="item_label")
    parser.add_argument("--output", type=Path, default=Path("human_label_verification_report.json"))
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        report = {"overall_status": "ERROR", "error_type": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['overall_status']}: {args.output}")
    return {"RECOMPUTED_FROM_SUPPLIED_LABEL_ROWS": 0, "UNAVAILABLE": 2}.get(report["overall_status"], 1)


if __name__ == "__main__":
    sys.exit(main())
