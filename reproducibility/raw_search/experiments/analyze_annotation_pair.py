#!/usr/bin/env python3
"""Validate two blinded annotation files and prepare an auditable comparison."""

from __future__ import annotations

import csv
import json
import argparse
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output/experiments/human_annotation_pack/annotation_comparison.json"
LABELS = ["clean", "input", "input_and_label", "uncertain"]
MUTABLE = {
    "annotator_id", "item_label", "strongest_page_index", "answer_visible",
    "full_page_accessible", "confidence", "notes",
}


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate item_id in {path}")
    return {r["item_id"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    inputs = {"A": args.annotator_a, "B": args.annotator_b}
    data = {name: read(path) for name, path in inputs.items()}
    ids = sorted(set(data["A"]) | set(data["B"]))
    if set(data["A"]) != set(data["B"]):
        raise ValueError("Annotator item sets differ")
    source_differences = []
    for item_id in ids:
        for col in data["A"][item_id]:
            if col not in MUTABLE and data["A"][item_id][col] != data["B"][item_id][col]:
                source_differences.append({"item_id": item_id, "column": col})
    a = [data["A"][i]["item_label"].strip() for i in ids]
    b = [data["B"][i]["item_label"].strip() for i in ids]
    invalid = sorted((set(a) | set(b)) - set(LABELS))
    if invalid:
        raise ValueError(f"Invalid labels: {invalid}")
    certain = [i for i, (x, y) in enumerate(zip(a, b)) if x != "uncertain" and y != "uncertain"]
    ab = [int(a[i] != "clean") for i in certain]
    bb = [int(b[i] != "clean") for i in certain]
    records = []
    for item_id in ids:
        ra, rb = data["A"][item_id], data["B"][item_id]
        la, lb = ra["item_label"].strip(), rb["item_label"].strip()
        records.append({
            "item_id": item_id,
            "benchmark_query": ra["benchmark_query"],
            "page_count": int(ra["page_count"]),
            "label_a": la,
            "label_b": lb,
            "agreement": la == lb,
            "disagreement_type": "agreement" if la == lb else (
                "uncertain-involved" if "uncertain" in (la, lb) else "substantive"),
            "strongest_page_a": ra["strongest_page_index"],
            "strongest_page_b": rb["strongest_page_index"],
            "answer_visible_a": ra["answer_visible"],
            "answer_visible_b": rb["answer_visible"],
            "confidence_a": ra["confidence"],
            "confidence_b": rb["confidence"],
            "full_page_a": ra["full_page_accessible"],
            "full_page_b": rb["full_page_accessible"],
            "notes_a": ra["notes"],
            "notes_b": rb["notes"],
            "urls": [ra[f"url_{j}"] for j in range(1, 11) if ra.get(f"url_{j}")],
            "snippets": [ra[f"snippet_{j}"] for j in range(1, 11) if ra.get(f"snippet_{j}")],
        })
    full_counts = {
        k: Counter(r["full_page_accessible"].strip().lower() for r in d.values()) for k, d in data.items()
    }
    full_union_yes = sum(
        data["A"][i]["full_page_accessible"].strip().lower() == "yes" or
        data["B"][i]["full_page_accessible"].strip().lower() == "yes" for i in ids
    )
    full_both_yes = sum(
        data["A"][i]["full_page_accessible"].strip().lower() == "yes" and
        data["B"][i]["full_page_accessible"].strip().lower() == "yes" for i in ids
    )
    missing_fields = {
        k: {
            field: sum(not r[field].strip() for r in d.values())
            for field in ("item_label", "answer_visible", "full_page_accessible", "confidence", "notes")
        } for k, d in data.items()
    }
    result = {
        "provenance_warning": (
            f"Annotators A01 and A02 report full-page access for {full_counts['A'].get('yes', 0)} and "
            f"{full_counts['B'].get('yes', 0)} items; at least one reports access for {full_union_yes}/154 items "
            f"and both report access for {full_both_yes}/154. Remaining judgments are snippet-level."
        ),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "annotator_ids": {k: sorted({r["annotator_id"] for r in d.values()}) for k, d in data.items()},
        "n": len(ids),
        "source_field_differences": source_differences,
        "label_counts": {"A": Counter(a), "B": Counter(b)},
        "exact_agreement_n": sum(x == y for x, y in zip(a, b)),
        "exact_agreement": sum(x == y for x, y in zip(a, b)) / len(a),
        "cohen_kappa_four_class": float(cohen_kappa_score(a, b, labels=LABELS)),
        "confusion_labels": LABELS,
        "confusion_matrix": confusion_matrix(a, b, labels=LABELS).tolist(),
        "binary_both_certain_n": len(certain),
        "binary_both_certain_agreement": sum(x == y for x, y in zip(ab, bb)) / len(certain),
        "binary_both_certain_kappa": float(cohen_kappa_score(ab, bb)),
        "disagreement_n": sum(x != y for x, y in zip(a, b)),
        "uncertain_involved_disagreement_n": sum(
            x != y and "uncertain" in (x, y) for x, y in zip(a, b)
        ),
        "full_page_counts": full_counts,
        "full_page_union_yes": full_union_yes,
        "full_page_both_yes": full_both_yes,
        "missing_fields": missing_fields,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
