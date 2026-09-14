#!/usr/bin/env python3
"""Merge agreement/adjudication labels and evaluate silver labels and detector decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "output/experiments/human_annotation_pack/annotation_comparison.json"
ADJUDICATED_EXTRACT = ROOT / "tmp/adjudicated_extract.json"
SILVER = ROOT / "output/experiments/human_annotation_pack/silver_reference_sealed.csv"
PREDICTIONS = ROOT / "output/experiments/raw_bing_item_audit/locked_test_predictions.jsonl"
OUT_DIR = ROOT / "output/experiments/human_annotation_pack"
VALID = {"clean", "input", "input_and_label", "uncertain"}


def exact_ci(successes, n, alpha=.05):
    if n == 0:
        return [None, None]
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return [lo, hi]


def metrics(rows):
    rows = [r for r in rows if r["human_label"] != "uncertain"]
    y = np.asarray([int(r["human_label"] != "clean") for r in rows])
    p = np.asarray([int(r["removed"]) for r in rows])
    s = np.asarray([float(r["score"]) for r in rows])
    tp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 1)).sum())
    fp, tn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 0)).sum())
    return {
        "n": len(rows), "positive_n": tp + fn, "clean_n": fp + tn,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / (tp + fn), "recall_exact_95_ci": exact_ci(tp, tp + fn),
        "clean_removal_rate": fp / (fp + tn), "clean_removal_exact_95_ci": exact_ci(fp, fp + tn),
        "precision": tp / (tp + fp), "precision_exact_95_ci": exact_ci(tp, tp + fp),
        "specificity": tn / (tn + fp),
        "auroc": float(roc_auc_score(y, s)),
    }


def main():
    comparison = json.loads(COMPARISON.read_text())
    matrix = json.loads(ADJUDICATED_EXTRACT.read_text())
    header = matrix[0]
    adjudicated = {r[0]: dict(zip(header, r)) for r in matrix[1:] if r and r[0]}
    disagreements = {r["item_id"] for r in comparison["records"] if not r["agreement"]}
    if set(adjudicated) != disagreements:
        raise ValueError("Adjudicated item IDs do not match the disagreement set")
    for item_id, row in adjudicated.items():
        label = str(row.get("adjudicated_label") or "").strip()
        rationale = str(row.get("adjudication_rationale") or "").strip()
        if label not in VALID or not rationale:
            raise ValueError(f"Incomplete adjudication for {item_id}")
    silver = {r["item_id"]: r for r in csv.DictReader(SILVER.open())}
    predictions = {r["source_id"]: r for r in map(json.loads, PREDICTIONS.read_text().splitlines())}
    final = []
    for r in comparison["records"]:
        item_id = r["item_id"]
        human = r["label_a"] if r["agreement"] else adjudicated[item_id]["adjudicated_label"]
        rationale = "A01/A02 agreement" if r["agreement"] else adjudicated[item_id]["adjudication_rationale"]
        pred = predictions[item_id]
        final.append({
            "item_id": item_id,
            "human_label": human,
            "human_binary": "" if human == "uncertain" else int(human != "clean"),
            "label_source": "agreement" if r["agreement"] else "third-adjudicator",
            "adjudication_rationale": rationale,
            "full_page_any": r["full_page_a"].strip().lower() == "yes" or r["full_page_b"].strip().lower() == "yes",
            "full_page_both": r["full_page_a"].strip().lower() == "yes" and r["full_page_b"].strip().lower() == "yes",
            "silver_category": silver[item_id]["released_silver_category"],
            "silver_binary": int(silver[item_id]["binary_silver_label"]),
            "score": pred["score"], "threshold": pred["threshold"], "removed": pred["removed"],
        })
    certain = [r for r in final if r["human_label"] != "uncertain"]
    yh = np.asarray([int(r["human_binary"]) for r in certain])
    ys = np.asarray([r["silver_binary"] for r in certain])
    result = {
        "label_counts": {lab: sum(r["human_label"] == lab for r in final) for lab in sorted(VALID)},
        "adjudicated_disagreements": len(disagreements),
        "primary_excludes_uncertain": sum(r["human_label"] == "uncertain" for r in final),
        "detector_vs_adjudicated_human": metrics(final),
        "detector_full_page_any": metrics([r for r in final if r["full_page_any"]]),
        "detector_snippet_only": metrics([r for r in final if not r["full_page_any"]]),
        "silver_vs_adjudicated_human": {
            "n": len(certain),
            "agreement": float((yh == ys).mean()),
            "cohen_kappa": float(cohen_kappa_score(yh, ys)),
            "silver_recall_against_human": float(ys[yh == 1].mean()),
            "silver_clean_error_against_human": float(ys[yh == 0].mean()),
        },
        "annotation_agreement_before_adjudication": {
            "four_class_agreement": comparison["exact_agreement"],
            "four_class_kappa": comparison["cohen_kappa_four_class"],
            "binary_both_certain_agreement": comparison["binary_both_certain_agreement"],
            "binary_both_certain_kappa": comparison["binary_both_certain_kappa"],
        },
        "limitations": [
            "Only 51/154 items had full-page access by at least one annotator; 103/154 were snippet-only.",
            "A02 left answer_visible blank for 73 items; final item labels and rationales were complete.",
        ],
    }
    (OUT_DIR / "human_adjudication_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "human_adjudicated_labels.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    with (OUT_DIR / "human_adjudicated_labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final[0]))
        w.writeheader(); w.writerows(final)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
