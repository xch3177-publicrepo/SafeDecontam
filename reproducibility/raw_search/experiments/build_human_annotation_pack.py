#!/usr/bin/env python3
"""Build a blinded, reviewer-ready annotation pack for the locked MMLU audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/experiments/raw_bing_item_audit/dataset.jsonl"
OUT = ROOT / "output/experiments/human_annotation_pack"


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    test = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["source_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    max_pages = max(len(r["pages"]) for r in test)
    base = ["item_id", "benchmark_query", "page_count"]
    page_columns = [x for i in range(1, max_pages + 1) for x in (f"url_{i}", f"snippet_{i}")]
    labels = [
        "annotator_id", "item_label", "strongest_page_index", "answer_visible",
        "full_page_accessible", "confidence", "notes",
    ]
    with (OUT / "annotation_blind.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base + page_columns + labels)
        w.writeheader()
        for r in test:
            row = {"item_id": r["source_id"], "benchmark_query": r["query"], "page_count": len(r["pages"])}
            for i, p in enumerate(r["pages"], 1):
                row[f"url_{i}"] = p["url"]
                row[f"snippet_{i}"] = p["text"]
            w.writerow(row)
    with (OUT / "silver_reference_sealed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "released_silver_category", "binary_silver_label"])
        w.writeheader()
        for r in test:
            w.writerow({
                "item_id": r["source_id"],
                "released_silver_category": r["category"],
                "binary_silver_label": r["label"],
            })
    manifest = {
        "benchmark": "MMLU",
        "partition": "locked test",
        "items": len(test),
        "silver_positive": sum(r["label"] for r in test),
        "silver_clean": sum(1 - r["label"] for r in test),
        "instructions": "Keep silver_reference_sealed.csv hidden until both annotators submit labels.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
