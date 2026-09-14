#!/usr/bin/env python3
"""Run the unchanged raw-search audit and export its in-memory replay inputs.

This wrapper uses a return profiler only to observe the audit function's local
variables. It does not replace the scorer, splitting, feature, or calibration
functions. Outputs are a new run, never the missing historical artifacts.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import run_raw_bing_item_audit as audit
import run_web_evidence_experiment as shared


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_replay(state):
    out = state["out"]
    rows = state["rows"]
    score = state["score"]
    names = list(state["models"])
    result = state["result"]
    records = []
    for split, items in rows.items():
        scores = {name: score(name, split) for name in names}
        for i, row in enumerate(items):
            records.append({
                "source_id": row["source_id"],
                "split": split,
                "silver_category": row["category"],
                "silver_label": int(row["label"]),
                "scores": {name: float(scores[name][i]) for name in names},
            })
    (out / "all_split_scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    selected = result["selected_scorer"]
    clean_cal = [r for r in records if r["split"] == "calibration" and r["silver_label"] == 0]
    (out / "calibration_clean_scores.json").write_text(json.dumps({
        "selected_scorer": selected,
        "source_ids": [r["source_id"] for r in clean_cal],
        "scores": [r["scores"][selected] for r in clean_cal],
        "calibration": result["calibration"],
    }, indent=2) + "\n")
    arrays = {}
    for split in rows:
        arrays[f"features_{split}"] = state["rawx"][split]
        arrays[f"labels_{split}"] = state["y"][split]
    np.savez_compressed(out / "raw_item_features.npz", **arrays)
    (out / "rerun_provenance.json").write_text(json.dumps({
        "status": "independent rerun; not a recovered historical result",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": sys.argv[1:],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "source_sha256": {"run_raw_bing_item_audit.py": digest(audit.__file__),
                          "run_web_evidence_experiment.py": digest(shared.__file__),
                          "run_external_with_replay.py": digest(__file__)},
        "data_source": result["data_source"],
        "annotation_sha256": digest(audit.ANN_ROOT / (result["dataset"] + "_annotations.json")),
        "dataset_sha256": digest(out / "dataset.jsonl"),
        "capture": "Read main return-frame locals after the unchanged original audit completes.",
    }, indent=2) + "\n")


def main():
    captured = {}
    code = audit.main.__code__
    def observe(frame, event, arg):
        if event == "return" and frame.f_code is code:
            captured.update(frame.f_locals)
    sys.setprofile(observe)
    try:
        audit.main()
    finally:
        sys.setprofile(None)
    if "result" not in captured:
        raise RuntimeError("The audit did not complete; no replay artifacts exported.")
    export_replay(captured)


if __name__ == "__main__":
    main()
