#!/usr/bin/env python3
"""Compare the fixed pilot CSV with unchanged historical construction code."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--paired-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    import run_r2_wild as historical
    historical.DATA = str(args.data_dir.resolve())
    historical.REP = str((args.data_dir / "derived_real/contamination_reports_realweb").resolve())
    class ConstructionComplete(Exception):
        pass
    def stop_before_model():
        raise ConstructionComplete()
    historical.load_pairs = stop_before_model
    state = {}
    def observe(frame, event, arg):
        if event == "return" and frame.f_code is historical.main.__code__:
            state.update(frame.f_locals)
    sys.setprofile(observe)
    try:
        historical.main()
    except ConstructionComplete:
        pass
    finally:
        sys.setprofile(None)
    generated = state["wild"].reset_index(drop=True).fillna("")
    saved = pd.read_csv(args.paired_csv).fillna("")
    fields = ["benchmark_text", "benchmark_answer", "candidate_text", "candidate_date", "benchmark_release",
              "family", "group_id", "decision_label", "relation", "subject", "silver"]
    assert len(saved) == len(generated) == 238
    for field in fields:
        assert saved[field].tolist() == generated[field].tolist(), field
    result = {"passed": True, "rows": len(saved), "checked_fields": fields,
              "construction": "unchanged run_r2_wild.py main stopped immediately before model loading",
              "source_sha256": hashlib.sha256(Path(historical.__file__).read_bytes()).hexdigest(),
              "paired_csv_sha256": hashlib.sha256(args.paired_csv.read_bytes()).hexdigest(),
              "counts": generated.silver.value_counts().to_dict()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
