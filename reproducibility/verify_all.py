#!/usr/bin/env python3
"""Run saved-evidence checks without fitting models or inventing human labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output directory; existing checks are preserved.")
    args.output.mkdir(parents=True)
    checks = [
        ("mmlu", ["verification/verify_audit.py", "--audit-dir", "raw_search/results/main_audit",
                  "--output", str(args.output.resolve() / "mmlu.json")]),
        ("external", ["verification/verify_external_replay.py", "evidence/external_reruns/ARC",
                      "evidence/external_reruns/hellaswag", "evidence/external_reruns/ceval"]),
        ("constructed", ["verification/verify_current_constructed.py", "evidence/constructed_current",
                         "--input-dir", "inputs/constructed",
                         "--output", str(args.output.resolve() / "constructed.json")]),
        ("auxiliary", ["verification/verify_auxiliary_probes.py", "evidence/auxiliary_reruns"]),
        ("calibration_unit_tests", ["-m", "unittest", "discover", "-s", "calibration_reference", "-v"]),
    ]
    results = []
    for name, arguments in checks:
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT,
                                text=True, capture_output=True)
        (args.output / (name + ".log")).write_text(result.stdout + result.stderr)
        results.append({"name": name, "exit_code": result.returncode,
                        "passed": result.returncode == 0, "command": [sys.executable, *arguments]})
        print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
    summary = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "checks": results,
               "human_row_level_replay": "Unavailable: final human label tables were not supplied.",
               "scope": "Saved-score, model-inference and arithmetic checks; no new model fitting."}
    (args.output / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
