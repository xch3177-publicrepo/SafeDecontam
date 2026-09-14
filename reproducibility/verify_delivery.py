#!/usr/bin/env python3
"""Verify the recorded payload hashes; computation has separate entry points."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
failures = []
checked = 0
for line in (root / "SHA256SUMS.txt").read_text().splitlines():
    expected, name = line.split("  ", 1)
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
    checked += 1
if failures:
    raise SystemExit("Integrity check failed: " + ", ".join(failures))
print(f"PASS: verified {checked} payload file hashes; no experiment was run.")
