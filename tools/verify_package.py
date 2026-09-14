#!/usr/bin/env python3
"""Verify the compact package against CHECKSUMS_SHA256.txt."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = ROOT / "CHECKSUMS_SHA256.txt"
IGNORED = {"CHECKSUMS_SHA256.txt"}
# Directories that a git checkout or a run of the experiments creates; they are
# not part of the distributed package (see .gitignore).
IGNORED_DIRS = {".git", ".venv", "__pycache__", "data", "external", "models", "output", "tmp", "runs"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    expected = {}
    for line in CHECKSUMS.read_text().splitlines():
        value, name = line.split("  ", 1)
        expected[name] = value
    actual_files = {
        p.relative_to(ROOT).as_posix(): p
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.relative_to(ROOT).as_posix() not in IGNORED
        and not (set(p.relative_to(ROOT).parts[:-1]) & IGNORED_DIRS)
    }
    missing = sorted(set(expected) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected))
    mismatched = sorted(name for name, value in expected.items()
                        if name in actual_files and digest(actual_files[name]) != value)
    if missing or extra or mismatched:
        raise SystemExit(f"missing={missing}\nextra={extra}\nmismatched={mismatched}")
    print(f"OK: verified {len(expected)} files")


if __name__ == "__main__":
    main()
