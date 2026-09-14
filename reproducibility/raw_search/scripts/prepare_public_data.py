#!/usr/bin/env python3
"""Fetch and verify the pinned public inputs used by SafeDecontam."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/Contamination_Detector"
RAW = ROOT / "data/raw/contamination_detector"
COMMIT = "ffe3c249309da9a7ad7f42e9145e1f8e4d36c77e"
BASE = "https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1"
ASSETS = {
    "benchmarks.zip": "689f98bd4edb9a7db75cdcd03d0b453ab8cdaf7f2334e46d3e4c6b42e50d335c",
    "bing_search.zip": "89bc89d70a9a4b0855c3da15598ac74dd2308fd36fa31d25a9a7d1d7a33bb6b5",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    if not REPO.exists():
        REPO.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/liyucheng09/Contamination_Detector.git", str(REPO))
    run("git", "-C", str(REPO), "fetch", "origin", COMMIT)
    run("git", "-C", str(REPO), "checkout", "--detach", COMMIT)

    RAW.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        archive = RAW / name
        if not archive.exists():
            print(f"Downloading {name} ...")
            with urllib.request.urlopen(f"{BASE}/{name}") as response, archive.open("wb") as target:
                shutil.copyfileobj(response, target)
        actual = digest(archive)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {archive}: {actual}")
        destination = RAW / archive.stem
        if not destination.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(destination)
        print(f"Verified {name}: {actual}")


if __name__ == "__main__":
    main()
