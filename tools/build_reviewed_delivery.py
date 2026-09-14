#!/usr/bin/env python3
"""Package the reviewed manuscript and current reproduction tree without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", ".DS_Store", "output", "tmp", "external", "runs"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file()
                  and not (set(p.relative_to(folder).parts) & EXCLUDED))


def manifest(folder):
    target = folder / "SHA256SUMS.txt"
    target.write_text("".join(f"{digest(p)}  {p.relative_to(folder).as_posix()}\n"
                              for p in payload_files(folder) if p != target))


def zip_tree(folder, target):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in payload_files(folder):
            name = folder.name + "/" + path.relative_to(folder).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
            archive.writestr(info, path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--version", default="iccc2026-repro-20260914-v2")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new directory to preserve prior delivery.")
    args.output.mkdir(parents=True)
    package = args.output / "SafeDecontam_ICCC2026_Reproduction"
    source = package / "experiment_source"
    shutil.copytree(ROOT / "reproducibility", source,
                    ignore=shutil.ignore_patterns(*EXCLUDED, "*.pyc"))
    manuscript = package / "manuscript"
    manuscript.mkdir()
    for name in ["SafeDecontam_ICCC2026_camera_ready_5p.pdf",
                 "SafeDecontam_ICCC2026_revised.tex", "IEEEtran.cls", "fig1_cropped.png"]:
        shutil.copyfile(ROOT / name, manuscript / name)
    shutil.copyfile(ROOT / "build_camera_ready.sh", manuscript / "build.sh")
    shutil.copyfile(ROOT / "LICENSE", package / "LICENSE")
    shutil.copyfile(ROOT / "CITATION.cff", package / "CITATION.cff")
    shutil.copytree(ROOT / "revisions/2026-09-14-external-review", package / "validation")
    availability = ROOT / "revisions/2026-09-14-availability"
    if availability.is_dir():
        shutil.copytree(availability, package / "validation/availability")
    shutil.copyfile(ROOT / "reproducibility/verify_delivery.py", package / "verify_delivery.py")
    metadata = {"package_version": args.version, "source_commit": args.source_commit,
                "manuscript_pdf_sha256": digest(manuscript / "SafeDecontam_ICCC2026_camera_ready_5p.pdf"),
                "manuscript_tex_sha256": digest(manuscript / "SafeDecontam_ICCC2026_revised.tex"),
                "baseline_commit": "7e1c30a8ada23e55784b5050a8a9628e0f2f4110"}
    (package / "VERSION.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (package / "README.md").write_text(
        "# SafeDecontam reviewed reproduction delivery\n\n"
        f"Package version: `{args.version}`. Source commit: `{args.source_commit}`.\n\n"
        "`manuscript/` contains the five-page PDF, matching LaTeX, figure and build script. "
        "`experiment_source/` contains code, inputs, saved scores, evidence and verification commands. "
        "`validation/REVIEW_RESPONSE.md` explains the evidence corrections and remaining access limits.\n\n"
        + ("`validation/availability/` records the subsequent code and data availability statement update; "
           "the external-review validation snapshot is preserved separately.\n\n"
           if availability.is_dir() else "") +
        "Run `python3 verify_delivery.py` to check file integrity. "
        "Then follow `experiment_source/README.md` for computational verification. "
        "A checksum check alone is not experimental reproduction.\n")
    compact = args.output / "SafeDecontam_ICCC2026_PDF_LaTeX"
    shutil.copytree(manuscript, compact)
    shutil.copyfile(package / "VERSION.json", compact / "VERSION.json")
    shutil.copyfile(package / "verify_delivery.py", compact / "verify_delivery.py")
    (compact / "README.md").write_text(
        "# SafeDecontam five-page manuscript package\n\n"
        f"Package version: `{args.version}`. Source commit: `{args.source_commit}`.\n\n"
        "This compact package contains the compiled five-page PDF, matching LaTeX source, "
        "IEEEtran class, figure, build script, version metadata and integrity checker. "
        "It does not contain experimental code or data.\n\n"
        "From this folder, check the delivered files before rebuilding:\n\n"
        "```sh\npython3 verify_delivery.py\n```\n\n"
        "To rebuild, install a LaTeX distribution with `pdflatex` and Poppler's `pdfinfo`, "
        "then run:\n\n"
        "```sh\nbash build.sh\n```\n\n"
        "The build checks for exactly five pages and rejects LaTeX errors, undefined references "
        "and overfull boxes. It writes `SafeDecontam_ICCC2026_camera_ready_5p.pdf`. "
        "Rebuilt PDF bytes can depend on the installed LaTeX toolchain; the supplied checksums "
        "describe the delivered files.\n")
    manifest(source)
    manifest(package)
    manifest(compact)
    full = args.output / "SafeDecontam_ICCC2026_5p_PDF_LaTeX_Code.zip"
    code = args.output / "SafeDecontam_ICCC2026_Experiment_Source.zip"
    paper = args.output / "SafeDecontam_ICCC2026_5p_PDF_LaTeX.zip"
    zip_tree(package, full)
    zip_tree(source, code)
    zip_tree(compact, paper)
    summary = {**metadata, "assets": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                                      for p in [full, code, paper]}}
    (args.output / "PACKAGE_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
