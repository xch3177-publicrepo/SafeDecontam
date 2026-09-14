# SafeDecontam reproducibility package

Code and compact audit outputs for **SafeDecontam: Risk-Calibrated Constraint
Verification for Benchmark Decontamination** (ICCC 2026 submission).

This package reproduces the paper's raw-search audit from the public artifacts
released with Li et al. (Findings of EMNLP 2024). It intentionally excludes
virtual environments, model weights, upstream repository history, search-data
duplicates, temporary files, and manuscript build products.

## Package contents

- `experiments/run_raw_bing_item_audit.py`: principal 40/20/20/20
  train/validation/calibration/locked-test audit.
- `experiments/run_web_evidence_experiment.py`: shared features, calibration,
  bootstrap utilities, and the exploratory selected-evidence audit.
- `experiments/build_human_annotation_pack.py`: constructs the blinded human
  annotation sheet from locked-test items.
- `experiments/analyze_annotation_pair.py`: checks two independent annotation
  files and computes agreement statistics.
- `experiments/finalize_human_adjudication.py`: combines agreements with a
  separate adjudication extract and evaluates frozen predictions.
- `scripts/prepare_public_data.py`: downloads, checks, and lays out the pinned
  public source data expected by the experiment code.
- `results/main_audit/`: the principal machine-readable result and frozen
  locked-test predictions reported in the paper.
- `docs/`: data provenance, evidence limitations, and annotation protocol.

## Quick start

Python 3.12.13 was used for the reported run.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/prepare_public_data.py
python3 experiments/run_raw_bing_item_audit.py
```

The dense run downloads `BAAI/bge-small-en-v1.5` through
SentenceTransformers unless it is already cached. For a lightweight external
audit without the dense feature:

```bash
python3 experiments/run_raw_bing_item_audit.py --dataset ARC --disable-dense
python3 experiments/run_raw_bing_item_audit.py --dataset hellaswag --disable-dense
python3 experiments/run_raw_bing_item_audit.py --dataset ceval --disable-dense
```

Outputs are written below `output/experiments/`. Existing reference results in
`results/` are never overwritten. See [REPRODUCE.md](REPRODUCE.md) for the
expected values and verification procedure.

## Scientific scope

The principal audit uses released search-overlap **silver labels** and Bing
title/snippet bundles. It validates the frozen split/calibration protocol at the
benchmark-item evidence level; it is not an end-to-end measurement of web
retrieval recall, model training history, or downstream accuracy inflation.
The independent human audit is a test-set audit and does not retroactively
replace the silver-label calibration reservoir. The author confirms the final human audit; its item-level source files and
adjudication records are not included in this delivery. Their absence limits
package-level replay, not the author-confirmed aggregate counts. Use
`../verification/verify_human_labels.py` if actual final label files are supplied.

## Data and licensing

The source data are not redistributed in this compact package. The preparation
script fetches the upstream release and verifies the two archive SHA-256 hashes.
Upstream artifacts remain governed by their own terms. The existing repository
MIT license covers the authors' SafeDecontam code; it does not relicense upstream
benchmark or search data. See `../THIRD_PARTY_NOTICE.md`.

## Citation

See `CITATION.cff`. Update the DOI/proceedings fields after publication.
