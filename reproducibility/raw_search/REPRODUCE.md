# Reproduction and result contract

Run all commands from the package root. The experiment code resolves inputs and
outputs relative to that root; it contains no personal absolute paths.

## 1. Environment

The paper run used Python 3.12.13. Install the exact Python dependencies from
`requirements.txt`. CPU and GPU floating-point implementations can produce
minor score differences; compare counts and metrics with reasonable numerical
tolerance rather than requiring bit-identical model probabilities.

## 2. Public inputs

Run:

```bash
python3 scripts/prepare_public_data.py
```

This creates the two paths consumed by the audit:

- `external/Contamination_Detector/reports/*_annotations.json`
- `data/raw/contamination_detector/bing_search/bing_search/<dataset>/*.json`

The script pins the upstream Git commit and validates the downloaded archives
before extraction. No labels are generated or changed by this step.

## 3. Principal MMLU audit

```bash
python3 experiments/run_raw_bing_item_audit.py
```

Expected contract in `output/experiments/raw_bing_item_audit/results.json`:

- 783 usable items; split counts 308/159/162/154.
- selected scorer: logistic regression.
- 106 clean calibration groups; one permitted exceedance.
- threshold: 0.6184764791; one-sided upper bound: 0.0439708206.
- locked silver test: 62/68 detected, 5/86 clean items removed.
- AUROC diagnostic: 0.9517783858.

The preserved reference object is `results/main_audit/results.json`.

## 4. External audits

```bash
python3 experiments/run_raw_bing_item_audit.py --dataset ARC --disable-dense
python3 experiments/run_raw_bing_item_audit.py --dataset hellaswag --disable-dense
python3 experiments/run_raw_bing_item_audit.py --dataset ceval --disable-dense
```

These are boundary/sensitivity audits, not substitutes for the principal MMLU
certificate.

## 5. Human-label workflow

Generate a blinded sheet only after the frozen test predictions exist:

```bash
python3 experiments/build_human_annotation_pack.py
```

Compare two independently completed CSV files:

```bash
python3 experiments/analyze_annotation_pair.py \
  --annotator-a annotations/annotation_completed_A01.csv \
  --annotator-b annotations/annotation_completed_A02.csv
```

The author-confirmed final human-label tables and original annotation records
are not included in this delivery. The commands above document the historical
workflow, whose narrative strings contain fixed original-study counts. For
fresh label files, use `../verification/verify_human_labels.py`; it derives
counts, agreement and intervals from actual rows and refuses missing or
mismatched IDs. It does not perform or certify adjudication.

## 6. Integrity verification

```bash
python3 tools/verify_package.py
```

`CHECKSUMS_SHA256.txt` covers every distributed file except itself and the ZIP
container. A clean verification must report no missing, extra, or mismatched
files.
