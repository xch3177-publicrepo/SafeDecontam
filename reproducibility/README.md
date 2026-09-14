# SafeDecontam reproduction and evidence package

This package separates archived experiment outputs, independent replays, new
training runs, and author-confirmed human aggregates. Read the
[evidence index](EVIDENCE_INDEX.md) before interpreting a successful checksum
check as scientific reproduction.

## Setup

Run the commands below from this directory: `reproducibility/` in the repository,
or `experiment_source/` in the complete ZIP. Python 3.12 is recommended for the
checked CPU environment.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python verify_delivery.py
python verify_all.py --output runs/verification-1
```

`verify_delivery.py` checks file hashes. `verify_all.py` checks the saved MMLU,
external, constructed, and auxiliary evidence and the separate calibration unit
tests without fitting models. Choose a new output directory for each invocation.

The full MMLU training path additionally needs the dependencies in
`raw_search/requirements.txt`, including SentenceTransformers and its frozen
embedding model. The lightweight external audits and saved-output checks do not.

## Contents

- `raw_search/`: preserved raw-search feature, fitting, calibration, annotation,
  and data-preparation implementations, with historical MMLU results and locked
  predictions. The audit implementation was already present in commit `7e1c30a`;
  the current delivery adds an external replay-export wrapper.
- `constructed_historical/`: preserved constructed-pair implementation. Its
  historical entry points retain their original execution layout; use the
  portable runners below with this directory as `--source-dir`.
- `inputs/constructed/`: the supplied pair CSVs used by that implementation.
- `evidence/constructed_historical/`: unchanged historical JSON outputs.
- `evidence/constructed_current/`: the complete new ten-seed run, fitted numerical
  model parameters, split membership, features, scores, and plotted points.
- `evidence/external_reruns/`: independently regenerated ARC, HellaSwag, and
  C-Eval outputs, predictions, split membership, and score/feature evidence.
- `evidence/auxiliary_reruns/`: the new web pilot and frozen cross-lingual probe
  using the current primary constructed model.
- `verification/`: portable checks of saved outputs and confidence bounds.
- `calibration_reference/`: separate equation implementation and synthetic tests.
  Synthetic output is not empirical evidence for the paper.

## Current constructed refit

The current table and figure use a new, fixed-configuration run with exported
split membership, features, model parameters, and scores. They replace earlier
manuscript numbers whose corresponding final run could not be found. The
supplied historical output is preserved separately and is not relabeled as a
reproduction of the new run. No setting was tuned to recover a previous recall
value. The constructed study remains exploratory because configuration and
threshold selection inspect calibration outcomes.

The following command refits all ten predefined seeds. The output directory
must not exist. Environment variables must be set before Python starts to fix
set iteration and BLAS threading.

```sh
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python runners/run_current_constructed.py \
  --input-dir inputs/constructed \
  --source-dir constructed_historical \
  --output runs/new-constructed

python verification/verify_current_constructed.py runs/new-constructed \
  --input-dir inputs/constructed \
  --output runs/new-constructed-verification.json

python runners/plot_current_operating_points.py \
  --run-dir runs/new-constructed \
  --output runs/new-constructed-figure.png
```

The plotting command also creates matching PDF and JSON files. Recorded scores
provide exact metric replay; a new fit is a separate reproducibility check and
may differ across platforms or numerical libraries. See the
[current run manifest](evidence/constructed_current/run_manifest.json) for the
checked environment, input/source hashes, settings, and seed list.

## Auxiliary probe refit

This command rebuilds the fixed primary model from the supplied constructed
training split, then applies it to the archived web-pilot and cross-lingual
pairs. It requires no private model cache and does not fit on auxiliary examples.
Use a new output directory.

```sh
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python runners/run_auxiliary_probes.py \
  --input-dir inputs/constructed \
  --source-dir constructed_historical \
  --output runs/new-auxiliary

python verification/verify_auxiliary_probes.py runs/new-auxiliary
```

The probe retains the historical exploratory threshold-selection procedure.
Its outputs are distinct from the raw-search audit and its clean-only
calibration protocol. The [auxiliary evidence README](evidence/auxiliary_reruns/README.md)
describes the score exports, exact-copy limitation, and verification record.

## External raw-search refit and evidence export

Prepare the pinned upstream data first. This requires Git and internet access;
the preparation script checks the downloaded archive hashes. Run each audit
into a new output directory so the historical evidence is preserved.

```sh
python raw_search/scripts/prepare_public_data.py

PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python raw_search/experiments/run_external_with_replay.py \
  --dataset ARC --disable-dense --output runs/external-ARC

PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python raw_search/experiments/run_external_with_replay.py \
  --dataset hellaswag --disable-dense --output runs/external-hellaswag

PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python raw_search/experiments/run_external_with_replay.py \
  --dataset ceval --disable-dense --output runs/external-ceval

python verification/verify_external_replay.py \
  runs/external-ARC runs/external-hellaswag runs/external-ceval
```

The wrapper observes the unchanged audit after it finishes. Alongside the normal
results and locked predictions, it exports all split scores, clean-calibration
scores, 27-dimensional features, and provenance. It does not replace the scoring,
splitting, or calibration functions. These runs use the lexical and constraint
adapter with dense embeddings disabled, as recorded in the supplied reruns.

## Full MMLU training

The preserved training implementation is available for a new run. Install its
additional dependencies, prepare the public inputs as above, and select a fresh
output directory:

```sh
python -m pip install -r raw_search/requirements.txt
python raw_search/experiments/run_raw_bing_item_audit.py \
  --dataset mmlu --output runs/new-mmlu
```

This path downloads the frozen embedding model if it is not already cached.
The current delivery verifies the available locked MMLU predictions and aggregate
calibration arithmetic; it does not claim a new full MMLU training replay or
reconstruction of the missing original validation/calibration score rows.

## Human audit scope

The author confirmed 82 contaminated, 71 clean, and one uncertain MMLU test item.
The frozen detector removes 67 of the contaminated items and none of the clean
items. The package can recompute intervals conditional on these aggregate counts.
Final human labels, the original A/B passes, adjudication records, and full-page
access records are not supplied, so the package cannot replay human AUROC,
agreement/kappa, or subgroup membership from item-level evidence. Those remain
author-confirmed reported results. A separate optional verifier accepts actual
human tables; it does not generate or fill missing labels.

Historical annotation scripts include narrative strings specific to their
original workflow. Use the current verification tools for fresh label tables.
Neither those strings nor a synthetic example establish that annotation occurred.

See [third-party notices](THIRD_PARTY_NOTICE.md) for the distinction between code
licensing and upstream benchmark/search-data terms.
