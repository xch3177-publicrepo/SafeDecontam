# Current complete constructed benchmark

The current result is a new ten-seed run of the unchanged historical scoring implementation with its original data, feature definitions, hyperparameters, and seed list. The missing run behind 93.70% was not recovered. Neither 93.70% nor the archived 90.21% was used as a tuning target. Historical records and the earlier one-seed diagnostic remain separate.

## Refit command

Use Python 3.12 with NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, and scikit-learn 1.5.2. The supplied source directory must contain the unmodified historical `safedecontam.py` and `run_v2.py`.

Run the commands in [Current constructed refit](../../README.md#current-constructed-refit)
from the package's `reproducibility/` directory (or `experiment_source/` in the
complete ZIP). They use the bundled `runners/`, `inputs/constructed/`, and
`constructed_historical/` paths and write into a fresh `runs/` directory.

The output directory must not exist. The runner refuses to overwrite an existing run, including a partial run. Set environment variables before Python starts; they fix set-iteration order and BLAS threading without changing any modeling parameter. The original historical environment was not recorded, and nonconvex fitting may still differ on another platform. Frozen score replay is the exact way to check the reported metrics; refitting is a separate numerical reproducibility check.

The current platform, versions, environment, hashes, timing, seed list, and model settings are recorded in `run_manifest.json`. JSON `null` thresholds represent positive infinity and abstention. Undefined rates also use JSON `null`.

## Recorded evidence

- `results.json`: primary eight-scorer results at both budgets, ten-seed summaries, margin family-local calibration and transfer.
- `seeds/<seed>/frame_train.csv`, `frame_cal.csv`, `frame_test.csv`: exact source-row mapping, labels, family/group identifiers, relation and candidate source. `source_csv_row` is a zero-based data-row index excluding the header. Full texts are recovered from the canonical CSV identified by its filename and verified input SHA256.
- `seeds/<seed>/features.npz` and `features.json`: exact training, calibration, and test feature matrices with column order.
- `seeds/<seed>/scores.npz`: full calibration and test score arrays across all three families. The primary seed includes eight scorers; every seed includes MLP probability and uncompressed margin scores.
- `seeds/<seed>/*_model.npz` and matching JSON: fitted numeric coefficients, biases, scaler parameters, feature order, estimator settings, classes, and iteration count. MLP probability and margin use the same fitted model.
- `seeds/<seed>/results.json`: complete per-seed recorded metrics.
- `source/`: exact experiment implementation and wrapper snapshot.
- `SHA256SUMS.txt`: identity of the complete run payload.

The optional `--private-runtime-dir` produces a local joblib cache for related frozen-scorer probes. It is not required for verification or refitting and should be excluded from a public package. The portable release uses numeric coefficient and score arrays instead.

## Scope

These constructed results remain exploratory: configuration and threshold selection inspect calibration outcomes. The no-constraint logistic model uses similarity features plus chronology; it is not a pure similarity-only baseline. A new, documented fit does not create an independent selection certificate. The primary MMLU raw-search audit is separate.

The unsupported historical exact-text-disjoint result was not recreated. The replacement figure uses only the saved calibrated operating points, without interpolation or reconstructed historical curves.
