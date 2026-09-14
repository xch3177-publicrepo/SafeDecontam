# Independent constructed-result replay

Run from any working directory:

```sh
python /path/to/verification/verify_current_constructed.py \
  /path/to/evidence/constructed_current \
  --input-dir /path/to/inputs/constructed \
  --output constructed_verification_report.json
```

Use the delivery's pinned `requirements-cpu.txt`. This verifier itself imports
only NumPy and standard-library code, plus the adjacent `verify_audit.py`
binomial-CDF implementation. It never imports the experiment's evaluation or
threshold-selection functions.

The report checks all eight primary scorers at 5% and 10%, the fixed ten-seed
MLP probability/margin runs and summary statistics, MATH local results, all
primary error categories, local family thresholds, the complete margin-score
transfer matrix, and the primary margin bootstrap interval. It also checks
source CSV hashes, exact source-row coverage and metadata, group-disjoint
split ownership reconstructed from the documented seed procedure, model and
feature dimensions, and saved-model inference against each score array.

Thresholds are independently selected over the documented candidate set
(unique calibration pair scores and positive infinity), with inclusive
`score >= threshold` decisions. Feasible family exceedance counts are found
using the binomial CDF at the risk budget; the selected threshold maximizes
calibration positive recall, then minimizes group damage, then maximizes the
threshold. JSON `null` thresholds mean positive infinity (abstention). The
bootstrap resamples positive source groups with replacement, preserving the
archived `RandomState(0)`, 2,000-replicate procedure.

`PASS` (exit 0) means all recorded values covered by these checks agree with
the replay. `FAIL` (exit 1) lists named mismatches; `ERROR` (exit 1) reports
missing/malformed inputs. Every consumed evidence file is hashed in the
report. Omitting `--input-dir` skips source CSV hash and row correspondence
checks and is explicitly disclosed in the report.

This is a replay of saved output and fitted-model inference. It does not
retrain the optimizer, regenerate raw-text features, establish label validity,
or prove sampling assumptions. The inherited exploratory calibration protocol
retains its existing inferential limitations even when numerical replay passes.
