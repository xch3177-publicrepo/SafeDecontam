# Independent MMLU audit verifier

This verifier recomputes statistics from the released frozen predictions. It
does not import the experiment implementation, fit a scorer, or claim that
aggregate arithmetic independently validates human labels.

```bash
python3 -m pip install -r ../requirements-cpu.txt
python3 verify_audit.py \
  --audit-dir /path/to/raw_search/results/main_audit \
  --output audit_verification_report.json
```

Python 3.12 or newer is recommended. The two required audit inputs are
`results.json` and `locked_test_predictions.jsonl`. `--human-aggregates` can
select another explicit aggregate input file; the default is the adjacent
`author_confirmed_aggregates.json`. Reports record SHA-256 hashes of all inputs.

## What is checked

- **Row-level replay:** the frozen threshold rule, 62/68 contaminated and 5/86
  clean removals under silver labels, all 67 removals, category recalls,
  AUROC by direct positive-negative pair comparison, and 2,000 unstratified
  source-group bootstrap replicates (NumPy seed 20260716, linear quantiles).
  Each source group has exactly one row; input order is preserved.
- **Aggregate arithmetic:** validation recall from positive/false-negative
  counts for all five scorers, scorer selection, split/category totals,
  exact binomial bounds conditional on 106 reported clean groups, the maximum
  allowable exceedance count, and minimum reservoir sizes 59/80 for one/three
  families. Binomial probabilities are computed by direct finite sums and
  bisection, independently of the experiment's SciPy beta-quantile code.
- **Author-confirmed aggregate calculations:** human recall 67/82, zero clean
  deletions among 71, precision 67/67, and full-page counts 15/20 and 0/30.
  The JSON reports exact two-sided 95% Clopper-Pearson recall/precision
  intervals and one-sided 95% clean-deletion upper bounds.

## Limits and result status

Validation score rows, calibration scores, human annotations, and annotator
records are absent from these inputs. Their AUROCs, score-derived thresholds,
sampling provenance, human-label accuracy, and kappa are **not replayed**.
Checking the recorded threshold against predictions does not reconstruct its
calibration order statistic. The population-risk guarantee's sampling
assumptions cannot be established by checking numeric outputs.

`PASS_WITH_LIMITATIONS` (exit 0) means every available arithmetic/row replay
check passed. It does not mean every paper claim was independently reproduced.
`FAIL` (exit 1) identifies numeric mismatches by name; `ERROR` (exit 1)
identifies missing, malformed, or invalid inputs. The report lists unavailable
evidence separately. Non-MMLU experiments and the historical constructed-pair
results are outside this verifier's scope.

`report.json` is a recorded execution against the bundled reference inputs.
Re-running changes its generation timestamp; input hashes and recomputed
statistics should match. The archived bootstrap depends on the input row
order, generator seed, replicate count, and NumPy RNG/quantile behavior.

## Optional item-level human-label replay

`verify_human_labels.py` derives every count from supplied rows; it has no
hard-coded annotation totals or annotator identities. No human labels are
included here. Without `--final-labels`, it writes an explicit `UNAVAILABLE`
report (exit 2) and does not substitute the aggregate file.

```bash
python3 verify_human_labels.py \
  --predictions /path/to/raw_search/results/main_audit/locked_test_predictions.jsonl \
  --final-labels /path/to/authorized_final_labels.csv \
  --output human_label_verification_report.json
```

The final CSV must have `item_id,human_label,full_page_accessible` columns.
Labels must be `clean`, `input`, `input_and_label`, or `uncertain`; access must
be `yes`, `no`, or `unknown`. Column-name overrides are available through
`--id-column`, `--label-column`, and `--access-column`. Every final-label ID
must match one prediction ID exactly. Missing IDs, extras, duplicates,
invalid labels, or inconsistent frozen decisions cause `ERROR` (exit 1).

With `--annotator-a A.csv --annotator-b B.csv`, the program also recomputes
four-class and binary-both-certain agreement/kappa. Annotator CSVs require
`item_id,item_label` (override with `--annotator-label-column`). Their ID sets
must also match exactly. A valid run (exit 0) reports counts, exact intervals,
AUROC, access subgroups, and agreement from the actual supplied records.
Undefined metrics use JSON `null` with an explanation. No item-level content
or annotator identities are included in the resulting report.

The older `analyze_annotation_pair.py` and `finalize_human_adjudication.py`
contain historical fixed denominator/provenance prose. Use this standalone
verifier for fresh labels rather than treating that historical prose as new
results. This verifier evaluates an already finalized label file; it does
not perform or certify adjudication.
