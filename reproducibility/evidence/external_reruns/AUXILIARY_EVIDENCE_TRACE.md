# Auxiliary evidence trace and external-audit reruns

Completed on 2026-09-14 UTC (2026-09-13 America/Los_Angeles). This work did not
edit a manuscript, change a Git repository, or fabricate missing historical
evidence. The user confirmed the separate final human audit (82 contaminated,
71 clean, one uncertain); that confirmation is accepted.

## External Table IV rows: independently reproduced

The compact code archive did not contain external-audit outputs. The supplied
Hang_data archive contains upstream search-overlap reports and annotations,
but not all raw search bundles needed for these audits. The pinned public
`bing_search.zip` was downloaded from Li et al.'s release v0.1.1 and verified
against SHA-256 `89bc89d70a9a4b0855c3da15598ac74dd2308fd36fa31d25a9a7d1d7a33bb6b5`.
The supplied ARC, HellaSwag and C-Eval annotation files were each byte-checked
against upstream commit `ffe3c249309da9a7ad7f42e9145e1f8e4d36c77e`.

The unchanged released script reproduced all three Table IV rows at manuscript
precision using Python 3.12, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2,
and dense features disabled:

| Dataset | Selected scorer | Removed positive / all positive | Removed clean / all clean | AUROC |
|---|---|---:|---:|---:|
| ARC | Logistic | 7/17 (41.18%) | 5/223 (2.24%) | 0.9053020311 |
| HellaSwag | Word TF-IDF | 14/22 (63.64%) | 0/155 (0.00%) | 0.9260997067 |
| C-Eval | Logistic | 60/129 (46.51%) | 3/144 (2.08%) | 0.9273255814 |

The completed runs are **new independent reruns**, not recovered original
outputs. `reruns/` retains the first direct executions. `replay_reruns/`
contains observation-only wrapper runs which additionally export every split
ID and all scorer outputs, the selected scorer's complete clean-calibration
array, and the 27-dimensional raw item features. The wrapper records the
original audit main function's return-frame locals; original source files
remain byte-identical to the latest supplied package. The direct-run and
wrapper-run `results.json`, `locked_test_predictions.jsonl`, and
`dataset.jsonl` are byte-identical for each dataset.

`verify_external_replay.py` independently recomputed scorer selection,
validation metrics, calibration feasibility/threshold, test counts and AUROC
from saved score arrays and matched the predictions. All checks passed; see
`external_replay_verification.json`.

Recommended portable package contents per dataset: `results.json`,
`locked_test_predictions.jsonl`, `all_split_scores.jsonl`,
`calibration_clean_scores.json`, `raw_item_features.npz`, and
`rerun_provenance.json`, plus the wrapper and independent verifier. The full
`dataset.jsonl` (title/snippet bundles, roughly 8.5/7.6/28 MB) is retained in
scratch; it can instead be regenerated from the hash-pinned public inputs to
keep redistributed third-party text out of a compact package.

## Historical web pilot: a numerical discrepancy is confirmed

`Hang_data/ICCC/middle_steps/r2_wild.json` and `r2_wild_pairs.csv` agree:
98/108 answer-bearing removed (90.74%), 8/11 question-only removed (72.73%),
and 1/119 cross-item controls removed (0.84%). The latest manuscript's 81.82%
question-only value is not supported by these archived outputs; it would
require 9/11 rather than the stored 8/11. No final-run artifact supporting
9/11 was found in the supplied packages or local SafeDecontam project.

The archived pilot code uses fixed historical MCQ-local thresholds. If the
manuscript uses these historical outputs, change only 81.82% to 72.73% and
identify the archived pilot configuration. Do not present this archival
count check as a new full-model reproduction.

## Cross-lingual probe: protocol and counts need separation

The historical `v2_results.json[xling_probe]` contains two distinct results:

| Condition | Test groups | Exact | English rephrase | Chinese rephrase | Overall positive removal |
|---|---:|---:|---:|---:|---:|
| Frozen three-family MLP + local threshold | 72 | 8/72 | 4/72 | 4/72 | 16/216 = 7.41% |
| MLP retrained including the xling training partition | 72 | 0/72 | 8/72 | 6/72 | 14/216 = 6.48% |

The current manuscript's percentages 0.0%, 11.1%, 8.3%, overall 6.48% match
the **retrained** archival entry, not the frozen English-trained model.
The review's denominator 36 is not supported by this archive: it has 240
total source groups, 72 test groups, 432 test pairs, and 216 positive test
pairs. Percentages alone cannot distinguish 0/36,4/36,3/36 from the archived
0/72,8/72,6/72. The archived `run_xling.py` executes the frozen condition;
the additional retrained entry exists in JSON, but its generating code was
not recovered in this trace. Retaining the frozen-condition claim therefore
requires using its own archived results (7.41%, with 11.1%,5.6%,5.6% by
positive relation), unless a matching later run is found.

The `C0_exact` candidates are genuinely exact benchmark text plus an answer
line: all 240 archived C0 rows start with the entire `benchmark_text`.
The pipeline scores them with the learned model and applies the same
calibrated threshold; there is no separate exact-match deletion override.
Thus exact-text false negatives are possible. A minimal clarification, if
a supported zero-exact result is retained, is:

> under the learned scorer, which has no exact-match override

Do not imply those are mistranslated copies or change labels merely to
explain the surprising result. Historical notes attribute the failure to
local hard negatives pushing the threshold above exact-copy scores; that
causal explanation is not independently verified by archived score arrays.

## Final human audit: accepted, not independently replayable from these inputs

The supplied packages and local project contain the blank 154-item sheet,
protocol, tools, and an explicitly separate earlier LLM-assisted annotation
pass (80 contaminated, 73 clean, one uncertain; snippet-only). These must
not be used as the source of the user-confirmed final human 82/71/1 audit.
Completed independent human A/B labels, final adjudicated item labels and
full-page membership records were not found. The package can honestly
preserve the confirmed manuscript values while stating that these private
item-level human records are not included. The aggregate 67/82 and 0/71
bounds can be recomputed from confirmed counts, but agreement and AUROC
require their respective underlying label tables for independent replay.

## Execution notes and limitations

- Sources were safely extracted only under this scratch directory, rejecting
  absolute/traversal/symlink members and excluding `_to_delete` duplicates.
- Python urllib failed TLS certificate validation against the pinned raw URL.
  Verification succeeded with curl using normal TLS validation; no TLS
  verification bypass was used.
- The environment did not permit `ps`; this did not affect experiment runs.
- Package-pinned scientific dependencies were installed in a local venv.
  Historical original runtime/model weights were not available.
- Full first and replay stdout/stderr logs are retained as `ARC.log`,
  `hellaswag.log`, `ceval.log`, and their `_replay.log` variants.
- Source input hashes are in `INPUT_MANIFEST.json`; each replay output has
  its runtime, code hashes, annotation hash and dataset hash in provenance.
