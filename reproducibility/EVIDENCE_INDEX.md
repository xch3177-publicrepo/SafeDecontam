# Evidence index for the reviewed reproduction package

The current manuscript starts from commit `7e1c30a` and corrects the constructed
and auxiliary results using the separately recorded 2026-09-14 UTC reruns.
Historical results are preserved under their own names and are not presented as
new measurements. The main MMLU audit is unchanged.

| Claim | Evidence | Verification scope |
| --- | --- | --- |
| MMLU silver locked test | `raw_search/results/main_audit/`; `verification/report.json` | All 154 frozen decisions, 62/68 contamination removals, 5/86 clean removals, AUROC and saved-order bootstrap independently replayed. |
| MMLU validation and calibration | Same historical `results.json` | Validation recall/count arithmetic and 106-group binomial bounds checked. The historical validation score rows and calibration score array are unavailable, so the validation AUROCs and score-derived threshold were not re-derived. |
| Human MMLU audit | `verification/author_confirmed_aggregates.json` | Author-confirmed 82/71/1 labels, 67/82 removal and 0/71 clean removal retained. Exact intervals recomputed conditional on the counts. Human AUROC, kappa, independence and full-page membership cannot be replayed without the original row-level records. |
| Current constructed Table III, Figure 1 and ten splits | `evidence/constructed_current/` | New fixed-configuration run; exact source-row membership, features, fitted numerical weights, calibration/test scores and all ten seeds included. Independent verifier passed 5,588 checks. Main margin result: 90.34% CR, 0.68% CDR, 2.01% GCDR; ten-seed recall 86.67 +/- 7.50%. |
| Constructed transfer and error breakdown | Same current run | Recomputed from the current margin scorer. Transfer uses source-family local thresholds on target-family test rows. Current 210 misses and 17 false removals match the exported rows. |
| ARC, HellaSwag, C-Eval | `evidence/external_reruns/` | New independent CPU runs of the unchanged raw-search implementation reproduce all three paper rows at reported precision. Full split scores, clean-calibration arrays and 27-dimensional item features included. |
| Current web pilot and cross-lingual probe | `evidence/auxiliary_reruns/` | New runs with the same frozen seed-42579 model as the current constructed study. Numerical inference and calibration independently checked; cache-free execution regenerates byte-identical outputs. |
| Historical constructed outputs | `evidence/constructed_historical/` | Original `v2_results.json` reports 90.21% margin recall. It did not support the earlier manuscript's 93.70%. Both differ from the new fixed-environment run and are kept separate. |
| Earlier exact-text-disjoint experiment | Earlier manuscript only | Corresponding final score evidence was not found. Its 13,905-pair performance claim has been removed from the current manuscript. |
| Calibration equation reference | `calibration_reference/` | Separate synthetic examples and unit tests; not an empirical rerun. |

## Current auxiliary values

The web pilot removes 98/108 answer-bearing records, 9/11 question-only records,
and 0/119 cross-item controls. The frozen cross-lingual model removes 0/72 exact
copies, 8/72 English derivatives and 2/72 Chinese derivatives: 10/216 = 4.63%
overall. Its exact-copy scores remain below the local cutoff; there is no
independent exact-match override.

The historical pilot recorded 8/11 question-only removals and one removed
control. Historical cross-lingual output contains distinct frozen and retrained
models; the earlier 6.48% belongs to the retrained result. The current paper
instead cites the fully exported new frozen-model run. Matching one old
percentage is not evidence that two runs have the same protocol.

## Statistical and computational boundaries

The constructed implementation inspects calibration outcomes when choosing
configuration and feasible thresholds, including remove-labeled calibration
rows. It remains exploratory. The raw-search audit has its separate
training/validation/clean-only-calibration/locked-test protocol. Re-running the
constructed code does not grant it the raw-search population-risk guarantee.

The population-risk statement retains independent within-family draws and a
fixed scoring/evidence protocol. A numeric check cannot establish those sampling
assumptions or certify transfer to another distribution.

The new fit records Python and numerical-library versions, hash seed, BLAS
thread settings, source/input hashes and all fixed split seeds. Nonconvex MLP
refitting can differ on another platform. Replaying the distributed numerical
weights and frozen scores is the exact check of the reported outcomes; a fresh
fit is a separate computational reproducibility check.

Raw web responses and model weights are obtained from their pinned upstream
sources. Full search text is not duplicated in the compact external rerun
payload; its digest is retained and its feature/score evidence is included.
