# External-review corrections and evidence

The revised five-page manuscript preserves the title, authors, abstract, equations, references, and principal MMLU audit from commit `7e1c30a`. Changes are confined to evidence-backed constructed/auxiliary results, their scope and figure, and the baseline label that previously omitted chronology.

## Constructed results

The run behind the earlier 93.70% recall was not found in the supplied archives or repository. The supplied historical JSON instead reports 90.21%, and an initial new fit did not reproduce that historical MLP exactly. Consequently the current manuscript uses a new complete run of the unchanged historical implementation, fixed original hyperparameters and ten predefined split seeds. Python hash seed 0, single-thread BLAS, library versions, source hashes and input hashes are recorded. No settings were tuned to recover an old percentage.

The current primary MLP-margin result is **90.34% CR, 0.68% CDR, 2.01% GCDR**. Ten-split recall is **86.67 ± 7.50%** (population SD; 69.99–95.03%). All eight baselines, budget sensitivity, error breakdowns, family-local thresholds, margin transfer and Figure 1 now use this same run. The probability scorer retains every item in the primary split because its saturated score scale yields abstention; four of ten probability-score splits have zero recall. The margin scorer does not abstain in these ten runs.

The package includes exact split membership, source-row mappings, feature arrays, fitted numerical weights/scalers and complete calibration/test score arrays. An independent implementation passed 5,588 checks, including source mapping, inference, thresholds, bootstrap, error categories and ten-seed summaries; a changed score was correctly rejected. The unsupported 13,905-pair exact-text-disjoint performance claim was removed. Constructed configuration and threshold selection inspect calibration outcomes, so this study remains explicitly exploratory.

## Other experiment evidence

New independent ARC, HellaSwag and C-Eval runs reproduce every published Table IV row at the reported precision. Their scores, 27-dimensional features, clean-calibration arrays, predictions and provenance are included.

The web pilot and frozen cross-lingual probe were regenerated with the same fixed primary model. Pilot removal is 98/108, 9/11, and 0/119. Cross-lingual removal is 0/72 exact copies, 8/72 English derivatives, and 2/72 Chinese derivatives, or 4.63% overall. The exact copies contain the original text, but their maximum margin 231.9291 is below the local threshold 316.9714. The manuscript adds the short explanation that the learned scorer has no exact-match override. Standalone execution without a private cache regenerates byte-identical auxiliary outputs.

## Principal audit and human labels

The MMLU silver locked test remains 62/68 contaminated removals and 5/86 clean removals. Its frozen predictions independently reproduce the confusion counts, AUROC and bootstrap intervals. Calibration-count arithmetic and the minimum reservoir counts were checked independently. The original MMLU validation and calibration score rows remain unavailable, so those score-derived quantities are not claimed as new replays.

The author-confirmed independent human audit 82/71/1 remains unchanged, including 67/82 recall and 0/71 clean removal. Its confidence bounds are recomputed conditional on those counts. The package does not contain the final human labels, A/B tables or adjudication/access records, so it does not claim to replay human AUROC, kappa or subgroup membership. No earlier LLM labels are substituted. The optional human verifier requires actual tables and derives its output dynamically.

## Acknowledgment and delivery

The manuscript already contains only the ChatGPT language-editing/typesetting disclosure and author responsibility statement. It has no funding acknowledgment. The external review checklist's request for confirmation by funders is not retained as a manuscript or release requirement.

The independence assumption in the population-risk claim is preserved. The final PDF remains five pages; the new figure is plotted from saved operating points, not an inferred historical curve. Earlier artifacts and failed checks remain separate historical records.
