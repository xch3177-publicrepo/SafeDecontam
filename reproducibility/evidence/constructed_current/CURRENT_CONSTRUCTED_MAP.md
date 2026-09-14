# Current constructed-run manuscript map

All figures below come from the complete current ten-seed run with fixed historical settings. The historical 93.70% and 90.21% numbers are not used as targets.

| Scorer | CR (%) | CDR (%) | GCDR (%) |
| --- | ---: | ---: | ---: |
| 4-gram containment | 25.15 | 0.08 | 0.37 |
| Shingle Jaccard | 5.47 | 0.40 | 1.65 |
| Normalized BM25 | 3.72 | 0.56 | 2.56 |
| Similarity fusion | 0.92 | 0.40 | 1.83 |
| No-constraint logistic | 41.66 | 0.64 | 2.56 |
| All-feature logistic | 26.53 | 0.12 | 0.55 |
| MLP probability | 0.00 | 0.00 | 0.00 |
| MLP margin | 90.34 | 0.68 | 2.01 |

## Main text values

- Primary margin: 90.34% CR, 0.68% CDR, 2.01% GCDR.
- Ten splits: 86.67 ± 7.50% recall (population SD), range 69.99–95.03%.
- Probability has 4/10 zero-recall splits; margin has 0/10.
- Calibration worst-family upper bound: 4.67%; test descriptive upper bound: 4.61%.
- Shared global threshold: arithmetic recall 92.21%, MCQ 85.43%; separate MATH local threshold 68.42%.
- Budget 5%→10%: threshold 56.40→10.67; at 10% CR 97.93%, CDR 1.00%, GCDR 3.30%.
- False negatives 210 = 98 released rephrasings + 63 other transformations + 30 answer fragments + 19 exact copies. Consult JSON for exact-family breakdown.
- False removals 17 = 14 GSM-Symbolic + 2 answer-changing GSM-Plus + 1 same-subject MMLU.
- Margin threshold transfer: arithmetic→MATH GCDR 28.95%; MCQ→MATH 11.84%; MATH→arithmetic recall loss 30.99 points and MATH→MCQ 35.34 points.

## Scope

This remains an exploratory constructed study because configuration and threshold selection inspect calibration outcomes. The new run supplies exact score, feature, split, and coefficient evidence; it does not turn exploratory selection into an independent certificate.

The earlier 13,905-pair exact-text-disjoint result has no recoverable score evidence and is not recreated here. Do not retain that quantitative claim.

The current plot shows only saved calibrated operating points. It does not imply a reconstructed historical threshold curve.
