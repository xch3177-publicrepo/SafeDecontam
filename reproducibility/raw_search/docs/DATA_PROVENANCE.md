# Data provenance and evidence levels

## Public source

- Li et al., *An Open-Source Data Contamination Report for Large Language
  Models*, Findings of EMNLP 2024.
- Repository: `https://github.com/liyucheng09/Contamination_Detector`
- Pinned commit: `ffe3c249309da9a7ad7f42e9145e1f8e4d36c77e`.
- `benchmarks.zip` SHA-256:
  `689f98bd4edb9a7db75cdcd03d0b453ab8cdaf7f2334e46d3e4c6b42e50d335c`.
- `bing_search.zip` SHA-256:
  `89bc89d70a9a4b0855c3da15598ac74dd2308fd36fa31d25a9a7d1d7a33bb6b5`.

## Experimental roles

- Training fits TF-IDF representations and supervised scorers.
- Validation selects the scorer.
- Untouched clean calibration sets the risk-controlled threshold.
- The locked test is evaluated after scorer and threshold selection.
- URLs appearing across splits are removed from every affected evidence bundle
  before representation fitting.

## Evidence and labels

- Released labels are search-overlap silver annotations.
- Evidence consists of saved Bing result titles and snippets rather than a
  contemporaneous archive of every full page.
- The principal unit is an MMLU item and its retrieved evidence bundle.
- The human audit concerns the locked test only and does not replace the
  silver-label calibration distribution.

## Scope limits

- Retrieval recall outside the released Bing responses is not measured.
- The audit does not establish that any page entered a model's training set.
- No downstream model is retrained, so contamination recall does not directly
  estimate benchmark-score inflation.
- Exchangeability between calibration groups and a future deployment
  population is an explicit assumption, not an empirical fact established by
  the package.
