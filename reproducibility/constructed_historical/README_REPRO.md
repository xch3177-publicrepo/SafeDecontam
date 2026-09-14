# Historical source: preserved implementation

The current paper uses the separately dated fixed-configuration run described in
`../EVIDENCE_INDEX.md`. This folder preserves the earlier implementation; the
portable current runner and saved scores are shipped separately. Run into a new
output directory and do not overwrite `../evidence/constructed_historical/`.

# Historical constructed-data experiments

This English guide accompanies the preserved historical scripts. The original Chinese README remains unchanged inside `archive/user_inputs/hang_data_20260912.zip` at `Hang_data/ICCC/middle_steps/code/README_REPRO.md`. Only this working-copy guide has been rewritten; the supplied experiment scripts are unchanged.

## Original execution layout

The historical code expects `middle_steps/code/` beside a separate `real_data_set/` tree. Its `reproduce.sh` uses that original layout and writes generated outputs. The convenient repository copies are provided for inspection, not as a claim that this historical shell script runs directly from the new layout.

To investigate a historical pipeline run, extract the complete original ZIP into a new ignored run directory and inspect its paths and dependency requirements before execution. Do not run a writer against preserved evidence. The portable root-level `tools/audit_constructed_archive.py` verifies the supplied CSV counts and regenerates historical split assignments without training.

## Script inventory

| Script | Historical purpose |
| --- | --- |
| `build_pairs_mcq.py` | Build first-version MCQ pairs. |
| `build_pairs_v2.py` | Build arithmetic, MCQ, and MATH v2 pairs, including benchmark variants and MATH matching. |
| `run_v2.py` | Main historical baselines, ablations, transfer probes, GSM1k evaluation, and ten seeds. |
| `run_margin.py` | Historical MLP-margin calibration and ten-seed analysis. |
| `run_r2_wild.py` | Early 238-pair web-evidence pilot; this is not the later 783-item raw-search audit. |
| `make_figures.py` | Render figures from historical result files. |
| `dense_baseline_mac.py` | Dense baseline using SentenceTransformers. |
| `build_family4.py` | Additional-family construction scaffold. |
| `run_xling.py` | Historical cross-lingual probe. |

## Result scope

The original README described its results as containing all paper numbers. The supplied historical `v2_results.json` did not match all metrics in the earlier five-page manuscript. For example, historical MLP-margin recall is 90.2069%, while the earlier manuscript reported 93.70%. Keep these versions distinct. See `../EVIDENCE_INDEX.md` for the current run and the historical discrepancy. The earlier 93.70% claim has been superseded in the current manuscript; this note preserves the reason for that correction.

No historical model training was rerun during repository preparation.
