# SafeDecontam

Code and evidence for **SafeDecontam: Risk-Controlled Benchmark Decontamination** (ICCC 2026, accepted).

## Current manuscript and packages

- [Five-page camera-ready PDF](SafeDecontam_ICCC2026_camera_ready_5p.pdf)
- [Matching LaTeX](SafeDecontam_ICCC2026_revised.tex)
- [PDF, LaTeX, code, and evidence ZIP](SafeDecontam_ICCC2026_5p_PDF_LaTeX_Code.zip)
- [Experimental source and evidence ZIP](SafeDecontam_ICCC2026_Experiment_Source.zip)
- [External-review corrections](revisions/2026-09-14-external-review/REVIEW_RESPONSE.md)
- [Evidence index and limits](reproducibility/EVIDENCE_INDEX.md)

The reviewed reproduction package is `iccc2026-repro-20260914`. The unqualified PDF filename is a byte-identical five-page alias. Run `bash build_camera_ready.sh` to build the paper; it rejects a page count other than five.

## Verify the available evidence

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r reproducibility/requirements-cpu.txt
python reproducibility/verify_delivery.py
python reproducibility/verify_all.py --output runs/verification-1
```

The checks replay saved scores and numerical model inference. They do not train models or fabricate missing human labels. Full refitting commands and data preparation are in [reproducibility/README.md](reproducibility/README.md).

The main MMLU audit remains unchanged. The current constructed results come from a new fixed-configuration ten-seed run with complete exported evidence, replacing earlier untraceable manuscript numbers. ARC, HellaSwag and C-Eval were independently rerun and match the paper. The pilot and frozen cross-lingual probe use the same newly exported primary model. Historical outputs are retained separately.

The author-confirmed human audit remains 82/71/1. Item-level human labels and original adjudication/access records are not included; only count-conditioned intervals can be recomputed from the supplied human aggregates. The constructed and auxiliary studies remain exploratory, separate from the raw-search clean-only calibration protocol.

## License and citation

The repository's existing [MIT license](LICENSE) applies to the authors' code. Upstream benchmarks, search data, models and the IEEEtran class retain their own terms; see [third-party notices](reproducibility/THIRD_PARTY_NOTICE.md). See [CITATION.cff](CITATION.cff). No paper DOI is asserted before the proceedings metadata is available.

## Repository provenance

This is the independent release copy under `xch3177-publicrepo/SafeDecontam`. The reviewed materials are on `codex/iccc2026-reproduction`; the repository's initial `main` branch is unchanged. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records the exact source and delivery commits. The package preserves selected historical experiment outputs for provenance without importing private Git history or obsolete manuscript packages.
