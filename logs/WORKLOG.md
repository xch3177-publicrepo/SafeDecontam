# Work log

## 2026-09-14T04:51:41+00:00 — Import reviewed reproduction snapshot

**Purpose and authorization.** The author approved an independent repository copy, created this public target, and invited the source maintainer with write permission. This commit imports the completed SafeDecontam manuscript, source, evidence, English documentation and verified packages on `codex/iccc2026-reproduction`. It does not modify main or repository visibility, and no force push is used.

**Provenance.** The reviewed source commit is `8d4685c3c0052875f2e8b0774eaed2b64b197ae9`; the private source delivery commit is `e5748609d9ebf7c8c20af11bdfc7c43a4dc1136f`. The independent target parent is `8369ef1fd6556d0e7665afb3ce69e052d8268b4d`. `SOURCE_PROVENANCE.json` and `PACKAGE_MANIFEST.json` record paths, version and hashes. Original private Git history, obsolete manuscript revisions, credentials and runtime caches are not imported. Selected historical experiment JSON remains explicitly historical.

**Changes and validation.** Untraceable constructed manuscript values were replaced with a complete fixed-configuration ten-seed run (primary recall 90.34%). Related constructed and auxiliary results, Figure 1, code, inputs, score arrays, numerical weights and documentation were updated together. Independent external audits match the paper's ARC, HellaSwag and C-Eval rows. Main MMLU results and author-confirmed human 82/71/1 counts remain unchanged. The final PDF has five pages and preserves the baseline abstract and author block. Raw logs, numerical/visual QA, failure records and remaining human-label limitations are in `revisions/2026-09-14-external-review/`.

**Packages.** Version `iccc2026-repro-20260914` contains the full PDF/LaTeX/code/evidence ZIP (SHA-256 `d5c677ff60e9fb99a562cf531159f0ca61328694b0565292ecc33eecfcfca986`) and source ZIP (`f6828efb23c92f8fca6bf3c72368eaa0864f9fe49cfa376454b5c9a477b11968`). Both pass CRC and all nested hash checks. The five scientific verification categories passed on the source tree; this snapshot preserves those files byte-for-byte. Root and reproduction integrity checks run before commit. The package version is distinct from an engine version.

**Commands and synchronization.** Selective copy from the reviewed source checkout; generate root hashes; run `python3 tools/verify_package.py` and `python3 reproducibility/verify_delivery.py`; inspect staged paths; commit and push only `HEAD:refs/heads/codex/iccc2026-reproduction`. The release is pinned to this delivery commit. Final remote SHA, main, visibility and asset hashes are recorded separately after upload, avoiding self-referential commits.

## 2026-09-14T05:32:30+00:00 — Sync the approved availability statement and v2 packages

**Authorization and branch.** The author approved the local five-page sentence and requested GitHub push plus a PDF/LaTeX ZIP. Their merge of the previous reproduction branch is already in main at `06d02802ec20505382ede566179d972c5ed990fa`. This update uses `codex/availability-20260914` based on that commit, without pushing main, using force or changing visibility.

**Changes and inputs.** Exactly one Section IV-B sentence adds the public code/selected-data link. The approved TeX hash is `dffb8c5468dc1de62d99bf4fdcab3293b1af26c47c9afbe295d0f46c9e80a79f`; PDF hash is `bae82dac354b40e2315f0b83159ef0a695bf96861c6e4f695b4cfa7e83911bd8`. All prior prose, numerical results and formatting commands are unchanged. Source commit `f6023578a5c97f13aa32b1a97f0961434fb21413` and private delivery commit `9b764e67ad4d326be580311a06ddcea1d2e7fe4c` are recorded in SOURCE_PROVENANCE.json. No private Git history is imported.

**Package version and verification.** `iccc2026-repro-20260914-v2` adds the compact PDF/LaTeX ZIP (`538d70e7d122a14db5ff4d592259cebdcbcd03d84b137adfa2d69069b51fbdc8`) and updates the full reproduction ZIP (`fd0e3985fe727e4b7e9b59d970a53e3168660be274474c7588914671d27062f2`). The experimental ZIP remains byte-identical to v1. All ZIPs pass CRC/path/hash checks. Extracting the compact ZIP, checking its manifest, and executing `bash build.sh` regenerates the same five-page PDF byte-for-byte. All five approved pages passed visual QA. Trial failures and evidence limitations remain recorded under the dated availability revision. No experiments were rerun.

**Synchronization.** Refresh root integrity, review staged scope, commit, and push only `HEAD:refs/heads/codex/availability-20260914`. Publish the v2 release pinned to this commit; preserve prior releases. A pull request can present the update for author-controlled merge. Final branch/main/visibility/release/asset-digest checks are saved in a separate local synchronization record after upload. The package tag is not an engine version.

## 2026-09-14T06:28:52.308970+00:00 — Approved final four wording changes

**Authorization.** The author reviewed the local five-page candidate and explicitly requested pushing it to main. This applies to the independent public SafeDecontam repository. No force push or history rewrite is used; the private source is synchronized on its existing camera-ready branch.

**Inputs and changes.** Public v2 source `5e466688c6754468ebe36b0718384d2219eed176`; current public main `06d02802ec20505382ede566179d972c5ed990fa`. Exactly four prose edits are documented under `revisions/2026-09-14-final-wording/`. Approved PDF SHA-256: `8e8a867104c67bbd504e69be9e954fb6492ae470f705ce8fee85f0f16729af26`; TeX SHA-256: `ce06082e726fd0a436a265cd210b0c9093a728495c0495b0422891c7e2b8b9c8`. No numerical, experimental, abstract, author, or formatting change.

**Validation.** Exact source replacements and numeric tokens checked; all five pages visually inspected; three-pass build free of overfull boxes, undefined references, missing characters and LaTeX errors. A fresh compact ZIP extraction rebuilt a byte-identical PDF. Validation is preserved with the revision; no new experiment or PDF eXpress certificate is claimed.

**Outcome and next synchronization.** This source snapshot precedes v3 package assembly. Review staged paths and commit; prepare v3 packages, verify integrity, and fast-forward push public main. Keep prior releases and experimental evidence unchanged. Final remote commit, visibility, release and uploaded asset hashes will be verified and recorded after upload.
