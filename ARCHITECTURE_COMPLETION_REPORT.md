# CSDA-AL architecture completion report

Date: 21 November 2026. Target: original eight-class ISIC 2019 closed-set research. User approval: "do all stages now" superseded remaining staged pauses. **Software architecture complete; no full ISIC training or scientific results.**

## Implemented scope

- Python 3.11 installable package, validated YAML configuration, exact tested dependency lock and CLI.
- Label-free records, grouped/content-bound splits, sealed TRAIN oracle with replay-safe charges, separate validation cost, immutable pools and recoverable persistence.
- Original PyTorch SimCLR ResNet-18, two-layer projection head, NT-Xent and dual-view transformations; extraction/caching of backbone features.
- Eight-logit ResNet classifier, strict SSL transfer, head-only and staged unfreezing API, acquired-only fitting, optional acquired-frequency weighted CE, real versioned state-dict checkpoints.
- SUPClust/TypiClust/K-means representative, two paper-consistent ablations, deterministic geometric fallbacks, random/entropy/Core-Set/BALD and prediction-weighted difficulty strategies with optional diversity.
- Fixed-budget active-learning loop, per-round state and audit logs, exact annotation accounting, checkpoint and transaction recovery.
- Split-aware metrics and final-test locks, per-class support/confusion, balanced/macro/micro/weighted summaries, head/middle/tail metrics, AULC, label savings, JSON/CSV and plotting utilities.
- Synthetic end-to-end tests, command verification, source/research traceability, license audit, diagrams, risk/limitation documentation and archive tooling.

## Actual validation

**131 final unit/integration/CLI/packaging tests passed**, with no skipped scientific experiment presented as a software pass. Ruff passed, configured Mypy passed, all 11 verification command groups met their expected exit status, and all package modules imported. See [machine results](docs/test_logs/verification.json), [test transcript](docs/test_logs/pytest.txt) and [JUnit cases](docs/test_logs/pytest.xml). The runtime was Python 3.11.15 / torch 2.6.0+cpu / torchvision 0.21.0+cpu.

The suite contains known-array analytic tests, real CPU ResNet optimizer steps, SUPClust independent formula/ablation checks, hidden-label permutation invariance, supported/missing class handling, stale split/cache checks, durable oracle costs, fault-injected model/state recovery, no-test engine boundary, CLI paths and archive exclusion/integrity tests. Code and tests are concrete; no placeholder runtime modules are used.

Source-audit verification separately confirmed 240 resource-manifest payload hashes, 228 matching standalone reference files and the identity of the extra master attachment. All three supplied papers were fully read. References remain outside the final architecture archive.

## Deviations and research limits

SUPClust neighbor scope/count, ceil top-10%, temperature/normalization and degenerate fallback are disclosed conventions where the paper is incomplete. Without_SUP is seeded random among typical candidates. The difficulty method is our prediction-membership adaptation of Flexible Sampling, not a reproduced published AL algorithm. Priors for unsupported validation classes are explicit; prediction bias may still miss rare classes. Validation labels are separately costed. Test data cannot influence the primary loop.

The official challenge includes an unknown test category; this eight-logit project does not claim full nine-category challenge recognition. Co-teaching/VOG cleaning, class-aware stopping, distributed annotation, clinical deployment, full-data/CUDA benchmarking and real model performance are not implemented or claimed. Optional FAISS is fallback-isolated; the tested backend is scikit-learn. No pretrained weights were downloaded or shipped. The architecture creates genuine synthetic checkpoints only during tests/demos, clearly tagged and omitted from the archive.

## Final artifact

Requested filename: **CSDA_AL_Complete_Architecture.zip**. The builder validates expected files, CRCs, member hashes, safe extraction and extracted-source imports, with optional full extracted tests. ARCHIVE_MANIFEST.json inside the archive records payload hashes. Final archive SHA-256 and verification output are supplied beside the ZIP and in the final response; the archive cannot include its own hash without circularity. Extracted test results and final package verification are recorded with the completed delivery after packaging.

## Extracted-delivery verification completed

The isolated extracted project passed **131 tests, 14 third-party deprecation warnings, in 39.19 seconds**. Package and CLI imports resolved inside the extracted source tree. ZIP CRC, manifest hashes and extracted-file hashes passed. See [actual extracted test output](docs/test_logs/archive_extracted_tests.txt) and [archive check record](docs/test_logs/archive_verification.json). The final repack includes these records and changes documentation only; source/config/test hashes are compared against the fully tested extracted payload. No runtime images, feature arrays or model weights are included. The delivered ZIP SHA-256 is recorded in the adjacent checksum and verification files.
