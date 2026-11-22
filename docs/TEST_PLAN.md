# Verification and stage acceptance plan

> Current implementation status (21 November 2026): Stages B-F were subsequently authorized together and implemented. The Stage A design below records original intent; current APIs and any differences are documented in [MODULE_GUIDE.md](MODULE_GUIDE.md) and the completion report.

The Stage A acceptance criteria below are now implemented in the test suite. Final execution passed 131 cases; see [Stage F review](STAGE_REVIEWS/STAGE_F.md) and test_logs/verification.json. The original source-audit checks remain in the Stage A review.

## Test boundaries

Use synthetic embeddings, tiny generated RGB images and known prediction arrays. Do not download datasets/weights or run full SimCLR/ISIC training. CPU optimization tests perform one or two batches only. Label all smoke artifacts synthetic. Deterministic checks are within a recorded numerical environment; cross-platform bitwise equivalence is not promised.

| Stage | Acceptance checks | Meaningful failure caught |
|---|---|---|
| B | Install/import in Python 3.11; invalid config/unknown keys; named class-column remapping; missing files/ambiguous truth; unique IDs; patient/lesion transitive grouping; duplicate content; split replay | Invalid schemas, grouping leakage, untested dependency claims |
| B | Oracle validation before any charge; repeated IDs/non-TRAIN rejection; exact-request replay; new-request reselection rejection; budget zero/negative/partial; state/ledger atomic recovery | Hidden-label exposure, double charging, pool corruption |
| C | ResNet output `[N,512]`; two-layer projector; independent two views; NT-Xent hand identity/finite gradients/one CPU step; invalid temperature/N=1; state-dict round trip | Wrong feature extraction, missing positives, bad denominator/reduction, disconnected gradients |
| C | Eight logits; probabilities sum to one; no target in inference/SSL loader; head-only encoder weights and BN buffers unchanged; unfreezing coverage; cumulative-label dataset restriction | Leakage, incorrect freeze semantics or optimizer groups |
| C | ID-aligned features; repeated deterministic extraction; raw/L2 identities; zero norms; cache wrong hash/model/order/corruption; CPU neighbors and optional FAISS ordinary L2 | Stale features, reordered predictions, distance mismatch |
| D | Hand SUP weights/typicality/SUP scores; first and later rounds; exact K; occupancy; largest eligible; top-10% rounding; three ablation ranks | Paper formula or selection-rule drift |
| D | Zero budget; empty/singleton/tiny pools; duplicate vectors; fewer effective centers; fallback or strict-mode reason; ties, permuted input rows; float32/64 tolerance | Degenerate crashes, invalid SUP, accidental row-order dependence |
| D | Random repeatability; entropy known ranks/zero probabilities; greedy Core-Set empty seed/min-distance update/no repeats; prediction ID/class alignment | Baseline defects that invalidate comparisons |
| D | Support-aware recall and smoothing; explicit prior classes; no-validation baseline; alpha endpoints; constant difficulty and tolerance; invalid probability arrays | Fictitious difficulty, score instability or malformed outputs |
| D | Optional BALD: zero for identical draws, positive for disagreeing confident draws; repeat seeds; dropout-free error; BN/RNG/modes restored after success/failure | False epistemic uncertainty and mutated inference models |
| E | Synthetic eight-class engine with zero initial TRAIN labels, budget-limited SUP seed then prediction-based query; at most two training steps per tested path; exact unique counts | An architecture that imports but cannot execute its primary pipeline |
| E | Failed trainer after annotation retains purchased labels; resume from transaction phases equals uninterrupted state/next selections; final partial batch/max-round stop | Nonrecoverable or inconsistent acquisition histories |
| E | Known-array fixed-eight confusion/accuracy/precision/recall/F1; support-zero policies; supported balanced accuracy; head/middle/tail memberships from acquired labels; AULC and target savings cases | Misleading metrics, hidden-frequency use, fabricated curve interpolation |
| E | Validation permitted; test lock required; strategy cannot consume test EvaluationResult; JSON logs are finite, ordered, versioned and contain no unrevealed targets | Test tuning, poisoned artifacts, provenance gaps |
| E | Every required CLI command runs/has help; dry runs produce honest artifacts; missing real dataset gives clear failure | Nonfunctional entry points or misleading no-op demos |
| F | All available tests/smokes, package install/import from extracted ZIP, ZIP CRC/manifest and exclusions, source-leak review, final report tied to logs | Incomplete or unverifiable final deliverable |

## No-label-leakage test design

1. Freeze image records, splits, embeddings, predictions, difficulty aggregates, budgets and seeds. Construct two TRAIN oracles with permuted hidden truth mappings and run each selector before annotation. Selected IDs, scores and audit metadata must match exactly. This tests direct hidden-TRAIN-label access; it does not claim later learned predictions are label-independent.
2. Give a selector a guarded context containing no oracle/dataset/targets and instrument oracle requests. There must be zero annotation calls until the controller validates the completed QueryBatch.
3. Insert validation/test IDs into a proposed batch and verify rejection before labels are revealed or charged. Check the SSL feature-store universe and all query contexts are TRAIN-only.
4. Change test truth while holding all training/validation inputs fixed. The entire acquisition history must remain unchanged. A final evaluator is never called during the active loop.
5. Trace every supervised training target to an acquired receipt. Validation labels may only feed evaluator/difficulty computation; gradients cannot use validation batches.
6. Inspect caches, events and configuration for unrevealed labels and full-pool class-frequency inputs. Post-annotation composition is computed after the current query is frozen.

These controls catch accidental information paths. Python object privacy alone cannot secure labels against arbitrary external code.

## Required command acceptance matrix for E/F

```powershell
python -m csda_al.cli inspect-data --config configs/dataset.yaml
python -m csda_al.cli pretrain-ssl --config configs/simclr.yaml --dry-run
python -m csda_al.cli extract-features --config configs/experiment.yaml --dry-run
python -m csda_al.cli cold-start --strategy supclust --budget 8 --synthetic
python -m csda_al.cli run-al --config configs/experiment.yaml --synthetic --dry-run
python -m csda_al.cli evaluate --help
python -m pytest -q
```

`inspect-data` requires user-supplied ISIC files; in an architecture-only environment its expected acceptance includes the explicit missing-input error. A separate synthetic fixture verifies successful ingestion. The final report must distinguish commands that succeeded, intentionally rejected missing inputs, were skipped, or failed; no prewritten PASS results.

Ruff and mypy run if installed/configured, with actual results recorded. Passing existing reference tests at Stage A does not count toward implemented CSDA-AL test coverage.
