# Implementation decisions and module map

> Current implementation status (21 November 2026): Stages B-F were subsequently authorized together and implemented. The Stage A design below records original intent; current APIs and any differences are documented in [MODULE_GUIDE.md](MODULE_GUIDE.md) and the completion report.

Implemented decisions and staged module map. All primary runtime modules listed below now exist; historical Stage A delivery notes are retained separately.

## Source-of-authority decisions

The user's message authorizes the bundled master prompt as the development specification. The final section originally limited work to Stage A; the user subsequently instructed "do all stages now", authorizing B-F together. The older problem slide and unrelated source comments do not authorize class-aware stopping or label cleaning. The additional attached master file has the same SHA-256 as the bundled master.

The original resource directory remains separate and unchanged. Final packaging includes the original integrated project and citations, without the supplied repositories, papers or data arrays. A project-wide software license is not assigned to the user's research work without an explicit owner choice; dependency and source notices will still be documented.

## Scheduled file map

Paths below are relative to the implemented `csda_al/` repository root. Every `__init__.py` added later will have an explicit package/export purpose; files will not be generated merely to imitate a tree.

| Stage | Files/modules | Responsibility and interface |
|---|---|---|
| B | `pyproject.toml`, `requirements.txt`, tested dependency lock, `.gitignore`, `src/csda_al/__init__.py` | Installable Python 3.11 package; CPU-first dependencies and version reporting. No assumed CUDA build. |
| B | `config.py`, `configs/{dataset,simclr,classifier,acquisition,experiment}.yaml` | Validated dataclasses, safe YAML, reject unknown keys and invalid budgets, resolve relative paths against config location, canonical config hash. Models can be described before model code is added. |
| B | `data/{metadata,split,dataset,transforms,oracle}.py` | Label-free records/loaders, fixed eight-class CSV parsing, grouped splits, separate acquired/validation/test datasets, request-based sealed oracle. Add model-specific transforms in C. |
| B | `engine/{pool_state,budget}.py`, `utils/{seed,io}.py` | Pool invariants, annotation accounting, deterministic local seeds, atomic persistence primitives. Full round orchestration waits for E. |
| B | `tests/test_{dataset,no_label_leakage,engine}.py` | Data, oracle, cost and state tests; expand engine tests in E. |
| C | `models/{backbone,simclr,classifier}.py` | ResNet-18 backbone, projection MLP, NT-Xent, eight-class model/adapter, optional configured dropout. No TensorFlow runtime. |
| C | `features/{extract,cache,nearest_neighbors}.py` | Aligned FeatureStore, versioned CPU embeddings, safe cache and Euclidean neighbor backend; optional FAISS fallback only if installed. |
| C | `training/{ssl_train,supervised_train,finetune}.py`, `engine/checkpointing.py` | Cumulative acquired-label fit, frozen/partial/full encoder phases, optimizer groups and real state-dict checkpoints. |
| C | `tests/test_simclr.py`, model/feature tests | One-step CPU synthetic optimizer tests, shape/gradient/buffer/cache/serialization tests. |
| D | `acquisition/{base,random,entropy,coreset,typiclust,supclust,bald,difficulty_aware}.py` | Validated QueryContext/QueryBatch protocol, all required selectors and SUPClust ablations; optional BALD without unsupported silent fallback. |
| D | `imbalance/{class_difficulty,probability_weighting,diagnostics}.py` | Paid-validation class difficulty, support/prior provenance, predicted-membership weighting, post-annotation composition diagnostics. |
| D | `tests/test_{supclust,difficulty,acquisition}.py` | Paper arithmetic, degeneration, ties, baseline consistency and oracle-invariance tests. |
| E | `engine/active_loop.py`, `utils/logging.py` | Fixed-budget controller, transaction/resume protocol, JSONL event/logger interface; no LNL or class-aware stopping branch. |
| E | `evaluation/{metrics,learning_curves,report}.py` | Split-aware metrics/lock, known-array checks, budget curves and real result serialization. |
| E | `cli.py`, `examples/{synthetic_smoke_test.py,sample_config.yaml}` | Real and synthetic CLI paths with explicit input checks and scoped dry runs. |
| E | `tests/test_{metrics,cli_smoke}.py`, integration tests | At most a couple of synthetic training steps; full pipeline, resume and command checks. |
| F | `scripts/package_project.py`, `ARCHITECTURE_COMPLETION_REPORT.md`, test logs, final diagrams | Exclusion-aware package builder; archive/manifest integrity, extracted import/test verification, real completion report. |

## Decisions and alternatives

| ID | Decision | Reason, tradeoff and verification |
|---|---|---|
| ADR-01 | CPU-compatible PyTorch/torchvision ResNet-18, Python 3.11 | Required by the specification. The current host has Python 3.13.9 and no torchvision; Stage B must create a separate 3.11 environment using available tooling, establish a compatible torch/torchvision pair and record actual tested pins. Host package presence is not compatibility proof. |
| ADR-02 | Original implementation from papers, no source concatenation | Avoid inconsistent label access, TensorFlow dependency and unclear nested licenses. Cite concepts and explicitly record any future substantive licensed adaptation. |
| ADR-03 | Fixed ID-based contracts and capability separation | Positional source APIs and label-bearing datasets are major leakage risks. Selectors receive immutable arrays/IDs, not arbitrary trainer/dataset callbacks. |
| ADR-04 | Default label-independent grouped splits | Prevents sealed class labels influencing acquisition setup. Rare classes may be absent from validation; expose support/prior policy rather than stratifying with hidden truth silently. |
| ADR-05 | Validation labels are a disclosed up-front cost | Difficulty has a supervised information requirement. Report training-budget and total-target-label axes, including costs for baselines using the same validation resource. |
| ADR-06 | Frozen SSL acquisition features by default | Makes cold-start geometry and caches reproducible. Fine-tuned feature acquisition requires explicit re-extraction/versioning as a future configured variant, never silent cache reuse. |
| ADR-07 | Explicit SUPClust geometry and degeneracy extension | Use ordinary Euclidean distances, within-cluster neighbors, ceiling top-10%, stable ID ties and logged farthest-first fallbacks. These details are engineering choices where the paper is incomplete. |
| ADR-08 | Raw paper difficulty and optional smoothed operational estimate | Preserve actual validation recall; default one prior-equivalent example with recall 0.5 stabilizes low support. Support-zero values remain flagged prior estimates. |
| ADR-09 | Prediction-weighted difficulty plus normalized entropy | Satisfies true unlabeled acquisition without true-class buckets. Model prediction bias can hide rare classes; the integration remains experimental. |
| ADR-10 | Optional explicitly trained dropout head for BALD | Standard ResNet-18 has no MC-dropout uncertainty by default. Freeze BatchNorm and restore modes/RNG. Do not inject randomness into a model and call it a reproduced Bayesian method. |
| ADR-11 | Atomic authoritative state plus replayable annotation requests | Avoid double charges and acquired-label loss after crashes. Support a single process/run writer initially; distributed annotation services are outside this architecture. |
| ADR-12 | Separate validation and final-test evaluation APIs | Stops the source pattern of inspecting test results every epoch from entering the primary controller. Runtime guards complement documented experiment discipline. |
| ADR-13 | Ordinary cross entropy by default | Keeps sampling and reweighting effects distinguishable. Weighted CE is opt-in with weights based only on acquired TRAIN labels and finite missing-class behavior. |
| ADR-14 | No automatic weight/data downloads | Synthetic tests stay offline and lightweight. Future real runs require explicit dataset paths and user-selected initialization. |

## CLI behavior planned for Stage E

The CLI will expose `inspect-data`, `pretrain-ssl`, `extract-features`, `cold-start`, `run-al` and `evaluate`. Commands follow the exact master-prompt examples. Dry-run means synthetic/tiny-step execution, not a no-op or fake checkpoint. Every artifact carries `synthetic=true` when appropriate. `evaluate --help` is safe without a dataset; final test execution needs a lock. Missing image/CSV inputs result in concise actionable errors and nonzero exit codes, never obscure downstream tracebacks. A command requiring validation difficulty explains missing paid validation inputs or uses the configured baseline.

## Historical Stage A delivery

`README.md`, `LICENSE_NOTES.md`, architecture/contracts/decisions/traceability/source-audit/risk/test-plan documents, three detailed source audits, Mermaid diagram source, and Stage A review/evidence logs. There is no `src/`, `pyproject.toml`, training script, model checkpoint or integrated unit-test suite yet. The subsequent explicit user authorization covered all remaining stages.
