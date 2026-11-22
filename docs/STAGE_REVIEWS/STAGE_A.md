# Architecture Stage A review

Status: submitted for explicit user approval on 21 November 2026. **NO REAL TRAINING.** No Stage B runtime modules or integrated tests were implemented.

## Goal and completed delivery

Audit the supplied literature/source snapshots and define a functional, testable, original architecture before coding. The design solves the zero-TRAIN-label initialization problem with SimCLR/SUPClust and specifies a prediction-based difficulty policy that does not require hidden class membership.

Read the complete authoritative master prompt, source map, blueprint and attribution notes; all 32 pages across the three supplied PDFs; the matching acquisition/training/model modules and relevant licenses. Verified all 240 resource-manifest payloads and all 228 files across the four standalone reference repositories. The separately attached master is byte-identical to the ZIP master. References were preserved separately.

Actual deliverables are documentation and audit evidence:

- [Architecture, algorithms and Mermaid diagram](../ARCHITECTURE.md), plus [standalone Mermaid source](../diagrams/csda_al.mmd).
- [Typed contract specifications](../DATA_CONTRACTS.md) for image records, features, oracle/receipts, pools, budgets, query context/batch, trainers, difficulty and evaluation.
- [Staged file map and implementation decisions](../IMPLEMENTATION_DECISIONS.md), [research traceability](../RESEARCH_TRACEABILITY.md), [risk register](../RISK_REGISTER.md), and [test acceptance plan](../TEST_PLAN.md).
- [Source and environment audit](../SOURCE_AUDIT.md), three detailed paper/source audits under `docs/audits/`, and [license notes](../../LICENSE_NOTES.md).
- This review, machine-readable integrity evidence, exact audit utility command, formula outputs and reference-test log.

## Research evidence and deviations

SUPClust Section 3 supplies the center-weight and SUP formulas, and page 4 supplies clustering/typicality selection. The design preserves `K=|L|+b`, occupied-cluster exclusion and top-10% filtering. Nearest-neighbor scope/count, ceiling rounding, deterministic ties, normalization and degenerate fallbacks are explicitly declared implementation choices. `without_SUP` is random within the typicality-filtered subset, distinct from TypiClust.

Flexible Sampling Section 2.3 supplies validation class difficulty and BALD. Its per-class accuracy is recall in this setting. Original true-class buckets are excluded; predicted membership `sum p(c|x)d_c` plus entropy fusion is our proposed adaptation. Priors/smoothing, validation label costs and missing class support are declared rather than treated as measured evidence.

Active Label Refinement Section 2.4 supplies a useful query/budget-loop precedent, adapted from correcting existing labels to acquiring new TRAIN labels. Co-teaching and VOG are excluded. Google SimCLR informs a fresh PyTorch implementation with conventional NT-Xent averaging and backbone feature extraction.

## Actual checks and outputs

| Check | Actual outcome |
|---|---|
| ZIP extraction, path containment and source manifest | PASS: 241 file entries; 240/240 payload sizes/hashes match |
| Standalone ZIP comparison | PASS: 12 + 44 + 137 + 35 files, zero mismatches/unbundled files |
| Additional master comparison | PASS: exact byte equality and same SHA-256 |
| Complete PDF text reading | PASS: 7 + 12 + 13 pages; selected method/formula pages rendered and visually inspected |
| Supplied starter pytest, root invocation | PASS: 3 passed in 5.65s, exit code 0; [raw log](stage_a_reference_tests.log) |
| Source singleton / duplicate-vector probes | Expected source limitations: ValueError / RuntimeError; not integrated project test failures |
| Independent formula arithmetic | PASS: assertions for entropy, membership, score rank and SUP rank; [actual output](stage_a_formula_checks.json) |
| Design cross-review | Three clarifications applied: effective one-center fallback, full-TRAIN features vs unlabeled predictions, total-label-cap budget formula |
| Final resource/document checks | See [machine-readable record](stage_a_evidence.json); no runtime source package is present |
| Integrated unit/integration/model tests | NOT RUN: reserved for approved implementation stages |

Reproducible audit invocation and relevant actual command forms are in [stage_a_commands.txt](stage_a_commands.txt). The evidence utility is a source-audit tool outside the project, not Stage B application code. No full-dataset or GPU experiment was run.

## Small synthetic demonstration

Three artificial eight-class probability vectors have first-two-class probabilities `(0.5,0.5)`, `(0.9,0.1)` and `(0.1,0.9)`, with all other probabilities zero. Using declared synthetic difficulties `(0.2,0.8,0.5,0.5,0.5,0.5,0.5,0.5)`:

| ID | Normalized entropy | Expected difficulty | Min-max difficulty | Fused score, alpha=0.5 |
|---|---:|---:|---:|---:|
| synthetic_a | 0.333333 | 0.500000 | 0.500000 | 0.416667 |
| synthetic_b | 0.156332 | 0.260000 | 0.000000 | 0.078166 |
| synthetic_c | 0.156332 | 0.740000 | 1.000000 | 0.578166 |

The descending order is `synthetic_c, synthetic_a, synthetic_b`. Candidate truth labels are not inputs. This is formula arithmetic, not an implemented acquisition or a lesion-classification result.

For one-dimensional cluster centers `0,2,6`, own center 0 and temperature 1, the other-center weights are `0.9820137900, 0.0179862100`. Candidate coordinates 0.2 and 0.5 yield SUP values `0.5342037750` and `0.6361546377`; the latter has higher SUP. This checks the formula component without claiming clustering, typicality or end-to-end selection is implemented.

## Limitations and next stage

There are no ISIC image/label files in the pack, no trained representation, no model checkpoint, no measured research result and no final project ZIP. The current host Python 3.13.9/no-torchvision environment is not the specified tested Python 3.11 stack. Source license and algorithm ambiguities are documented. Prediction bias, small validation support and geometric proxy quality require later experiments; synthetic checks cannot resolve them.

After approval, Stage B will create the installable package, validated configuration, data/split interfaces, oracle and pool/budget state with executable tests. Models, acquisition implementation, orchestration and packaging remain gated at C-F respectively.

The governing master prompt says: **"Never start the following stage until I explicitly approve."** Its final section says **"Do not begin coding Stage B until I approve Stage A."** This is why work stops here.

**Do you approve this architecture stage?**
