# Stage A risk register

> Current implementation status (21 November 2026): Stages B-F were subsequently authorized together and implemented. The Stage A design below records original intent; current APIs and any differences are documented in [MODULE_GUIDE.md](MODULE_GUIDE.md) and the completion report.

This register originated at Stage A. Software controls are now implemented and tested in the listed stages; representation quality, clinical validity and scientific benefit remain unverified research risks.

| ID | Risk and implication | Planned mitigation / verification | Stage |
|---|---|---|---|
| R01 | Hidden TRAIN labels reach selection through datasets, class buckets or cached arrays | Capability-limited QueryContext, label-free loader, distinct truth importer/oracle, fixed-split hidden-label permutation tests, trace oracle calls | B/D/E |
| R02 | Validation labels treated as free; misleading zero-label efficiency claim | Separate up-front unique validation cost, total-target-label reporting, explicit zero TRAIN-label wording, same validation resources for comparisons | B/E |
| R03 | Validation labels/test data influence SSL or acquisition pool | Freeze manifests before training, split-aware loader/store checks, reject queried non-TRAIN IDs; no test evaluator dependency in controller | B/C/E |
| R04 | Same patient/lesion or duplicate images cross splits | Group connected components; verify content duplicates; report missing patient IDs and fallback grouping level | B |
| R05 | Rare classes absent in validation or acquired data | Fixed class order, raw support flags, explicit prior smoothing or strict error; report missing classes and metric denominators | B/D/E |
| R06 | Rare-class images predicted as majority class receive low difficulty | Keep uncertainty/diversity baselines, log only post-annotation composition, later sensitivity/ablation studies; no guarantee of minority discovery | D/E |
| R07 | Geometry does not align with diagnostic boundaries | Treat as hypothesis; test random/entropy/representative/TypiClust baselines later, avoid class-coverage guarantees from eight initial labels | D/E |
| R08 | Duplicate vectors/single center break ideal SUPClust occupancy rule | Nonempty-center handling, exact duplicate diagnostics, declared farthest-first fallback or strict mode; one score provenance per selected ID | D |
| R09 | Source FAISS squared distances or self-neighbor assumptions alter method | Explicit Euclidean contract, identity-based self exclusion, backend agreement and hand-calculated tests | C/D |
| R10 | Feature scaling, temperature and top-fraction conventions differ from paper | Version each choice; record normalization/temperature/k/rounding; future sensitivity experiments, no exact reproduction claim | C/D |
| R11 | Large `K=|L|+b` K-means and neighbors become expensive | Chunk scoring, avoid dense N-by-N distance matrices, profile separately, optional exact FAISS; no silent cluster cap | C/D |
| R12 | Nearly constant difficulty causes min-max amplification | Explicit tolerance/constant-signal diagnostic and configurable absolute normalization; log extrema | D |
| R13 | Fake BALD from dropout-free model or changing BatchNorm | Explicit trained dropout configuration, module-specific stochastic mode, RNG/mode restoration and failure tests | C/D |
| R14 | Clinical color/border information removed by SSL transforms | Conservative logged defaults; high-risk augmentations opt-in; real performance/clinical suitability unverified | C |
| R15 | Frozen feature cache stale or rows misaligned after fine-tuning | IDs and hashes for preprocessing/model/splits, validate all dimensions/versions, re-extract explicitly | C |
| R16 | Failure/resume causes double annotation charges or lost pool transitions | Replayable request IDs and a single committed pool/ledger snapshot; fault injection across persistence phases | B/E |
| R17 | Final test scores repeatedly inspected for tuning | Explicit evaluation lock and final-only CLI path; no test loop inside active learning | E |
| R18 | Missing-class metrics inflate or depress comparisons invisibly | Report fixed-eight and supported-class variants, undefined empty cases, stable class order, known-array tests | E |
| R19 | Eight-logit study confused with official nine-category challenge task | Primary held-out closed-set protocol; reject unknown labels; document separate external assessment rather than silent remapping | B/E |
| R20 | Old reference repos contain source defects or incompatible dependencies | Write original modules; inspect math; current package environment isolated from references; lock only tested Python 3.11 stack | B-F |
| R21 | Unclear licenses, nested noncommercial code, unattributed copying | No source copying by default, source-specific license notes, exclude repositories/PDFs/split arrays from final ZIP | A/F |
| R22 | CPU smoke success mistaken for scientific validation | Mark every synthetic artifact; no real checkpoints/performance tables; report implementation and research validation separately | C-F |
| R23 | Deep classifier overfits extremely small initial pool | Frozen SSL linear mode and staged fine-tuning, acquired-class support logs; later empirical comparisons required | C/E |
| R24 | Grouped split ratios are approximate and no labels are available to stratify | Persist achieved counts/groups/seed, report validation missingness; any future paid stratified setup must be a separately declared protocol | B |
| R25 | Public Python APIs mistaken for adversarial label security | Document accidental-leak prevention scope; no adversarial guarantees against arbitrary filesystem or reflection access | B-F |

No class-aware stopping, Co-teaching, VOG cleaning, clinical deployment or automatic full-dataset training is part of the mitigation plan.
