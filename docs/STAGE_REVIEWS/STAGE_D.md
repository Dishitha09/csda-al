# Architecture Stage D: acquisition algorithms and class difficulty

Status: implemented and tested on synthetic inputs. The user's later authorization to complete all stages superseded the earlier individual review stops; this report preserves the Stage D review evidence without requesting approval again. No ISIC model was trained, and no scientific performance result is reported.

## Goal and research relevance

Cold-start acquisition must work before any TRAIN class is known, and subsequent imbalance-aware acquisition must not infer eligibility from hidden true classes. This stage implements those operations with explicit ID-based contracts.

- Ono et al., **SUPClust: Active Learning at the Boundaries** (2024), printed pages 2-4, supplies typicality, center weights, boundary-proximity scoring, occupied-cluster exclusion and the four-step selection process.
- Hacohen, Dekel and Weinshall, **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets** (2022), supplies the typicality baseline. The supplied reference's squared FAISS distances, GPU requirement and hard cluster cap are not silently inherited.
- Ju et al., **Flexible Sampling for Long-Tailed Skin Lesion Classification**, supplies validation class-difficulty and uncertainty principles. Predicted class membership and entropy/difficulty fusion are explicitly the experimental CSDA-AL integration, rather than the paper's known-class training sampler.
- Khanal et al., **Active Label Refinement for Robust Training of Imbalanced Medical Image Classification Tasks in the Presence of High Label Noise** (2024), informs modular acquisition and entropy/Core-Set/BALD comparisons. No Co-teaching or VOG implementation is introduced.

See [the source audit](../audits/SUPCLUST_AUDIT.md) and [research traceability](../RESEARCH_TRACEABILITY.md) for precise reference paths and distinctions.

## Implemented files and interfaces

All files listed here contain original functional code rather than placeholder definitions:

| File under `src/csda_al/` | Behavior |
|---|---|
| `acquisition/base.py` | Frozen validated `PredictionBatch`, `QueryContext`, `QueryBatch`, strategy protocol, ID alignment, finite probability validation, deterministic ranking, JSON-safe export and configuration fingerprint. |
| `acquisition/random.py` | Local-seed uniform selection, canonical ID order and initial/random baseline aliases. |
| `acquisition/entropy.py` | Eight-class normalized predictive entropy with a defined `0 log 0` convention. |
| `acquisition/coreset.py` | Greedy Euclidean farthest-first coverage with incremental minimum distances; explicit centroid seed for an empty cover. |
| `acquisition/supclust.py` | Euclidean kNN typicality, stabilized center weights, chunked SUP scores, K-Means representative baseline, exact requested cluster rule, occupied-cluster exclusion, ablations and audited degeneracy handling. |
| `acquisition/typiclust.py` | Maximum within-cluster typicality baseline sharing the project cluster eligibility policy. |
| `acquisition/bald.py` | BALD scores from `[T,N,8]` probability arrays; reproducible MC dropout helper that freezes BatchNorm and restores model modes and RNG state. |
| `acquisition/difficulty_aware.py` | Experimental expected-difficulty fusion; difficulty-only variant; optional shortlist diversity reranking; explicit missing-validation fallback. |
| `acquisition/__init__.py` | Public exports and validated name-to-strategy factory. |
| `imbalance/class_difficulty.py` | Validation-only class recall, smoothing, unsupported-class prior flags, annotation-cost checks and checkpoint/split provenance. |
| `imbalance/probability_weighting.py` | Prediction-weighted expected difficulty; explicit pool min-max or identity normalization with constant-signal diagnostics. |
| `imbalance/diagnostics.py` | Predicted composition and separately labeled retrospective post-annotation composition. |
| `imbalance/__init__.py` | Public difficulty/diagnostic exports. |

The public selector contract is `select(QueryContext) -> QueryBatch`. FeatureStore contains every TRAIN ID, while PredictionBatch contains exactly the current unlabeled IDs; alignment is a join by ID rather than an assumption of shared row positions. Array inputs are copied and marked read-only. The context has no label oracle, truth mapping, dataset object or validation examples. The controller independently validates selected IDs before annotation.

Factory names include `random`, `entropy`, `coreset`, `bald`, `kmeans`, `typiclust`, `supclust`, `supclust_without_sup`, `supclust_without_typicality`, `difficulty_only`, `difficulty_aware` and `entropy_difficulty`. Invalid names or constructor options fail explicitly.

## SUPClust implementation

The effective query size is the smaller of the context's batch budget and remaining unlabeled count; the controller also enforces remaining annotation cost. The requested cluster count is always `|L| + b`, not a fixed diagnostic-class count. Input IDs are sorted before seeded K-Means with explicit restart count. Occupied clusters contain previously acquired IDs, regardless of their diagnostic label. Largest nonempty unoccupied clusters are queried first, with a deterministic member-ID tie rule.

Within a cluster of size `m`, ordinary Euclidean distances to `min(k_neighbors,m-1)` other members yield:

```text
typicality(x) = 1 / max(mean_neighbor_distance, epsilon)
w_ij = softmax_j(-||c_i-c_j||_2 / temperature), j != i
SUP_i(x) = 1 / max(sum_{j != i} w_ij ||x-c_j||_2, epsilon)
```

Typicality excludes self by row identity, including duplicate-vector cases. The candidate count is `max(1,ceil(typical_fraction*m))`, defaulting to the paper's 10%. Singleton typicality is a finite sentinel, with an explicit singleton flag. SUP candidates are scored in float64; candidate-center distances are chunked to avoid a three-dimensional distance tensor. The own cluster center is excluded from weights and scoring.

Defaults `k_neighbors=10`, `temperature=1`, `n_init=10`, epsilon floors, within-cluster neighborhoods and ceil rounding are recorded engineering conventions. The supplied paper does not specify all of these as universal experimental settings. Features carry their normalization and encoder identity through FeatureStore; no selector silently renormalizes its geometry.

The variants are distinct:

- `supclust`: filter by typicality, then maximize SUP.
- `without_typicality`: maximize SUP over every member of the selected cluster.
- `without_SUP`: seeded random draw among top-typicality candidates, matching the paper's ablation description.
- `typiclust`: maximize typicality, with the project's declared eligibility and Euclidean-distance conventions.
- `kmeans`: choose the nearest member to its own center.

Identical embeddings can make the requested number of nonempty clusters impossible. The implementation logs requested/effective cluster counts and uses only nonempty centers. When SUP has fewer than two centers, it is undefined. When eligible clusters cannot fill a query, farthest-first coverage fills the deficit using only remaining IDs and features. With no labeled/selected seed, the first fallback is nearest the candidate centroid. Every such ID has a reason and `None` SUP score. Strict mode raises instead. This is a disclosed software extension, not paper SUPClust. Zero budget returns an empty batch without clustering.

## Difficulty adaptation and uncertainty

Only paid validation labels may enter `compute_class_difficulty`. Test provenance is rejected. For class support `n_c` and true positives `TP_c`:

```text
recall_smoothed_c = (TP_c + smoothing * recall_prior) / (n_c + smoothing)
d_c = 1 - recall_smoothed_c
D(x) = sum_c p(c|x) d_c
U(x) = -sum_c p(c|x) log p(c|x) / log(8)
score(x) = alpha*U(x) + (1-alpha)*normalize(D(x))
```

Unsupported classes receive an explicitly flagged prior value, or strict mode raises. Their raw recall is undefined and serializes as JSON `null`. Empty validation returns no difficulty estimate. Annotation cost must cover every available validation label, and estimates carry model and split identities. Predictions and difficulty from different checkpoints are rejected.

Default difficulty normalization is min-max over the current unlabeled pool. Constant signals yield zero normalized difficulty and a diagnostic; identity normalization is available. Fusion defaults to `alpha=.5`, with no claim that this has been tuned. Missing validation triggers a configured entropy fallback or a clear exception. Zero-label cold-start selection remains the engine's responsibility and uses SUPClust before classifier-based querying.

Optional diversity reranking operates on a top-score shortlist and combines acquisition scores with greedy minimum feature distance. It consults only embeddings and acquired-ID membership. BALD computes `H(mean_t p_t) - mean_t H(p_t)` using normalized entropy. MC dropout requires an explicitly configured dropout layer; a dropout-free classifier raises. The helper accepts a tensor batch, never a target-bearing dataset, and preserves BatchNorm buffers, RNG state and module modes.

## Actual tests and commands

Owned tests:

- `tests/test_supclust.py`: hand-calculated Euclidean equations; requested cluster rule in first/subsequent rounds; deterministic IDs under row permutation; float32/float64 agreement; singletons, exhausted pools, duplicate vectors, strict fallback; ablation distinctions; invalid parameters.
- `tests/test_difficulty.py`: known-array recall and smoothing; unsupported-class priors; empty validation; annotation-cost/provenance failures; prediction-weighted membership; constant normalization; retrospective composition.
- `tests/test_acquisition.py`: every strategy's repeatability, budget/uniqueness/ID validity, JSON audit export; known entropy/BALD values; farthest-first geometry; distinct entropy/difficulty rankings; optional diversity; missing validation; stale identities; label-free fields; MC dropout mode/BatchNorm/RNG preservation.

Actual command from the workspace root, using the isolated Python 3.11 environment:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& '.\.venv-csda\Scripts\python.exe' -m pytest -q csda_al/tests/test_supclust.py csda_al/tests/test_difficulty.py csda_al/tests/test_acquisition.py
```

Final output is preserved in [the raw scoped test log](../test_logs/stage_d_pytest.txt):

```text
.............................................                            [100%]
45 passed in 3.78s
```

Earlier scoped runs also passed: 45 tests in 8.60s and 8.23s. The final run includes the strengthened hand-built ablation test and defensive center-shape validation. These are software checks on synthetic arrays and a tiny dropout network, not ISIC evaluation results.

The scoped lint command also passed after splitting three test statements onto separate lines:

```text
.venv-csda/Scripts/python.exe -m ruff check csda_al/src/csda_al/acquisition csda_al/src/csda_al/imbalance csda_al/tests/test_supclust.py csda_al/tests/test_difficulty.py csda_al/tests/test_acquisition.py
All checks passed!
```

Its output is preserved in [the lint log](../test_logs/stage_d_ruff.txt).

Integration review identified type-checker uncertainty about constructor-optional candidate IDs and optional smoothing metadata. A read-only `candidate_ids` property now exposes the validated tuple without changing `unlabeled_ids=None` constructor semantics. Heterogeneous audit dictionaries have explicit types, and smoothing export resolves its optional default. No type errors were suppressed. The subsequent scoped checks passed: **45 tests in 3.71s**, Ruff **All checks passed!**, and mypy **Success: no issues found in 13 source files**, using `--ignore-missing-imports --follow-imports=silent --disable-error-code=import-untyped`. See [typing follow-up evidence](../test_logs/stage_d_typing_followup.txt).

## Synthetic demonstration: NO REAL TRAINING

A hand-built one-dimensional pool has two clusters. Each has 29 coincident dense points and one outlier. IDs are `a00..a29` and `b00..b29`; values are 29 zeros plus 2, and 29 tens plus 12. Budget is two and seed is 42.

The implemented test confirms:

```text
without_SUP:        a00, b02   (seeded random within the top-three typical candidates)
TypiClust:          a00, b00   (maximum typicality, canonical ID ties)
without_typicality: a29, b00   (SUP favors the first cluster's boundary-side outlier)
```

These IDs illustrate the algorithmic distinction; they do not represent diagnostic classes or performance. An all-identical feature pool produces an audited fallback with no invented SUP values. Known-array BALD disagreement between two certain classes produces `log(2)/log(8)`, while repeated identical predictions produce zero.

## Limits and follow-up

The adaptation can miss difficult minority lesions predicted confidently as majority classes; predicted memberships cannot reveal that error without annotation. Retrospective composition is available after query commitment for later analysis and never changes the current selection.

Pool min-max normalization can amplify small difficulty differences. Validation priors are assumptions with visible flags, not observed recall. Difficulty estimates require a paid validation subset, so zero acquired TRAIN labels must not be described as zero total annotation cost. Future scientific comparisons must include this cost for every method.

K-Means and exact neighbor search may become expensive at large annotation budgets. The implementation avoids an `N x N` full-pool matrix, but scalable approximate backends and performance studies remain future work. Determinism is scoped to the tested numerical environment; dependency versions are recorded by run infrastructure.

The project TypiClust baseline follows declared shared policies instead of claiming byte-for-byte official reproduction. The full integrated engine and final archive verification belong to Stages E/F. This stage introduces no class-aware stopping, label cleaning, real training, pretrained downloads or fabricated experimental curves.
