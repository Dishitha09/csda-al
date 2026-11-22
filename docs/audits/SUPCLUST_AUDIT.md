# Stage A source audit: SUPClust and TypiClust

Status: design evidence only; no CSDA-AL implementation. Audited 2026-11-21. No real training or experimental evaluation was performed.

## Sources and reading coverage

The complete seven-page supplied PDF is **SUPClust: Active Learning at the Boundaries**, Yuta Ono, Till Aczel, Benjamin Estermann and Roger Wattenhofer, ICLR 2024 Workshop on Practical Machine Learning for Low Resource Settings; arXiv:2403.03741v1, dated 6 March 2024. All pages, including references, were read using `pypdf`. Printed pages 2, 3 and 4 were also rendered with `pypdfium2` and visually inspected. The rendered equations agree with the extracted text.

Local reference paths below are relative to `reference_resources/ISIC2019_AL_Architecture_Resource_Pack/`, which remains separate from the original integrated project:

- `papers/supclust_active_learning_at_boundaries.pdf`.
- `reference_code/supclust_unofficial_starter/README.md`, `supclust_selector.py`, and `tests/test_supclust_selector.py`, read completely.
- `reference_code/typiclust/README.md`, `deep-al/pycls/al/typiclust.py`, `deep-al/pycls/datasets/utils/features.py`, root `LICENSE`, `deep-al/LICENSE`, and `scan/LICENSE`.
- The complete master specification was supplied in the inherited task context and reviewed before this audit. Its Stage A gate governs this work.

The unofficial starter is explicitly not author-released SUPClust code. The TypiClust repository identifies itself as official code for Hacohen, Dekel and Weinshall (2022), **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets**, as well as subsequent methods. This audit does not treat those other methods as project requirements.

## Verified method and equations

The paper's printed page 4, first paragraph, gives the four-step SUPClust method: train self-supervised representations, cluster into the labeled count expected after the current query, exclude clusters already occupied by labeled examples and choose the largest remaining clusters, retain the most typical 10% within each cluster, then maximize SUP. Printed page 4 section 4.1 uses SimCLR ResNet-18 features for ISIC 2019 and describes both a deep classifier and a linear classifier on frozen SSL embeddings.

For a training pool `P = L union U`, current labeled IDs `L`, and requested query size `b`, the requested cluster count is:

```text
K_clusters = |L| + b
```

`L` contains annotation status, not class labels. Cluster all training embeddings, including previously labeled images; validation and test records are excluded. An occupied cluster has at least one ID in `L`. For ordinary nondegenerate K-Means with `K_clusters` nonempty clusters, at most `|L|` clusters are occupied, so at least `b` eligible clusters exist. A shortage is therefore a degeneracy/implementation condition, not a normal reason to inspect ground truth.

Equation (1), printed page 2 in related work, is:

```text
typicality(x) = [mean_{z in kNN(x)} ||x - z||_2]^-1
```

The paper states nearest neighbors in an embedding space. It does **not** explicitly fix the neighbor count or the global-versus-within-cluster search scope. The within-cluster convention is supported by the supplied TypiClust implementation, which passes only a cluster's candidate features to `calculate_typicality` (lines 97-100). It must be documented as an implementation choice rather than a fully specified detail of the SUPClust paper.

Equations (2) and (3), printed page 3 in section 3, are:

```text
w_ij = exp(-||c_i - c_j||_2 / T)
       / sum_{k != i} exp(-||c_i - c_k||_2 / T),  j != i

SUP_i(x) = [sum_{j != i} w_ij ||x - c_j||_2]^-1
```

The source uses another index placement for the same weights. Weights are shared by candidates in the same cluster; they depend on center-to-center distances, and the candidate score uses candidate-to-other-center distances. The own center must be omitted. Distances are ordinary Euclidean distances, not squared distances. Select the highest SUP from the typicality-filtered candidates. Softmax must be stabilized numerically. `T` must be finite and positive.

Figure 4's caption uses `T = 1`. This does not establish that temperature as the globally specified hyperparameter of every experiment. The seven-page paper does not give the implementation's exact `k_neighbors`, rounding rule for the top fraction, K-Means restart count, singleton policy, or feature normalization policy. Each becomes a versioned CSDA-AL assumption.

Cluster boundaries are a proxy derived from representation geometry. Their coincidence with diagnostic boundaries is a research hypothesis, not a property the software may assume or enforce using hidden classes. Initial acquisition of eight images does not guarantee coverage of eight diagnostic classes.

## Reference code findings

| Reference location | Actual behavior | CSDA-AL consequence |
|---|---|---|
| Starter `select_supclust`, lines 71-100 | Accepts IDs/features and labeled-status IDs; validates basic shape, finite features, unique IDs and query feasibility; clusters into `|L| + b` using K-Means with `n_init=10` and a seed. | Algorithm skeleton is consistent with the paper. Reimplement with stronger typed contracts; do not expose a dataset or oracle to the strategy. |
| Starter `_cluster_typicality`, lines 42-53 | Within-cluster Euclidean neighbors, `min(10, m-1)`, removes one zero-distance self slot, reciprocal with `1e-12` denominator floor; singleton score `1.0`. | These are explicit conventions. Remove self by row identity in the new implementation; tied duplicate distances need robust handling. |
| Starter lines 103-120 | Excludes occupied clusters; sorts eligible clusters by descending size then raw cluster index; top fraction uses `max(1, ceil(0.1*m))`; typicality ties follow input order. | Keep the fraction/rounding policy explicit. Canonical ID order should make tie outcomes independent of accidental feature-row ordering. |
| Starter lines 123-140 | Stabilized center softmax, omits own center, maximizes reciprocal weighted Euclidean distance through `argmin`; caps reciprocal denominator at `1e-12`. | Eq. (2)-(3) match; future numeric tests should independently verify values and rankings. |
| Starter lines 95-112 | Rejects `K_clusters < 2`; raises when duplicate embeddings cause too few nonempty eligible clusters. | Missing small-pool and degeneracy policies must be supplied and clearly marked as departures from the ideal method. |
| Starter lines 85-88 | Positive-temperature and neighbor checks exist, but finite temperature and integral argument validation are incomplete. | Reject NaN/infinite temperature and nonintegral counts with clear errors. |
| Starter tests, lines 14-40 | Three tests cover deterministic first query, no reselection on a second query, and several bad inputs. | Passing these tests is useful source audit evidence; it does not establish full reproduction or CSDA-AL correctness. |
| TypiClust `get_nn`, lines 7-16 | Unconditionally imports FAISS and constructs a GPU index. FAISS `IndexFlatL2` returns squared L2 distances. | Do not adopt this runtime or its distance mismatch with the SUPClust equation. Use explicit Euclidean distances with CPU fallback. |
| TypiClust `calculate_typicality`, lines 27-31 | Reciprocal mean of FAISS distances with `1e-5` offset. | This code's squared distances are a relevant implementation difference; literal reproduction is distinct from the paper-consistent Euclidean baseline. |
| TypiClust lines 34-40, 45-47, 60-73 | Switches to MiniBatchKMeans beyond 50 clusters, caps total clusters at 500, uses `K_NN=20`, and does not pass the stored seed directly to K-Means. | These are reference implementation settings, not SUPClust requirements. The integrated SUPClust method must not silently impose the cap. |
| TypiClust lines 83-103 | Drops clusters of size <=5; orders by labeled count then size, cycles through clusters, and uses `min(20, floor(m/2))` neighbors. | Distinct from SUPClust occupied-cluster exclusion. A simplified project baseline should disclose differences rather than claim identical official behavior. |
| TypiClust feature loader, lines 23-35 | Normalizes features by default; paths assume a particular reference layout. | Store normalization identity in FeatureStore, reject zero norms when normalizing, and avoid hardcoded reference paths. |

The supplied TypiClust `labels` variable in selection holds cluster assignments, not diagnostic truth. It is not itself evidence of hidden-label acquisition. Its positional-index interfaces are nevertheless unsuitable for the integrated ID-based contracts.

## Ablations: important distinction

Paper printed page 4 section 4.2 explicitly describes **without SUP** as randomly choosing from the top 10% typical candidates. It is not equivalent to always choosing the most typical point; the latter is the TypiClust comparison. Planned implementations must therefore be distinct:

- `SUPClust`: filter by typicality, maximize SUP.
- `without_typicality`: maximize SUP over all eligible members of the selected cluster.
- `without_SUP`: seeded random choice among the typicality-filtered members.
- `TypiClust`: maximize typicality, with baseline conventions stated.
- `KMeansRepresentative`: nearest-to-own-center representative, a separate baseline.

Each random draw uses a run-derived local seed and canonical IDs, and records its method/version.

## Proposed implementation decisions for review

These are design recommendations for Stage D, not implemented behavior:

1. Use a fixed, L2-normalized SimCLR backbone embedding store for primary acquisition. Keep raw embeddings accessible through a separately identified cache configuration. The normalization choice changes geometry and temperature scale and must be recorded.
2. Use within-cluster ordinary Euclidean neighbors; default `k_neighbors=10` is a disclosed assumption. For size `m > 1`, use `min(k_neighbors, m-1)` other examples. Singleton typicality uses a documented finite convention because no neighbor-based estimate exists. Mark the singleton status in diagnostics.
3. Retain `max(1, ceil(0.1*m))` candidates. Canonicalize all input rows by image ID before deterministic K-Means. Use explicit `random_state`, `n_init=10`, and algorithm settings; compare seeds only within a tested dependency environment.
4. Request exactly `|L|+b` clusters after clipping `b` only to remaining annotation budget and unlabeled count. Record both requested and effective counts. Do not silently replace the paper rule with `K=8`, a maximum-cluster cap, or `K=b` in later rounds.
5. For fewer distinct vectors than requested clusters, use nonempty clusters for center scoring, emit a degeneracy diagnostic, and fill any query deficit through a predeclared deterministic geometry-based or seeded-random fallback among remaining unlabeled IDs. Log every fallback selection. In the normal case, select one image per unoccupied cluster only. Fallback behavior is a project extension.
6. For only one usable center, SUP is undefined. Use an explicitly labeled representativeness fallback and return no misleading finite SUP value. Zero query budget returns an empty QueryBatch; infeasible or malformed contexts fail before selection.
7. Compute scoring in float64 on CPU, use a disclosed positive denominator floor for zero distances, and record degenerate flags. Validate finite features, nonzero feature dimension, count types, finite positive temperature, unique aligned IDs, exact training-pool membership and labeled/unlabeled disjointness.
8. Avoid an `N x N` distance matrix and use nearest-neighbor search plus chunked candidate-to-center distances. Optional FAISS must convert squared L2 output to Euclidean distance. It may change only performance within declared tolerances, not the method definition.

## Licenses and provenance

The full extracted input must be preserved unchanged. This audit is a file-level observation, not a legal opinion.

| Scope | Observed notice | Plan |
|---|---|---|
| `typiclust/LICENSE` | MIT, copyright 2022 Avihu Dekel and Guy Hacohen. | Cite method; preserve notices if any substantial source is later reused. |
| `typiclust/deep-al/LICENSE` | MIT, copyright 2022 Akshay L Chandra and Vineeth N Balasubramanian. | A distinct nested notice; do not omit it if adapting that source. |
| `typiclust/scan/LICENSE` | **Creative Commons Attribution-NonCommercial 4.0 International**, confirmed at lines 1 and 57-58. Noncommercial sharing/adaptation clauses at lines 151-155. `scan/README.md` lines 170-172 agrees. | Correct the input pack's overly broad MIT summary in new project notes. Do not copy SCAN source into the original package. The root MIT notice does not erase the nested notice. |
| `supclust_unofficial_starter/` | No LICENSE file found. README explicitly identifies a non-author starter. | Read-only reference; independently implement equations and cite the paper. Do not assign it an assumed MIT license. |
| SUPClust paper | User-supplied research literature. | Cite title/authors/version and page/equation locations. Exclude supplied PDF from the final original project archive. |

No reference code is copied into CSDA-AL by this audit.

## Actual source checks and synthetic demonstrations

**NO REAL TRAINING.** These executions use existing reference code only; no project module or project test was implemented.

Commands run from the workspace were Python here-strings via `python -` to extract all PDF pages with `pypdf.PdfReader`, print source with line numbers, and render printed pages 2-4 with `pypdfium2.PdfDocument`. Rendered files were inspected using the image viewer. `rg --files -g '*LICENSE*' -g '*NOTICE*' -g '*COPYING*'` enumerated the nested license files. Source reads do not claim source execution coverage.

The starter's supplied tests were run from its own directory with bytecode and pytest cache writes disabled to preserve references:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q -p no:cacheprovider tests
```

Actual output:

```text
...                                                                      [100%]
3 passed in 29.48s
```

An additional read-only reference invocation imported `select_supclust` through `importlib.util` and passed three small synthetic inputs. For four vectors `[-10,0],[-9,0],[9,0],[10,0]`, IDs `a,b,c,d`, budget 2, and default seed 42:

```text
two_clusters [('a', 2, 1.0, 0.051282), ('c', 2, 1.0, 0.054054)]
singleton_request ValueError SUP score requires at least two clusters and at most N
duplicate_features RuntimeError not enough eligible clusters: inspect K-Means assignments
```

The singleton input was `[[0,0]]` with budget 1. The duplicate input was four identical zero vectors with budget 2; scikit-learn emitted a `ConvergenceWarning` that only one distinct cluster was found. These failures are intentional audit findings about reference edge cases, not failed CSDA-AL tests. Scores above are synthetic geometry outputs, not classifier or medical results.

## Required future tests and remaining uncertainty

- Independently hand-calculate Euclidean typicality, weight normalization/exclusion, SUP rankings, and the three variants; test both first and subsequent queries.
- Verify zero budget, pool exhaustion, one center, singleton and two-point clusters, top-fraction rounding, all duplicate and partly duplicate embeddings, unavailable eligible clusters, zero-distance denominators, and fallback audit metadata.
- Verify permuted feature-row order preserves selected IDs, same-seed repeated runs are stable, deterministic tie behavior, float32/float64 agreement within tolerance, finite validation failures, and behavior under the supported CPU dependency lock.
- Prove strategies are invariant to changes in sealed hidden class labels; ensure validation/test IDs and oracle objects cannot enter QueryContext.
- Verify occupancy uses labeled IDs only; no query repeats, output <= budget, and returned IDs are training-only and previously unlabeled.
- Exact paper experimental normalization, neighbor count, universal temperature, K-Means settings and rounding are unresolved from the supplied seven-page paper. Document chosen defaults and future sensitivity experiments. Do not claim exact published reproduction or performance until later ISIC experiments.

Only this audit document was added by the audit task. Stage B and later implementation remain gated on user approval of Stage A.
