# Flexible Sampling source audit

Status: Architecture Stage A, 21 November 2026. This document records a literature and source review, not a reproduction or an implementation. No real training, model evaluation, reference-code execution, or label acquisition was performed.

## Scope and source identity

Read the complete `docs/MASTER_CODEX_PROMPT.md`, all 12 pages of `papers/flexible_sampling_long_tailed_skin_lesion.pdf` (10 main-paper pages and 2 supplementary pages), and the supplied Flexible Sampling modules listed below. Paths in this audit are relative to the separate `reference_resources/ISIC2019_AL_Architecture_Resource_Pack/` directory.

Paper: Lie Ju, Yicheng Wu, Lin Wang, Zhen Yu, Xin Zhao, Xin Wang, Paul Bonnington, and Zongyuan Ge, **Flexible Sampling for Long-tailed Skin Lesion Classification**. The supplied PDF identifies itself as arXiv:2204.03161v2, 28 May 2025. The supplied repository's citation identifies the original publication as MICCAI 2022, pp. 462-471, DOI `10.1007/978-3-031-16437-8_46`. These dates describe the supplied revision and cited original publication respectively; they are not competing claims about one version.

The repository README calls its implementation official. The supplied archive has no commit identifier or signed provenance. Treat that wording as a repository claim, not independent verification of author provenance. No results reported in that README have been reproduced here.

## Evidence from the paper

| Location | Content | Consequence for CSDA-AL |
|---|---|---|
| Section 2.1, PDF pp. 3-4, Eq. (1) | Contrastive representation learning without semantic supervision. | Supports label-free SSL features, but does not prove those features are balanced on ISIC 2019. Use the separate original PyTorch SimCLR module. |
| Section 2.2, p. 4, Eq. (2) | The class prototype averages embeddings belonging to a known true class. | Requires labels for the full set used to calculate prototypes. Cannot initialize a zero-label AL pool. |
| Section 2.2, pp. 4-5, Eq. (3) | Select samples nearest their corresponding known class prototype; the anchor distribution uses an imbalance scaling factor. | This is labeled curriculum subset construction. Replace it with SUPClust in CSDA-AL. |
| Section 2.3, p. 5, Eq. (4) | `p_c = 1 - a_e(c)`, normalized to sum to the number of classes. | Adopt the class-difficulty principle. Its class-conditioned accuracy is numerically class recall for a single-label classifier. |
| Section 2.3, pp. 5-6, Eq. (5) | Mutual information is predictive entropy minus expected conditional entropy. | Optional BALD uses the positive main-text formula. |
| Section 2.3, p. 6 | Newly sampled counts are class-specific, requiring a class membership assignment before querying. | The original formulation is not a label-free new-annotation policy. CSDA-AL must use prediction probabilities rather than hidden membership. |
| Section 3.1, p. 6 | ISIC-2019-LT is a constructed Pareto long-tail subset; an additional experiment uses a 14-class archive extension. | Main CSDA-AL target remains original eight-class ISIC 2019. These split files and class distributions are not default dataset contracts. |
| Section 3.2, p. 6 | Uses 224-pixel inputs, ResNet-18 for ISIC-2019-LT and ResNet-50 for the 14-class experiment, Adam, warm-up, and plateau-triggered querying. | Architecturally relevant, but CSDA-AL will use fixed annotation budgets and separately configurable training schedules. |
| Supplementary document, PDF p. 12 | The pseudocode calculates `entropy2 - entropy1`. | This is the negative of Eq. (5). It must not be copied literally and ranked descending as positive BALD. |

Important metric distinction: the expression "validation class accuracy" in the paper means correct predictions divided by true examples of that class, the same quantity as recall. Selecting recall is an explicit operational definition for CSDA-AL; changing the candidate membership from true class buckets to predicted probabilities is the substantive new adaptation.

## Supplied code audit

Line numbers refer to the unchanged supplied source.

| Source / symbols | Findings |
|---|---|
| `reference_code/flexible_sampling/flexsampling/core/anchor.py:32-58`, `extract_features` | Returns features **and all labels**. The feature extraction path therefore cannot be given to an acquisition strategy. Its generic spatial reduction at lines 54-55 does not correspond to pooling NCHW spatial dimensions in the usual case. |
| `anchor.py:60-68`, `compute_prototypes`; `anchor.py:70-134`, `select` | Uses true labels to compute prototypes, counts, and class-specific sample indices. The count compression caps each class relative to a minimum count, rather than reproducing the paper's exact class-rank Pareto expression. This is a source-to-paper difference, separate from the AL label-access problem. |
| `flexsampling/core/curriculum.py:30-54`, `compute_class_probabilities` | Implements Eq. (4)'s normalization, but missing dictionary keys become accuracy zero (lines 44-46). Missing validation support is thereby silently treated as maximum difficulty. CSDA-AL must retain support and missingness explicitly. |
| `curriculum.py:56-100`, `query` | `all_labels` is required; lines 76-80 bucket unseen candidates by true class. Directly using this code for new-label acquisition would leak hidden labels. Per-class count calculation at lines 88-93 does not guarantee a global batch budget, and `max(1, ...)` forces at least one sample even at zero class difficulty. The nearby comment and actual expression also differ concerning division by the number of classes. |
| `flexsampling/core/pipeline.py:120-130`, `_get_all_labels` | Eagerly reads every training label before selection. This is valid for its labeled curriculum task, but incompatible with the CSDA-AL oracle contract. |
| `pipeline.py:179-211`, `_evaluate` | Computes true-class recall while calling it per-class accuracy, then averages over all configured classes; unsupported classes contribute zero. CSDA-AL will distinguish supported-class balanced accuracy, fixed-eight macro reporting, and missing support. |
| `pipeline.py:222-242`, `_do_query` | Scores the full labeled dataset and passes `_all_labels` to curriculum selection. No new-label oracle or unique annotation ledger is implemented. |
| `pipeline.py:261-301` | Runs SSL, label-dependent anchor selection, and shape-matched encoder weight transfer. CSDA-AL requires explicit encoder compatibility and feature identity rather than silently transferring whichever state-dict keys match. |
| `pipeline.py:303-395` | Defaults to SGD, optional class-aware minibatches and adaptive class-balanced focal loss; queries on plateaus or a fixed epoch interval. These choices differ from the main paper's stated Adam/plain-CE warm-up settings. The function does not reinitialize its optimizer after each query although the paper says it does. These are reference implementation choices, not required CSDA-AL behavior. |
| `flexsampling/core/bald.py:129-139` | Uses `H(mean p) - mean H(p)`, agreeing with main-text Eq. (5), unlike the supplementary sign. |
| `bald.py:22-43`, `_inject_dropout` | Injection only recognizes `nn.Sequential` objects ending in activation modules. Standard torchvision ResNet-18 residual stages contain BasicBlocks rather than ending in a bare activation; the routine may inject nothing and yield deterministic, near-zero BALD. Adding dropout only at inference is also a modeling assumption, not equivalent to training a dropout model. |
| `bald.py:95-150`, `score` | Disables BatchNorm updates via `eval()` and then enables dropout. It changes dropout rates temporarily but lacks exception-safe cleanup, seed isolation and exact restoration of the caller's mode. CSDA-AL should use explicit configured dropout, keep BatchNorm frozen, isolate RNG, and restore all module modes in `finally`. |
| `flexsampling/core/ssl_pretrain.py:20-41` | Uses aggressive cropping, jitter, grayscale and blur. These can remove lesion borders or change diagnostic color; they need explicit documented controls rather than adoption as clinical defaults. |
| `ssl_pretrain.py:63-75` | The generic fallback returns the identical already-transformed tensor twice. That is not an independently sampled two-view SimCLR transformation. CSDA-AL must require raw-image access and independently generate both views. |
| `ssl_pretrain.py:93-112`, `nt_xent_loss` | Correctly masks diagonal and selects cross-view positive indices, assuming normalized projections supplied by the caller. CSDA-AL should validate temperature, shape, finiteness and useful batch size. |
| `flexsampling/data/isic.py:72-79` | Loads pickled NumPy objects and silently skips filenames absent from a label dictionary. CSDA-AL should parse explicit CSV columns and fail on missing records; reference NumPy objects will not become runtime input. |
| `data/isic.py:88-108` | Dataset exposes `.labels`, and `__getitem__` returns truth for every index. CSDA-AL must separate unlabeled image access from oracle-revealed labeled datasets. |
| `examples/configs/isic_8class.yaml:10-13` | Uses a ResNet-50, whereas the paper uses ResNet-18 for its eight-class experiment. This is not a justification to change CSDA-AL's required eight-class ResNet-18. |
| `README.md:23-27` | Claims a reproduction using balanced accuracy, ResNet-50 and three seeds. Paper main-text Table 1 uses top-1 accuracy, ResNet-18 and five trials. Those are not directly comparable experimental protocols, and no reproduction claim is adopted. |

## Label order and provenance

The inspected source/configuration declares `num_classes=8`, but does not declare a diagnostic-name-to-index map. Its `dic.npy` is a pickled mapping; it was not deserialized or used to infer labels or class order. Therefore no correspondence between its numerical labels and CSDA-AL's fixed order has been established. CSDA-AL will define `MEL, NV, BCC, AK, BKL, DF, VASC, SCC` and map CSV headers by name, never by arbitrary column position. An `AKIEC` alias may only be accepted through an explicit documented mapping to `AK` when the source schema requires it. The 14-class extension and supplementary subtype terminology do not change this eight-class contract.

## Proposed CSDA-AL adaptation (new research component)

1. Cold-start selection uses SUPClust with **zero acquired training labels**. No class-level difficulty exists at this point and none is calculated.
2. Reserve a disjoint validation set using label-free identity/group splitting. Validation examples cannot be queried or used for SSL. Reveal their labels only through an explicit validation-label provision, record the number of unique paid validation labels separately, and evaluate the first trained classifier after initial annotation. A preexisting externally labeled validation set still has a disclosed label cost and provenance. A genuinely zero-total-label claim cannot include such a set for free.
3. From validation labels and predictions compute support `n_c`, correct count `t_c`, recall `r_c=t_c/n_c` for supported classes, and `d_c=1-r_c`. Pass only this aggregate difficulty and its support metadata to acquisition. Never pass validation labels, training truth, oracle, or a label-bearing dataset in `QueryContext`.
4. The proposed default uses explicit prior smoothing `d_c=(n_c-t_c + lambda*d0)/(n_c+lambda)`, with `lambda=1` and `d0=0.5`. Missing validation support consequently receives prior difficulty `0.5`, with an `unsupported`/prior-used flag; a strict configuration may reject missing support instead. A raw supported-class `1-recall` mode remains configurable. This policy and its prior are CSDA-AL assumptions, not paper equations. Report support/missingness and smoothing settings. If no validation labels are available, explicitly switch the configured strategy to entropy rather than silently inventing a difficulty vector.
5. For unlabeled predictions `p(c|x)`, calculate `D(x)=sum_c p(c|x)*d_c`; no hard true-class bucket is constructed. Normalize with the predeclared rule in the final design. If candidate min-max normalization is chosen, zero range must produce a constant zero contribution and a diagnostic; nearly equal inputs must use a documented tolerance to avoid amplifying numerical noise.
6. Combine normalized predictive entropy `U(x)=H(p(x))/log(8)` with normalized difficulty, `score=alpha*U+(1-alpha)*D_norm`. Fix alpha and policies before a run; use deterministic descending scores and image-ID tie breaks. Optional diversity reranking must only use features and predicted scores.
7. Annotation is performed only after IDs have been selected and validated. The true composition of the returned annotation batch can be logged for retrospective bias diagnostics, but cannot revise the current selection or become a hidden class-frequency input to the strategy.

`D(x)` estimates difficulty through the current model's class probabilities. It may miss rare-class images predicted as a majority class, and uncertainty does not guarantee rarity, clinical severity, or labeling usefulness. The proposed score is experimental; no improvement or label-efficiency claim has been established.

## Licensing and implementation boundary

Recursive filename inspection found no `LICENSE`, `LICENCE` or `COPYING` file in the supplied Flexible Sampling tree. `README.md:168-170` says MIT, and `pyproject.toml:10` contains `license={text="MIT"}`. These are metadata assertions rather than a bundled full permission/copyright notice; they do not establish complete reuse provenance. Follow the master prompt: retain the tree unchanged as read-only research material, cite the paper and paths, and write original CSDA-AL code from mathematical descriptions. Do not redistribute its sources, split arrays, data CSV, results table, or paper in the final project ZIP. No external legal conclusion or reuse permission is claimed by this audit.

## Required later tests

- Hand-computed true-class recall/difficulty with uneven supports and all eight fixed classes; unsupported-class flags and configured prior; optional smoothing and bounds.
- Identical candidate predictions and constant difficulty produce deterministic scores/ties; no division by zero; alpha endpoints reproduce difficulty-only and entropy-only ranks.
- Invalid probabilities, nonfinite values, wrong class order, row/ID misalignment, negative support and stale model/difficulty fingerprints are rejected.
- Changing sealed training labels without changing inputs leaves selected IDs and scores unchanged; selection objects cannot reach label-bearing datasets or the oracle.
- Zero acquired-label round invokes only the label-free strategy; validation-label availability never changes the cold-start query.
- Training query count, paid validation count and total unique annotation count remain separate; validation/test IDs cannot enter candidates.
- Optional BALD: identical MC predictions give zero; disagreeing confident samples give positive mutual information; repeat seed reproduces scores; BatchNorm running statistics and original module modes are restored after success and exceptions. Standard no-dropout ResNet must produce an explicit unsupported-configuration error rather than silently fabricated uncertainty.
- Budget exactness when feasible, final partial batch, depleted pool, uniqueness and deterministic ID ties, independent of class counts.
- Retrospective batch composition runs after annotation and cannot affect the current query.

## Actual review commands and outcomes

- `python -` with `Path.read_text` printed the complete master prompt and PDF skill: PASS.
- `python -` with `pypdf.PdfReader` extracted and read every page of the supplied PDF: PASS, 12/12 pages.
- `python -` with numbered `Path.read_text(...).splitlines()` inspected README, pyproject, anchor, curriculum, BALD, SSL pretrainer, pipeline, dataset and eight-class YAML: PASS. A second read of BALD/pipeline supplied complete output after the first batch was output-truncated.
- `rg -n 'NV|MEL|BCC|BKL|VASC|AK|class_names|class_to|label_map|class_map|mapping|seed|manual_seed' ... -g '*.py' -g '*.yaml' -g '*.toml' -g '*.md'`: no explicit class-name map found in the inspected source/config set; examples contain manual seeding.
- Recursive license/copying filename check with `Path.rglob`: PASS, no matching file found.
- Rendering availability: system `pdftoppm` absent; `pypdfium2` available. Rendered complete PDF pages 5 and 12 using `pdfplumber.pages[n].to_image(resolution=110)` and visually inspected both PNGs: PASS; Eq. (4)/(5) and the reversed supplementary pseudocode were legible and confirmed. Images are review intermediates under `tmp/pdfs/`, outside the deliverable.

No unit tests or synthetic model experiments were run in this sub-audit. It is Stage A source analysis, with implementation and executable tests reserved for approved later stages.
