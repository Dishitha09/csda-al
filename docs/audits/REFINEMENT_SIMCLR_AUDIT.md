# Stage A: Active Label Refinement and SimCLR source audit

Status: design evidence only; no integrated runtime modules or model training were executed. Audited 21 November 2026. All source paths below are relative to the unchanged extracted `ISIC2019_AL_Architecture_Resource_Pack/`. Line numbers refer to that supplied snapshot, not to a current online repository.

## Paper identity and complete reading record

The supplied `papers/active_label_refinement_imbalanced_medical.pdf` is **Active Label Refinement for Robust Training of Imbalanced Medical Image Classification Tasks in the Presence of High Label Noise**, by Bidur Khanal, Tianhong Dai, Binod Bhattarai, and Cristian Linte. The first page identifies arXiv:2407.05973v3, 24 October 2024; the supplied repository README identifies it as their MICCAI 2024 paper. All 13 PDF pages were read, including references and supplementary Sections 6-8. Metadata alone does not contain a title, so identity was checked from page 1. The source provides the repository URL <https://github.com/Bidur-Khanal/imbalanced-medical-active-label-cleaning.git>.

The audit used `pypdf.PdfReader` to extract all pages. Pages 4 and 5 were additionally rendered with installed `pypdfium2` at scale 1.6 and visually inspected, confirming Eq. (1) and the Section 2.4 oracle/budget equations. `pdftoppm` and `pdfinfo` were not found on PATH. Temporary render files are outside the project at `tmp/pdfs/refinement_audit/page_4.png` and `page_5.png`.

### Evidence relevant to CSDA-AL

| Source location | Actual contribution | CSDA-AL use and boundary |
|---|---|---|
| pp. 1-3; Fig. 1 | Two-phase learning with noisy labels followed by expert label correction | Background for modular acquisition, rather than an algorithm for zero-label cold start |
| Section 2.1, p. 3 | Inject synthetic noise into already labeled data | Excluded from the primary project |
| Section 2.3, pp. 4-5; Eq. (1) | Feature-level VOG over epochs combined with small-loss Co-teaching selections | Excluded: no Co-teaching, VOG history, noise-rate estimator, or noise-correction branch in the primary runtime |
| Section 2.4, p. 5 | Rank a residual noisy set, query `a_l` examples, use an oracle to clean labels, update clean/noisy sets, retrain until total budget `A_l` | Adapt the round/budget/oracle structure to disjoint labeled/unlabeled TRAIN pools; this changes the task and must be declared |
| Sections 3.1-3.2, pp. 5-6 | ISIC 2019 with eight classes; reported random image splits and macro-F1 evaluation | Eight-class task and class-level reporting are relevant; supplied random split is not evidence of patient/lesion isolation |
| Section 3.3, pp. 6-7 | ResNet-18, ImageNet initialization, standard supervised optimization | Architecture precedent, not a requirement to download weights or adopt its noise settings |
| Section 5, p. 9 | Limits on model and noise distributions | No claim that the paper validates CSDA-AL or our difficulty-weighted acquisition |
| Supplement Section 8, pp. 12-13 | Contrasts reuse of LNL model and retraining a model on guessed-clean data | The prose/caption Strategy I/II labels are inconsistent. This has no effect on our model path because LNL is excluded |

Section 2.4 uses an unnumbered selection expression over the noisy subset and an oracle call. Its logic transfers as a software pattern; neither its noisy subset nor its guessed-clean labels can become our unlabeled pool. Reported paper outcomes are not CSDA-AL outcomes and are intentionally not reproduced here.

## Reference implementation inspection

Read the complete `reference_code/active_label_refinement/main_active_learning.py` (888 lines), `active_learning_strategies.py` (156 lines), `coreset.py` (52 lines), and `solver.py` (396 lines). Inspected `LNL.py` class definitions and relevant training/VOG scope, plus dataset and metric helpers in `utils.py`, README, and LICENSE.

| Source location | Observation | Required original implementation decision |
|---|---|---|
| `main_active_learning.py:119-120`, `736-740` | Saves true training labels before corruption and later directly replaces noisy targets | Our selector never receives a dataset with a `.targets` truth array. Only `LabelOracle.annotate(selected_ids)` releases TRAIN labels |
| `main_active_learning.py:179-253`; `LNL.py:15`, `386`; `solver.py:135`, `278`, `318` | Dual-network noisy-label training and gradient instrumentation are central to this reference | No import of LNL, BackPACK, gradient-cleaning code, or copied solver into the primary package |
| `main_active_learning.py:301-410` | Round-based dispatch to random/entropy/Core-Set and cumulative cleaned subset | New strategy protocol, explicit ID-keyed pool state, budget clamping at engine boundary, no noisy-label state |
| `active_learning_strategies.py:9-33`, `58-84` | Candidate prediction/embedding loaders yield `(images, target, index)` even when targets are unnecessary | Separate unlabeled image loader returns image IDs and images only. Prediction and feature outputs carry IDs explicitly |
| `active_learning_strategies.py:87-94` | Sorts negative entropy ascending, which is equivalent to high entropy first | Implement positive normalized entropy and descending score with lexical ID tie resolution; test zeros safely |
| `active_learning_strategies.py:36-55` | MC prediction uses `model.train()` | Never use global train mode for MC dropout: freeze BatchNorm running statistics and enable only explicit dropout layers temporarily, restoring all prior modes |
| `active_learning_strategies.py:113-120` | Stores `E[H(p_t)] - H(E[p_t])` and sorts ascending; this sign/sort combination selects largest BALD | Implement conventional nonnegative `H(E[p_t]) - E[H(p_t)]` and descending order, with finite/roundoff checks. The reference sign itself is not an acquisition reversal |
| `main_active_learning.py:185-196`, `260-266` | Standard torchvision ResNet-18 is used without adding a dropout head | MC dropout is ineffective without an explicitly configured dropout component. If BALD is offered, require dropout probability above zero and multiple passes; document the optional head change |
| `active_learning_strategies.py:58-84` | Embeddings are obtained using a named hook and positional concatenation | Prefer explicit `encode(images) -> [N,512]`, deterministic eval transform, no hook assumptions, and ID alignment |
| `coreset.py:16-49` | Initial distance update is attempted even for an empty selected list; selected list is not extended during the loop; duplicate/zero-distance and tiny-pool behavior lack safeguards | Build original greedy minimum-distance selection with an explicit empty-seed policy, candidate mask, per-addition update, chunking, finite validation, and no repeated ID |
| `main_active_learning.py:368-393` | Acquisition loaders read `train_dataset` with the training augmentation | Acquisition inference must use deterministic transforms even while supervised training uses augmentation |
| `main_active_learning.py:420-435`, `508-510`; `LNL.py:875-898` | Test set is inferred/logged repeatedly during training; checkpoint choice uses validation macro-F1 | In CSDA-AL the engine has validation access only; test evaluation requires a locked final-evaluation request. Repeated visibility creates human tuning risk even if test metrics do not directly select the checkpoint |
| `main_active_learning.py:475` | `(epoch - recalibration_epochs) <= 5` holds for all ordinary epochs | Do not copy the purported last-five-epoch summary; explicit bounded history only if needed |
| `utils.py:43-58`, `65-82`; `main_active_learning.py:582` | Dynamic confusion-matrix label sets and unguarded division/AUROC can mis-handle missing classes | Fix label order to eight, always report support masks, define zero-division policy, and mark undefined metrics explicitly |
| `main_active_learning.py:326`, `424` | Cosine scheduler has `T_max=label_cleaning_rounds` but steps every recalibration epoch | Separate epoch scheduler duration from annotation-round count and persist scheduler state |
| `main_active_learning.py:674-677`, `787-790` | Checkpoints are written directly; Neptune setup is mandatory at entry | Atomic versioned checkpoints and local logger as default; remote logging optional behind an interface |

These are static inspection findings, not claims from running the old implementation. Source comments also say acquisition/Core-Set code was adopted from other projects (`active_learning_strategies.py:7`, `coreset.py:5`); do not assume a top-level license resolves all possible third-party provenance.

## SimCLR source audit

The supplied SimCLR repository is Google Research's TensorFlow implementation, with an additional `tf2/` implementation. It is a research reference; CSDA-AL's runtime will use only PyTorch/torchvision. Relevant inspected sources: root `objective.py`, `model.py`, `model_util.py:141-177`, `data_util.py` color/crop/blur/train/eval functions, `resnet.py:341-401` and `531-706`, README, and LICENSE. No supplied pretrained checkpoint was downloaded, converted, or executed.

README `200-210` identifies the foundational reference: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton (2020), **A Simple Framework for Contrastive Learning of Visual Representations**, arXiv:2002.05709. README `213-220` also identifies SimCLRv2, which must not be silently substituted for the requested two-layer SimCLR architecture. The original SimCLR PDF is not one of the three supplied PDFs; the citation and implementation are supplied in the repository.

| Source location | Evidence | PyTorch design |
|---|---|---|
| `objective.py:34-88` | L2 normalization, temperature-scaled similarities, self-mask, positives across paired views, both directions | Original stable NT-Xent using concatenated `[z_view1,z_view2]`; positive index `(i+N) mod 2N`; diagonal masked to negative infinity |
| `objective.py:82-86` | Adds the mean directional losses, `loss_a + loss_b` | Use the standard mean over all `2N` anchors: half this scalar for identical logits. Document reduction so a factor-of-two learning-rate effect is not mistaken for a correctness error |
| `model.py:42-58`, `70-78` | Two transforms, shared encoder, projection for contrastive objective | Two independently sampled transformations of each source image, one shared encoder, two-layer MLP projection |
| `model_util.py:141-177` | Projection is separable from backbone representation; nonlinear configuration supports varying depth | Required head is `512 -> hidden_dim -> projection_dim` with ReLU; default projection dimension 128. No need to inherit all SimCLRv2 options |
| `resnet.py:670-679`, `684-706` | Global average pooling; ResNet-18 has four `[2,2,2,2]` groups and 512 output features | torchvision ResNet-18 with `fc=Identity`; expose `[N,512]` backbone features before projection |
| `model.py:62-67` | Linear evaluation freezes BatchNorm statistics | Head-only classifier training freezes encoder parameters and keeps encoder/BatchNorm in eval mode |
| `data_util.py:61-83`, `304-325`, `387-414` | Strong color jitter, grayscale, random crops down to 8% area, blur | Do not adopt ImageNet augmentation strength unchanged: mild lesion-oriented transforms and explicit risky-transform configuration |
| `data_util.py:447-519` | Training and deterministic evaluation preprocessing are separate | Persist transform identity in feature cache; cache/extraction use deterministic acquisition preprocessing |
| README `194-198` | Batch size and pretrained weight scales affect real representation performance | Small CPU tests establish operational correctness only, not useful dermoscopic representation quality |

The original loss accepts class labels elsewhere for supervised evaluation, but the PyTorch SSL data path must not construct or consume target diagnostic labels. Positive pair identities are image-instance identities, not class identities. False negatives across different images of the same lesion/class are a research limitation, not permission to access labels during SSL.

### NT-Xent specification for later Stage C

For `N >= 2` images, let `z_1,...,z_2N` be normalized projections, `j(i)` the paired view, `tau > 0`, and `s(i,k)=z_i^T z_k / tau`.

`ell_i = -s(i,j(i)) + logsumexp({s(i,k): k != i})`

`L = sum_i ell_i / (2N)`.

Validate matching view shapes, at least two distinct instances in the batch, finite inputs and temperature, and stable float32 accumulation for mixed precision if later enabled. Reject the batch-size-one case because it has no negative instance and its loss degenerates to zero. One CPU optimizer step on synthetic images verifies gradients; it cannot establish the representation's clinical relevance. Extraction returns encoder features, not normalized projections. Optional L2 normalization of extracted features is an explicit, versioned feature-space choice.

### Classifier/fine-tuning and cache implications

The classifier emits eight logits in the fixed order `MEL,NV,BCC,AK,BKL,DF,VASC,SCC`; `AKIEC` aliases, if supported at ingestion, are mapped explicitly to `AK`. A learned SSL encoder can initialize the classifier by copying its backbone state. Frozen-linear mode preserves backbone weights/statistics. Fine-tuning uses declared stages and separate backbone/head optimizer groups. Rebuilding groups must preserve or deliberately reset optimizer state with an audit event. Unseen classes in a small labeled subset remain real output categories with no guarantee of learned discrimination.

Use a frozen SSL feature store for reproducible initial clustering and optional SSL diversity, separate from classifier embeddings used by a Core-Set baseline. A cache key includes encoder state digest, architecture, preprocessing fingerprint, feature normalization, ordered IDs, and schema version. Fine-tuning invalidates classifier-feature caches. Checkpoints preserve model, optimizer/scheduler, round/pool/config fingerprints, RNG state, class order and data split digest. Atomic publication and schema validation are original infrastructure requirements rather than guarantees inherited from these repositories.

## Licensing findings

- `reference_code/active_label_refinement/LICENSE` is MIT, copyright 2024 Bidur Khanal. If substantive code were adapted, preserve the permission and copyright notice. The plan uses fresh code and cites the research; preserve the supplied reference tree unchanged.
- `reference_code/simclr_google_research/LICENSE` is Apache License 2.0; inspected source files have 2020 SimCLR Authors Apache headers. Redistribution of adapted source requires appropriate notices/license handling. The intended PyTorch implementation is original; cite SimCLR and retain any future direct-reuse attribution explicitly.
- No proposal here redistributes the supplied PDFs, Google checkpoints, old repositories, or external assets inside the final original architecture ZIP. A paper's availability does not establish a redistribution license.

## Required tests, to be implemented only after their stage gates

1. SSL: `[N,512]` encoder and projection dimensions; paired augmentation independence with fixed seed reproducibility; known-logit NT-Xent identity including reduction; finite backward/one optimizer step; batch-size-one/invalid-temperature rejection; save/reload equality; no labels in SSL loader.
2. Classifier: exactly eight logits; probability rows sum to one; labeled-only fit; head-only encoder parameters and BatchNorm buffers unchanged; staged unfreezing group coverage; deterministic embedding extraction and checkpoint round trip.
3. Acquisition: permutation/ID alignment checks; changing sealed hidden labels leaves selected IDs/scores unchanged; deterministic ties; invalid/non-TRAIN/repeated selection rejection; zero probabilities; Core-Set empty seed, duplicate features and exhausted pool; BALD finite values and BatchNorm unchanged.
4. Evaluation: known-array confusion matrix with eight rows/columns, explicit support masks, zero-support policy, no test evaluator in engine, final-evaluation lock enforcement, validation charge ledger independent of TRAIN acquisition ledger.
5. Integration: a few synthetic CPU steps only; resume at crash boundaries produces the same unique annotations, pool partition and next selection; no checkpoints pretending to be trained on ISIC.

## Actual commands and outcomes in this audit

- `python -` scripts using pathlib: read full master prompt, full paper extraction, and line-numbered supplied source reads: **PASS** (text extraction returned 13 pages).
- `rg -n` on source symbols and related dataset/metric definitions: **PASS** (static source navigation only).
- `Get-Command pdftoppm,pdfinfo -ErrorAction SilentlyContinue`: neither command found; used available PDFium rendering instead.
- `python -` with `pypdfium2.PdfDocument(...)[page].render(scale=1.6).to_pil().save(...)`: **PASS**, pages 4 and 5 rendered and visually checked.
- No pytest suite, training loop, dataset download, scientific evaluation, or source import requiring old dependencies was run in this design-only audit. Later-stage tests above are acceptance requirements, not completed tests.
