# Source integrity and environment audit

Historical Stage A audit snapshot, before B-F implementation. Audit date: 21 November 2026 (Asia/Calcutta). Current tested environment and implementation status are in the completion report. The complete master prompt was read before design. The source map, blueprint, attribution notes, all three papers and relevant source modules were reviewed. Three focused parallel audits supplied page/equation and code-symbol evidence; no stage implementation was delegated or begun.

## Extracted resources

Input: `C:\Users\Dishitha Ramesh\Downloads\ISIC2019_AL_All_Resources_and_Codex_Prompt.zip`.

Extracted to the separate workspace directory `reference_resources/ISIC2019_AL_Architecture_Resource_Pack/`. Archive member paths were resolved and checked against the extraction root before extraction. The archive has 241 file entries. The supplied `RESOURCE_MANIFEST.csv` lists 240 payload files; every file size and SHA-256 matched. The additional file is the manifest itself. The original ZIP and extracted reference contents were not edited.

The four separately mentioned ZIPs correspond exactly to these bundled snapshots:

| Standalone ZIP | Bundled reference tree | Compared files | Mismatches / unbundled files |
|---|---|---:|---:|
| `imbalanced-medical-active-label-cleaning-main.zip` | `active_label_refinement` | 12 | 0 / 0 |
| `FlexSampling-main.zip` | `flexible_sampling` | 44 | 0 / 0 |
| `TypiClust-main.zip` | `typiclust` | 137 | 0 / 0 |
| `simclr-master.zip` | `simclr_google_research` | 35 | 0 / 0 |

This verifies duplicate snapshot correspondence, not upstream authorship, a repository commit, current maintenance status or research results.

The separately attached `CODEX_MASTER_PROMPT_ISIC2019_ARCHITECTURE_ONLY.md` matches the bundled master byte for byte: SHA-256 `0919fee9da3b629e8c3ff3e88b16e5b6fbd2c54b95e4b7eb76fb43d034d2e1d5`.

## Paper completeness and identity

| Supplied PDF | Pages read | Extracted text characters | Identity verified from content |
|---|---:|---:|---|
| `supclust_active_learning_at_boundaries.pdf` | 7 / 7 | 21,020 | Ono et al., SUPClust, ICLR 2024 workshop, arXiv:2403.03741v1 |
| `flexible_sampling_long_tailed_skin_lesion.pdf` | 12 / 12 | 27,043 | Ju et al., Flexible Sampling, supplied arXiv:2204.03161v2; cited original MICCAI 2022 publication |
| `active_label_refinement_imbalanced_medical.pdf` | 13 / 13 | 29,249 | Khanal et al., Active Label Refinement, MICCAI 2024, arXiv:2407.05973v3 |

All 32 pages, including references and supplements, were read across the focused audits. Formula/method pages were rendered and visually checked: SUPClust pages 2-4; Flexible Sampling pages 5 and 12; Refinement pages 4-5. System Poppler was unavailable; installed PDFium/pdfplumber rendered these pages. Intermediate PNGs remain outside the project. No supplied paper or source result table has been presented as CSDA-AL evidence.

## Important audit findings

- SUPClust starter formulas agree with the paper's center weights and SUP definition, but the starter rejects a single cluster and fails on all-identical embeddings. Paper defaults for several geometry details are incomplete. Our explicit fallback/settings are engineering extensions.
- TypiClust's FAISS path computes squared L2; the paper's typicality formula uses Euclidean distance. Its nested SCAN license is CC BY-NC 4.0, despite MIT notices at other levels.
- Flexible Sampling's prototypes and class buckets use full true labels. Its validation class accuracy equals true-class recall. The unlabeled membership/score fusion is the proposed CSDA-AL adaptation. Supplementary BALD pseudocode reverses the sign of the main-text equation.
- Active Label Refinement is a noisy-label correction task. Source patterns exposing targets to candidate loaders and repeatedly evaluating test data will not be reproduced in the primary engine.
- Google SimCLR is TensorFlow-based. Its sum of directional mean losses is twice the planned conventional all-anchor mean. A fresh PyTorch implementation will make the reduction explicit.
- A standard ResNet-18 has no stochastic dropout head by default; BALD must be explicitly supported, not simulated by toggling the entire network into training mode.

See the three [detailed audit reports](audits/SUPCLUST_AUDIT.md) and [traceability map](RESEARCH_TRACEABILITY.md) for exact source locations. License findings are consolidated in [LICENSE_NOTES](../LICENSE_NOTES.md).

## Environment observation, not dependency validation

The shell resolves `python` to an existing unrelated environment at `C:\Users\Dishitha Ramesh\recipe_pipeline_from_scratch\.venv\Scripts\python.exe`. It reports Python 3.13.9 (Anaconda build). Observed installed distributions: torch 2.12.0, NumPy 2.4.6, scikit-learn 1.9.0, SciPy 1.17.1, pandas 3.0.3, Pillow 12.2.0, PyYAML 6.0.3, pytest 9.1.1, pdfplumber 0.11.9 and pypdf 6.21.4. **torchvision is not installed**. `uv` is available; `py`, `pdftoppm` and `pdfinfo` were not found on PATH.

The reference tests use this existing CPU environment. It does not satisfy the required Python 3.11/torchvision project stack. Stage B will establish an isolated Python 3.11 environment and record compatible tested pins; Stage C will verify actual torch/torchvision model operations. No packages or downloaded model weights were installed during Stage A.

## Completed checks

The root audit ran the supplied SUPClust starter suite read-only with bytecode and pytest cache disabled: **3 passed in 5.65 seconds**, exit code 0. A parallel audit also ran the same three tests; this is repeated source verification, not six different tests. Source edge probes exposed singleton/duplicate failures as recorded in the SUPClust audit. No integrated CSDA-AL tests exist yet.

Pure NumPy formula arithmetic verified SUP weighting and the proposed entropy/difficulty score on artificial inputs. It did not invoke a trainer, model, oracle, image dataset or integrated selector. See [formula output](STAGE_REVIEWS/stage_a_formula_checks.json), [reference-test log](STAGE_REVIEWS/stage_a_reference_tests.log), [machine-readable integrity record](STAGE_REVIEWS/stage_a_evidence.json), and [review/check commands](STAGE_REVIEWS/stage_a_commands.txt).

Official ISIC dataset scope and attribution were additionally checked against [ISIC data](https://challenge.isic-archive.com/data/) and [2019 challenge](https://challenge.isic-archive.com/landing/2019/). The SimCLR citation was checked against [the primary PMLR publication](https://proceedings.mlr.press/v119/chen20j.html). These sources clarify closed-set scope and method attribution; no real data or checkpoints were downloaded.
