# CSDA-AL Implementation Audit

**Document ID:** CSDA-AL-AUD-001  
**Date:** 2026-11-22  
**Scope:** File-by-file inspection of `csda_al` repository vs. research papers and specifications

---

## 1. Classification Taxonomy

Each component is classified into one of six standard evaluation categories:
1. **Fully Implemented and Verified (FIV):** Implemented, unit/integration tested, verified mathematically against paper specifications.
2. **Implemented but Not Integrated (INI):** Component works standalone or unit tested, but not wired into the main active learning loop or CLI.
3. **Partially Implemented (PI):** Core functions exist, but key paper features, edge-case handlers, or options are absent.
4. **Implemented but Methodologically Incorrect (IMI):** Implemented in a way that violates active learning assumptions (e.g. data leakage, wrong objective, improper proxy).
5. **Missing (MIS):** Required by project scope or papers, but no implementation exists in the codebase.
6. **Experimentally Unvalidated (EU):** Code is architecturally sound and passes synthetic unit tests, but has not completed end-to-end execution on the actual ISIC 2019 dataset.

---

## 2. File-by-File Detailed Audit

### A. Dataset and Preprocessing (`src/csda_al/data/`)

| File / Component | Status | Classification | Audit Findings & Technical Evidence |
|---|---|---|---|
| `metadata.py` | Complete | **FIV** | Correctly defines the 8 diagnostic classes (`MEL`, `NV`, `BCC`, `AK`, `BKL`, `DF`, `VASC`, `SCC`) in strict canonical order. Implements class weights, label mappings, and strict schema validation. |
| `dataset.py` | Complete | **FIV** | Implements `ISICDataset`, PIL image loading, robust error handling, transform pipelines, and split integrity. |
| `prepared.py` | Complete | **FIV** | Validated against actual ISIC 2019 dataset (25,331 images). Verifies SHA-256 image hashes, partition manifests, and prevents leakage. |
| `split.py` | Complete | **FIV** | Strict 70% train (17,732) / 10% val (3,800) / 20% test (3,799) patient-stratified split with zero image or lesion ID leakage. Cryptographically fingerprinted (`1ef5b18099...`). |
| `duplicates.py` | Complete | **FIV** | Implements perceptual hashing (dHash/pHash) and exact match filtering across partitions. |
| `oracle.py` | Complete | **FIV** | Benchmark oracle seals ground truth training labels; exposes labels only upon committed query receipts. Prevents unlabelled pool snooping. |
| `transforms.py` | Complete | **FIV** | Standard ImageNet normalization ($224 \times 224$), SimCLR dual-view augmentations (color jitter, gaussian blur, random resized crops). |
| `synthetic.py` | Complete | **FIV** | Generates mock in-memory datasets for fast CPU unit testing without requiring the full 9 GB image archive. |

---

### B. Feature Extraction & Cold-Start Initialization (`src/csda_al/features/` & `acquisition/supclust.py`)

| File / Component | Status | Classification | Audit Findings & Technical Evidence |
|---|---|---|---|
| `models/simclr.py` | Complete | **FIV / EU** | ResNet-18 backbone with 2-layer MLP projection head (512 -> 512 -> 128) and NT-Xent loss. Verified on CPU synthetic fixtures. Validated real 5-step checkpoint exists (`outputs/isic2019/simclr.pt`). Full 100-epoch convergence experimentally unvalidated on CPU. |
| `features/extract.py` | Complete | **FIV** | Extracts 512-dimensional pen-ultimate feature vectors from the backbone. Enforces deterministic batch inference. |
| `features/cache.py` | Complete | **FIV** | In-memory and on-disk atomic caching of image representations keyed by model fingerprint and image ID. |
| `features/nearest_neighbors.py`| Complete | **FIV** | Exact and approximate (k-NN) neighbor search using cosine/Euclidean distance for typicality density estimation. |
| `acquisition/supclust.py` | Complete | **FIV / EU** | Faithful to Ono et al. (ICLR 2024 Workshop): K-Means clustering, cluster typicality calculation ($1 / \text{mean\_dist}$), boundary SUP scoring (Softmax Unsupervised Prediction), and occupied cluster penalty. Contains ablation variants (`without_SUP`, `without_typicality`). |
| `acquisition/typiclust.py` | Complete | **FIV** | Faithful to Hacohen et al. (NeurIPS 2022): typicality-based clustering without boundary uncertainty. |
| `models/backbone.py` | Partial | **PI** | Supports ResNet-18, but transfer learning initialization with ImageNet pretrained weights (`torchvision.models.resnet18(weights=...)`) was previously disabled in favor of scratch/SimCLR. |

---

### C. Active Learning Acquisition (`src/csda_al/acquisition/`)

| File / Component | Status | Classification | Audit Findings & Technical Evidence |
|---|---|---|---|
| `base.py` | Complete | **FIV** | Abstract base class `AcquisitionStrategy`, sealed `PredictionBatch`, `QueryBatch`, and strict query context contracts. Enforces zero label leakage into query methods. |
| `random.py` | Complete | **FIV** | Uniform random baseline (B0). |
| `entropy.py` | Complete | **FIV** | Standard Shannon entropy uncertainty sampling (M4). Verified mathematically. |
| `coreset.py` | Complete | **FIV** | Greedy k-Center facility location in feature space (M5). |
| `bald.py` | Complete | **FIV** | Bayesian Active Learning by Disagreement via Monte Carlo Dropout (M6). |
| `difficulty_aware.py` | Complete | **FIV / EU** | Adapted difficulty-aware acquisition: dynamically weights predictive entropy by class difficulty estimated from validation recall. Validated on synthetic tests, experimentally unvalidated on full ISIC. |
| `acquisition/bengar.py` | Missing | **MIS** | Bengar et al. (WACV 2022) Class-Balanced Active Learning is **missing** from `src/csda_al/acquisition/`. Need to implement the constrained/penalized acquisition balancing pseudo-label distributions. |
| `acquisition/flex_sampling.py`| Missing | **MIS** | Original Flexible Sampling (Ju et al., MICCAI 2022) anchor-prototype curriculum is only present as the adapted `difficulty_aware.py`; faithful original paper baseline module is missing. |

---

### D. Training Pipeline (`src/csda_al/training/` & `models/`)

| File / Component | Status | Classification | Audit Findings & Technical Evidence |
|---|---|---|---|
| `models/classifier.py` | Complete | **FIV** | ResNet-18 backbone with linear classification head (512 -> 8). Dropout option for BALD MC inference. |
| `training/supervised_train.py`| Complete | **FIV** | Supervised classifier training with Cross-Entropy / weighted Cross-Entropy, Adam/SGD, cosine annealing scheduler, gradient clipping, and checkpoint management. |
| `training/ssl_train.py` | Complete | **FIV** | SimCLR pretraining with NT-Xent loss. Verified end-to-end on real images. |
| `training/finetune.py` | Complete | **FIV** | Linear probing and full fine-tuning routines from SSL/pretrained weights. |
| `imbalance/class_difficulty.py`| Complete | **FIV** | Computes per-class learning difficulty from validation loss/recall trajectories ($\text{difficulty}_c = 1 - \text{Recall}_c$). |
| `imbalance/probability_weighting.py`| Complete | **FIV** | Implements class-frequency loss weighting and focal/effective-number adjustments. |

---

### E. Active Learning Engine & Stopping (`src/csda_al/engine/`)

| File / Component | Status | Classification | Audit Findings & Technical Evidence |
|---|---|---|---|
| `pool_state.py` | Complete | **FIV** | Immutable `PoolState` tracking labelled and unlabelled IDs, query round counts, and acquisition transitions. |
| `budget.py` | Complete | **FIV** | Manages initial budget (e.g. 80), batch budget (e.g. 40), and validation annotation overhead accounting. |
| `checkpointing.py` | Complete | **FIV** | Atomic state serialization, SHA-256 verification, and crash recovery. |
| `active_loop.py` | Partial | **PI** | Robust transactional round loop, but stopping rule is hardcoded to fixed budget exhaustion (`termination: "fixed_budget_round_limit_or_exhausted_pool"`). Class-aware early stopping is missing. |
| `engine/stopping.py` | Missing | **MIS** | Class-aware stopping criterion module is missing. |

---

### F. Evaluation Suite (`src/csda_al/evaluation/`)

| File / Component | Status | Classification | Audit Findings & Technical Evidence |
|---|---|---|---|
| `metrics.py` | Partial | **PI** | Computes Accuracy, Balanced Accuracy, Precision, Recall, F1 (macro/micro/weighted), per-class metrics, confusion matrix, head/middle/tail breakdown. **Missing Macro-AUROC** computation from probabilities. |
| `report.py` | Complete | **FIV** | Cryptographic validation locks, atomic JSON serialization, schema versioning. |
| `learning_curves.py` | Complete | **FIV** | Calculates Area Under Learning Curve (AULC), round-by-round trajectory aggregation, and budget-efficiency metrics. |
| Visualization scripts | Partial | **PI** | Plot generation routines exist in scripts, but publication-quality figure exporters for the 12 required graphs in `results/figures/` need final consolidation. |

---

### G. Specialized Pipelines

| Module | Status | Classification | Audit Findings & Technical Evidence |
|---|---|---|---|
| Khanal Noisy-Label Refinement | Missing | **MIS** | MICCAI 2024 Co-teaching + VOG pipeline for noisy labels is not present in `csda_al` (was intentionally omitted from main clean loop, but required as separate experimental module). |
| Bengar Class-Balanced AL | Missing | **MIS** | WACV 2022 implementation missing. |
| Flexible Sampling Full Baseline| Missing | **MIS** | Ju et al. original anchor + curriculum baseline missing as a standalone comparator. |

---

## 3. Action Plan for Phase 2 Implementation

To bring the project to 100% architectural and algorithmic completeness:
1. **Implement Bengar Class-Balanced Acquisition (`src/csda_al/acquisition/bengar.py`)**: Entropy uncertainty with pseudo-label distribution balancing constraints. Register in `make_strategy`.
2. **Implement Standalone Flexible Sampling Baseline (`src/csda_al/acquisition/flex_sampling.py`)**: Faithful reproduction of Ju et al. anchor selection & curriculum.
3. **Implement Class-Aware Stopping Criterion (`src/csda_al/engine/stopping.py`)**: Minority recall stability + macro-F1 plateau on validation data. Integrate into `active_loop.py`.
4. **Implement Macro-AUROC & ROC Evaluation (`src/csda_al/evaluation/metrics.py`)**: One-vs-rest macro-AUROC computation with scikit-learn.
5. **Implement Transfer-Learning Cold Start Initialization**: Support ImageNet pretrained ResNet-18 feature extraction in `features/extract.py` and `cli.py`.
6. **Implement Khanal Noisy-Label Refinement Extension (`experiments/noisy_label_refinement/`)**: Standalone module demonstrating Co-teaching + VOG under label noise.
7. **Create Experiment Configs (B0, B1, M1-M8, C1-C4)**: For reproducible benchmark runs across seeds 0, 1, 2.
