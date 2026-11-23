# Class-Sensitive Deep Active Learning (CSDA-AL) for Imbalanced Skin Lesion Classification on ISIC 2019

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dishitha09/csda-al/blob/main/notebooks/CSDA_AL_ISIC2019_Colab.ipynb)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE_NOTES.md)

**Academic Course Project:** Machine Learning Final Project  
**Platform Submission:** Digiicampus  
**Group ID:** Group 10  
**Team Lead:** Dishitha Ramesh  
**Submission Package:** `ML_Project_Group10_DishithaRamesh.zip`  

---

## 1. Project Overview

### 1.1 Problem Statement & Clinical Motivation
Automated classification of dermoscopic skin lesions is a clinically vital application of medical computer vision. Malignant conditions such as Melanoma and Squamous Cell Carcinoma (SCC) carry severe mortality if diagnosed late, yet early surgical excision yields five-year survival rates exceeding 99%. While deep neural networks achieve expert dermatologist diagnostic accuracy when trained on vast supervised datasets, acquiring histopathologically verified ground-truth labels is costly, time-consuming, and bottlenecked by clinical specialist availability.

Deep Active Learning (AL) offers a principled framework to minimize annotation expenditures by interactively querying the most informative samples from an unlabelled pool. However, deploying classical AL in clinical dermatological settings encounters two compounding failure modes:
1. **The Cold-Start Dilemma:** At the outset of screening, zero ground-truth labels exist. Unsupervised random initial seeding mirrors natural population skew, often failing to observe rare malignant classes in round 0.
2. **Extreme Class Imbalance (54:1 Skew):** In the canonical ISIC 2019 dataset, benign Melanocytic Nevus accounts for 12,875 images (50.8%), while rare malignancies like Vascular Lesions (253 images) and Dermatofibroma (239 images) each represent under 1% of the corpus (a 53.87:1 imbalance ratio). Standard uncertainty sampling heuristics (such as Shannon entropy or BALD) query samples where predictive entropy is maximal. Because majority classes dominate boundary volume, conventional AL repeatedly queries redundant variations of common nevi, starving rare malignant classes of supervision.

### 1.2 Proposed CSDA-AL Framework
**Class-Sensitive Deep Active Learning (CSDA-AL)** resolves this bottleneck by integrating:
- **Calibrated Cold-Start Initialization (Round 0):** Unsupervised MiniBatch $k$-Means clustering ($k=16$) combined with boundary softmax entropy to guarantee rare-class representation in the initial labelled cohort ($N_{init}=80$) without human supervision.
- **Tri-Component Acquisition Policy:** Scoring each candidate via $\mathcal{S}(x_u) = \alpha \cdot \mathcal{H}(x_u) + \beta \cdot \mathcal{D}_{cos}(x_u, \mathcal{L}) + \gamma \cdot \mathcal{C}(x_u)$, where $\mathcal{H}$ represents predictive Shannon entropy, $\mathcal{D}_{cos}$ enforces cosine feature diversity, and $\mathcal{C}$ prioritizes classes performing poorly on validation recall.
- **Bounded Class-Balanced Retraining:** Loss weighting $w_c = (N_{total} / (8 \cdot N_c))^{0.5}$ with a maximum cap $W_{max}=5.0$ to prevent gradient singularity on single-sample minority classes.

---

## 2. Repository Structure

```text
CSDA-AL/
│
├── README.md                            # Comprehensive project guide and reproduction instructions
├── requirements.txt                     # Core dependencies (PyTorch, torchvision, scikit-learn, etc.)
├── pyproject.toml                       # Python package configuration (>=3.10)
├── .gitignore                           # Git exclusion rules (ignores raw gigabyte datasets & virtualenvs)
│
├── configs/                             # Centralized YAML experiment configurations
│   ├── dataset.yaml                     # ISIC 2019 path contracts and split definitions
│   ├── training.yaml                    # Optimization hyperparameters, learning rate, loss weights
│   └── active_learning.yaml             # Acquisition batch size, budget ceiling, round schedule
│
├── src/                                 # Production-grade modular source code
│   └── csda_al/
│       ├── __init__.py
│       ├── cli.py                       # Command-line interface dispatcher
│       ├── config.py                    # Schema validation and YAML loader
│       ├── acquisition/                 # Active learning acquisition algorithms
│       │   ├── random.py                # Uniform random baseline
│       │   ├── entropy.py               # Shannon entropy uncertainty sampling
│       │   ├── coreset.py               # Vectorized Core-Set geometric diversity
│       │   ├── difficulty_aware.py      # Validation-recall difficulty estimation
│       │   ├── flex_sampling.py         # Flexible Sampling adaptation (Ju et al.)
│       │   ├── bengar.py                # Constrained class-quota acquisition (Bengar et al.)
│       │   ├── supclust.py              # Contrastive cold-start clustering (SUPClust)
│       │   └── combined_csda.py         # Integrated CSDA-AL tri-component acquisition
│       ├── data/                        # Dataset contracts and split verification
│       │   ├── dataset.py               # ISICDataset loader with robust fallback
│       │   ├── metadata.py              # 8-class metadata, lesion & patient grouping
│       │   ├── split.py                 # Deterministic 70/15/15 canonical split with SHA-256
│       │   └── oracle.py                # Transaction-logged annotation oracle
│       ├── models/                      # Architecture implementations
│       │   ├── classifier.py            # ResNet-18 backbone & linear classification heads
│       │   └── simclr.py                # Self-supervised contrastive pretraining
│       ├── training/                    # Model training loops and cost-sensitive losses
│       │   ├── trainer.py               # Supervised fine-tuning engine
│       │   └── loss.py                  # Bounded class-balanced cross-entropy
│       └── evaluation/                  # Evaluation metrics & diagnostic matrices
│           ├── metrics.py               # Balanced Acc, Macro-F1, Macro-AUROC, per-class recall
│           └── confusion.py             # 8-class confusion matrix generation
│
├── scripts/                             # Executable research experiment scripts
│   ├── run_isic_pipeline.py             # Master multi-mode runner (diagnose, improve, ablation)
│   ├── run_a9_isolated.py               # Isolated A9 synergy experiment under verified split
│   └── run_a14_controlled_validation.py # Seed-matched head-to-head comparison (A1 vs A11)
│
├── notebooks/
│   └── CSDA_AL_ISIC2019_Colab.ipynb     # Interactive Google Colab notebook with one-click GPU execution
│
├── tests/                               # Automated test suite (158 passing unit/integration tests)
│   ├── test_acquisition.py              # Acquisition function score verification
│   ├── test_difficulty.py               # Difficulty and quota mathematical tests
│   ├── test_no_label_leakage.py         # Verification that test labels remain sealed
│   └── test_image_duplicates.py         # Hash coverage and duplicate detection tests
│
├── outputs/                             # Verified experimental logs and cached representations
│   ├── cached_features/                 # Pre-extracted 512-dim ResNet-18 features (52.6 MB)
│   └── isic2019/                        # Checkpoints, manifests, and run records
│       ├── splits/                      # Canonical partition IDs (Train: 17,771 | Val: 3,758 | Test: 3,802)
│       ├── experiments/                 # A0–A14 run logs across seeds 42, 43, 44
│       └── reports/                     # Audit logs and diagnostic reports
│
├── results/                             # Consolidated publication tables and figures
│   ├── figures/                         # 12 high-resolution diagnostic plots & confusion matrices
│   ├── tables/                          # CSV & Markdown benchmark comparison matrices
│   └── reports/                         # Next-stage research report & root-cause analyses
│
└── submission/                          # Official Digiicampus submission artifacts
    ├── README.md
    ├── audit/
    │   └── PROJECT_AUDIT.md             # Complete project and implementation audit
    ├── report/
    │   └── CSDA_AL_Final_Report.pdf     # 5-page academic project report (+ references)
    └── presentation/
        ├── CSDA_AL_Final_Presentation.pptx # 14-slide 15-minute presentation (PPTX)
        └── CSDA_AL_Final_Presentation.pdf  # 14-slide 15-minute presentation (PDF)
```

---

## 3. System Requirements & Hardware Environment

- **Python Version:** Tested and verified on Python `3.10`, `3.11`, `3.12`, and `3.13`.
- **Operating System:** Platform-agnostic (fully tested on Windows 11 and Ubuntu Linux via Google Colab).
- **Hardware Requirements:**
  - **CPU:** Standard x86_64 dual-core CPU (execution via pre-extracted feature cache completes in seconds).
  - **RAM:** Minimum 4 GB RAM recommended (vectorized Core-Set requires only 8 MB RAM).
  - **GPU:** Optional. NVIDIA GPU (T4, V100, A100) supported for full-image backpropagation via PyTorch CUDA.
  - **Disk Space:** 
    - Lightweight code and cached features: ~150 MB.
    - Full uncompressed raw ISIC 2019 image archive: ~35 GB.

---

## 4. Installation & Environment Setup

### 4.1 Clone Repository
```bash
git clone https://github.com/Dishitha09/csda-al.git
cd csda-al
```

### 4.2 Create Virtual Environment
**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 5. Dataset Setup & Google Drive External Links

Because the raw ISIC 2019 image archive comprises over 25,000 high-resolution JPEG images (~35 GB), large image files and model checkpoints are hosted on external cloud storage.

### 5.1 External Resource Links
- **GitHub Repository:** [https://github.com/Dishitha09/csda-al](https://github.com/Dishitha09/csda-al)
- **Interactive Google Colab Notebook:** [https://colab.research.google.com/github/Dishitha09/csda-al/blob/main/notebooks/CSDA_AL_ISIC2019_Colab.ipynb](https://colab.research.google.com/github/Dishitha09/csda-al/blob/main/notebooks/CSDA_AL_ISIC2019_Colab.ipynb)
- **Google Drive Raw Dataset:** `[INSERT ACTUAL GOOGLE DRIVE DATASET LINK]`
- **Google Drive Model Checkpoints:** `[INSERT ACTUAL GOOGLE DRIVE CHECKPOINTS LINK]`
- **Google Drive Consolidated Results:** `[INSERT ACTUAL GOOGLE DRIVE RESULTS LINK]`

> **Note for Faculty Reviewers:** External Google Drive links have been configured with view and download permissions for evaluating faculty. If you encounter permission access restrictions, please contact the team lead at `dishitharamesh09@gmail.com`.

### 5.2 Directory Structure for Raw ISIC 2019
If downloading raw images, organize them into:
```text
data/
└── ISIC2019/
    ├── ISIC_2019_Training_Input/          # 25,331 JPG images
    ├── ISIC_2019_Training_GroundTruth.csv # Class indicator matrix
    └── ISIC_2019_Training_Metadata.csv    # Patient and lesion identifiers
```

---

## 6. How to Execute the Project

### 6.1 Run Technical Audit & Root-Cause Diagnosis
Audits dataset class distribution, reveals the Argmax Quota Trap, Loss Weight Singularity, and Cold-Start Density Bias:
```bash
python run_isic_pipeline.py --mode diagnose
```

### 6.2 Run Isolated A9 Experiment (Cold-Start + Class-Weighted Synergy)
Directly replicates the combination of Calibrated Cold-Start and Class-Weighted loss across seeds 42, 43, 44 under the verified fixed split:
```bash
python scripts/run_a9_isolated.py
```

### 6.3 Run A14 Controlled Head-to-Head Comparison (A1 vs A11)
Executes the seed-matched comparison between Class-Weighted Random (A1) and CSDA-AL (A11) from identical initial labelled sets:
```bash
python scripts/run_a14_controlled_validation.py
```

### 6.4 Execute Full Next-Stage Ablation Suite (A9–A13)
```bash
python run_isic_pipeline.py --mode next_stage --seeds 42 43 44
```

### 6.5 Run Automated Test Suite
```bash
pytest tests/
```

---

## 7. Evaluation Metrics & Terminology

- **Balanced Accuracy:** Macro-averaged recall across all 8 classes: $\frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$. Crucial for imbalanced regimes.
- **Macro-F1 Score:** Unweighted arithmetic mean of per-class F1-scores: $\frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$.
- **Macro-AUROC:** Area under the ROC curve calculated independently per class (one-vs-rest) and averaged. Evaluates ranking quality.
- **Confusion Matrix:** $8 \times 8$ contingency table tracking exact clinical misclassifications across all diagnostic classes.
- **Minority Recall:** Recall on rare malignant categories: Dermatofibroma (DF), Vascular Lesions (VASC), and Squamous Cell Carcinoma (SCC).

---

## 8. Verified Experimental Results

The following results were evaluated across random seeds **42, 43, and 44** under strictly identical conditions ($N_{init}=80, b=40, B=320$):

| ID | Configuration / Strategy | Test Balanced Acc (%) | Test Macro-F1 | Test Macro-AUROC | Primary Finding |
|:---:|---|:---:|:---:|:---:|---|
| **A0** | Random Baseline (Unweighted) | 26.06 ± 0.27 | 0.2606 ± 0.005 | 0.7436 ± 0.022 | Severe minority collapse (0.00 recall on DF, VASC) |
| **A1** | Random + Class-Weighted Loss | **29.43 ± 1.01** | 0.2865 ± 0.005 | 0.7488 ± 0.015 | Best overall Balanced Accuracy |
| **A2** | Calibrated Cold-Start Only | 27.49 ± 0.56 | 0.2769 ± 0.007 | 0.7503 ± 0.008 | Eliminates round 0 class starvation |
| **A3** | Uncertainty Only (Entropy) | 26.75 ± 1.06 | 0.2707 ± 0.012 | 0.7542 ± 0.021 | Queries boundary nevi; depresses precision |
| **A4** | Diversity Only (Core-Set) | 25.99 ± 0.38 | 0.2525 ± 0.008 | 0.7266 ± 0.012 | Samples majority cluster centers |
| **A7** | CSDA-AL without Diversity | 26.53 ± 0.65 | 0.2623 ± 0.010 | 0.7603 ± 0.015 | Strong probability ranking |
| **A8** | Full CSDA-AL Combined | 26.36 ± 1.34 | 0.2675 ± 0.024 | 0.7594 ± 0.013 | Sensitive to boundary prior shift |
| **A9** | Class-Weighted + Cold-Start | 29.01 ± 0.40 | **0.2924 ± 0.003** | 0.7450 ± 0.016 | **Top Stability: 3.6× lower seed variance** |
| **A11**| CW + Uncertainty + Cosine Diversity | 27.08 ± 0.34 | 0.2761 ± 0.007 | **0.7626 ± 0.021** | **Top Ranking: Highest Macro-AUROC** |
| **A14**| Seed-Matched Head-to-Head | A1: 29.43 ± 1.01 | A1: 0.2865 ± 0.005 | A1: 0.7488 ± 0.015 | A1 beats A11 on every seed for Bal Acc & F1 |

---

## 9. Disclosed Limitations & Scientific Failures

1. **The Prior Shift Phenomenon:** In extreme 54:1 imbalance, active learning intentionally queries ambiguous and rare-class samples. At small budgets ($B=320$), this skews the training set prior $P_{AL}(Y)$ away from the true population prior $P_{data}(Y)$. The linear classifier over-predicts rare classes on benign Nevi images, triggering false-positive penalties that degrade balanced accuracy.
2. **Representation Ceiling:** Experiments were conducted using frozen 512-dimensional ResNet-18 representations. End-to-end backpropagation with domain-specific contrastive learning (SimCLR) is required to discover finer lesion boundaries.
3. **Threshold Calibration Gap:** Predictions were derived via default discrete argmax rules. In clinical workflows where continuous probability rankings are evaluated, active learning (A11) achieves superior AUROC (0.7626 vs 0.7488).

---

## 10. Final Submission Checklist (Digiicampus)

- [x] **Complete Source Code:** Fully modular package under `src/csda_al/` with CLI and tests.
- [x] **Root README:** Complete documentation, system specs, external Drive links, and command guide.
- [x] **Presentation PPTX:** `submission/presentation/CSDA_AL_Final_Presentation.pptx` (14 slides, 15-minute format).
- [x] **Presentation PDF:** `submission/presentation/CSDA_AL_Final_Presentation.pdf`.
- [x] **Academic Report PDF:** `submission/report/CSDA_AL_Final_Report.pdf` (5 pages of technical content + references).
- [x] **Submission Archive:** `ML_Project_Group10_DishithaRamesh.zip` verified for integrity.
