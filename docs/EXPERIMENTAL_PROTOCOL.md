# CSDA-AL Experimental Protocol

**Document ID:** CSDA-AL-EXP-001  
**Project:** Cold-Start Difficulty-Aware Active Learning on ISIC 2019  
**Status:** Canonical & Audited  

---

## 1. Dataset Specification: ISIC 2019

The primary dataset is **ISIC 2019: Skin Lesion Analysis Towards Melanoma Detection**, sourced from the International Skin Imaging Collaboration archive (BCN_20000, HAM10000, and MSK datasets).

### 1.1 Class Distribution & Diagnostic Categories

| Class Index | Abbreviation | Diagnostic Category | Total Count | % of Dataset | Imbalance Tier |
|---|---|---|---|---|---|
| 0 | **MEL** | Melanoma | 4,522 | 17.85% | Head / Major Malignant |
| 1 | **NV** | Melanocytic Nevus | 12,875 | 50.83% | Dominant Head |
| 2 | **BCC** | Basal Cell Carcinoma | 3,323 | 13.12% | Middle |
| 3 | **AK** | Actinic Keratosis | 867 | 3.42% | Tail / Minority |
| 4 | **BKL** | Benign Keratosis | 2,624 | 10.36% | Middle |
| 5 | **DF** | Dermatofibroma | 239 | 0.94% | Extreme Rare Tail |
| 6 | **VASC** | Vascular Lesion | 253 | 1.00% | Extreme Rare Tail |
| 7 | **SCC** | Squamous Cell Carcinoma | 628 | 2.48% | Tail / Rare Malignant |
| **Total** | | **8 Classes** | **25,331** | **100.0%** | **Imbalance Ratio ~ 54:1** |

---

## 2. Partitioning, Patient Stratification & Leakage Controls

### 2.1 Cryptographically Frozen Split
- **Training Pool ($\mathcal{U}_{\text{train}}$):** 17,732 images (70.0%)
- **Validation Set ($\mathcal{V}$):** 3,800 images (15.0%)
- **Held-Out Test Set ($\mathcal{T}_{\text{test}}$):** 3,799 images (15.0%)
- **Split Fingerprint:** `1ef5b18099a5e35bc8d101277f5228d41e82c931fa60c9854763bcfdf386a55d`
- **Duplicate Prevention:** Perceptual dHash and byte SHA-256 deduplication applied across splits.
- **Oracle Isolation:** Training labels are sealed behind `LabelOracle`. The unlabelled pool exposes only `image_id` strings and pixel tensors.

---

## 3. Active Learning Protocol

### 3.1 Budgeting Parameters
- **Initial Labelled Set Budget ($B_0$):** 80 samples (Round 1)
- **Batch Acquisition Budget ($B$):** 40 samples per round
- **Maximum Annotation Budget ($B_{\max}$):** 320 samples (Rounds 1 through 7)
- **Validation Overhead ($B_{\text{val}}$):** 3,800 samples (accounted separately as fixed clinical validation cost)

### 3.2 Model & Training Pipeline
- **Backbone Architecture:** ResNet-18 (Torchvision canonical, 512-D penultimate features).
- **Self-Supervised Pretraining:** SimCLR with 2-layer MLP projection head (512 $\to$ 512 $\to$ 128), NT-Xent loss ($\tau = 0.5$).
- **Classifier Head:** Linear layer ($512 \to 8$) with Dropout ($p = 0.1$).
- **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$), Cosine Annealing learning rate schedule.
- **Image Preprocessing:** Bilinear resize to $224 \times 224$, standard ImageNet channel normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 4. Benchmark Experimental Matrix (14 Methods)

| ID | Method Name | Initial Strategy | Active Query Strategy | Core Literature Reference |
|---|---|---|---|---|
| **B0** | Random Baseline | Random | Uniform Random | Standard Control Baseline |
| **B1** | Supervised Upper Bound | Full Training Pool | N/A | Full Pool Supervision (17,732 labels) |
| **M1** | SUPClust Cold-Start | SUPClust | Uniform Random | Ono et al. (ICLR 2024 Workshop) |
| **M2** | Flexible Sampling | Random | Flexible Curriculum (Ju et al.) | Ju et al. (MICCAI 2022) |
| **M3** | Bengar Class-Balanced | Random | Constrained Quota Allocation | Bengar et al. (WACV 2022) |
| **M4** | Shannon Entropy | Random | Maximum Predictive Entropy | Shannon (1948) / Settles (2009) |
| **M5** | Core-Set | Random | Greedy k-Center Distance | Sener & Savarese (ICLR 2018) |
| **M6** | BALD | Random | MC-Dropout Mutual Information | Gal et al. (ICML 2017) |
| **M7** | Difficulty-Aware | Random | CSDA-AL Validation Difficulty | Ju et al. / Proposed |
| **M8** | Khanal Noisy Refinement | Random (with noise) | Co-teaching + VOG Refinement | Khanal et al. (MICCAI 2024) |
| **C1** | Combined: SUP + Flex | SUPClust | Flexible Curriculum | Ono et al. + Ju et al. |
| **C2** | Combined: SUP + Bengar | SUPClust | Bengar Class-Balanced | Ono et al. + Bengar et al. |
| **C3** | Combined: Flex + Bengar | Random | Flex Difficulty + Bengar Quotas | Ju et al. + Bengar et al. |
| **C4** | **Proposed CSDA-AL Full** | **SUPClust** | **CSDA-AL (Difficulty + Bengar + Stop)** | **Proposed Unified Framework** |

---

## 5. Evaluation Metrics & Statistical Analysis

### 5.1 Primary Metrics
1. **Validation Macro-F1:**
   $$\text{Macro-F1} = \frac{1}{C}\sum_{c=0}^{C-1} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c + \epsilon}$$
2. **Balanced Accuracy:**
   $$\text{Balanced Accuracy} = \frac{1}{|C_{\text{supported}}|}\sum_{c \in C_{\text{supported}}} \text{Recall}_c$$
3. **Minority-Class (Tail) Recall:** Mean recall across the three rarest diagnostic categories: DF, VASC, and SCC.
4. **Macro-AUROC:** Macro-averaged one-vs-rest Area Under the Receiver Operating Characteristic curve across all supported classes.
5. **Normalized Area Under the Learning Curve (AULC):**
   $$\text{AULC} = \frac{1}{B_{\max} - B_0}\int_{B_0}^{B_{\max}} \text{MacroF1}(b) \, db$$
6. **Annotation Label Savings:**
   $$\text{Savings} = 1 - \frac{B_{\text{method}}(S)}{B_{\text{random}}(S)}$$
   where $B_{\text{method}}(S)$ is the budget required to achieve benchmark performance threshold $S$.
