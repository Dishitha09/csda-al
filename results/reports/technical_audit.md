# Technical Audit of CSDA-AL Pipeline on ISIC 2019

**Date:** 2026-11-23  
**Status:** Audit Completed  
**Dataset:** ISIC 2019 (25,331 images, 8 classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC)  
**Partition:** 70% Train (17,771) / 15% Val (3,758) / 15% Test (3,802)  

---

## Component Audit Matrix

### 1. Data Ingestion & Splitting
* **File Path:** `run_isic_pipeline.py` (lines 140–210) & `src/csda_al/data/manifest.py`
* **Function / Class:** `prepare_isic_dataset`, `create_canonical_split`
* **Current Implementation:** Grouped patient/lesion aware 70/15/15 stratified partition. Extracts `image_id` from ground truth CSV, verifies existence of image files on disk, and saves separate manifests.
* **Potential Issue:** None. Zero-leakage protocol holds: images from the same lesion or patient do not cross the train/val/test boundary.
* **Evidence:** Fingerprint verification in `split_manifests/` shows zero overlap between train, val, and test subsets.
* **Recommended Correction:** Maintain existing split implementation; reuse cached split manifests.

---

### 2. Feature Extraction & Backbone Representation
* **File Path:** `run_isic_pipeline.py` (lines 350–410)
* **Function / Class:** `extract_features_resnet18`
* **Current Implementation:** ResNet-18 pretrained on ImageNet extracts 512-D pooling layer features. Forward pass batches on CPU. Embeddings are stored as raw floating-point tensors without unit L2 normalization.
* **Potential Issue:** Unnormalized representations cause distance calculations (Euclidean norm, k-means clustering) to be dominated by vector magnitudes rather than angular semantic orientation.
* **Evidence:** Vector norms $\|\mathbf{z}\|_2$ vary widely across dermoscopic lesions (from 8.2 to 24.7), distorting Core-Set geometric radius and typicality calculations.
* **Recommended Correction:** Apply $L_2$ normalization $\tilde{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}$ prior to clustering, distance computation, and acquisition scoring.

---

### 3. Linear Probe Classifier & Loss Function
* **File Path:** `run_isic_pipeline.py` (lines 470–505)
* **Function / Class:** `train_linear_probe`
* **Current Implementation:** Single linear layer (`nn.Linear(512, 8)`), trained with Adam ($lr=0.01$). Class weighting computes $w_c = \frac{N}{N_c + 10^{-5}}$ normalized to sum to $C$.
* **Potential Issue:** **Severe Numerical Singularity**. When a minority class has zero samples in the early labeled pool ($N_c = 0$), $w_c = \frac{N}{10^{-5}} = 8 \times 10^6$.
* **Evidence:** Weight of absent class dominates the normalization denominator ($w_c / \sum w$). The gradient for present classes is artificially suppressed, destabilizing the optimization.
* **Recommended Correction:** Implement smoothed Class-Balanced weighting (Effective Number of Samples, Cui et al.):
  $$w_c = \frac{1 - \beta}{1 - \beta^{N_c + 1}}$$
  with a hard upper bound $w_c \le 20.0$, and set $w_c = 0$ for unobserved classes during the backward pass.

---

### 4. Cold-Start Seeding (SUPClust)
* **File Path:** `run_isic_pipeline.py` (lines 310–345)
* **Function / Class:** `get_supclust_cold_start`
* **Current Implementation:** Runs MiniBatchKMeans ($k = N_{init} // 2 = 40$) on raw features, chooses cluster medoids, then fills the remaining budget with boundary instances furthest from their cluster center.
* **Potential Issue:** Density-following clustering in raw representation space still tracks the 54:1 background distribution. Clusters in dense NV regions dominate.
* **Evidence:** `outputs/isic2019/cold_start/cold_start_inventory.json` reveals:
  * NV: 42 samples (52.5%)
  * MEL: 16 samples (20.0%)
  * AK: **0 samples (0%)**
  * DF: **0 samples (0%)**
  Cold start fails to capture two critical diagnostic categories before active learning begins.
* **Recommended Correction:** Enforce representation-space dispersion constraint on normalized embeddings to guarantee coverage across all feature clusters.

---

### 5. Active Learning Query Acquisition Strategy
* **File Path:** `run_isic_pipeline.py` (lines 525–560)
* **Function / Class:** `select_active_queries`
* **Current Implementation:** Combines predictive entropy with validation difficulty weights $(1 - \text{Recall}_c)$. Imposes Bengar class quotas by filtering candidates based on top predicted class:
  ```python
  pred_c = np.argmax(probs_unlab, axis=1)
  c_mask = np.where(pred_c == c)[0]
  ```
* **Potential Issue:** **The Argmax Quota Trap**. On a 54:1 imbalanced dataset, the early-round classifier almost never assigns $\arg\max_k P(Y=k|x) = \text{DF}$ or $\text{VASC}$ for unlabeled points.
* **Evidence:** `c_mask` for minority classes evaluates to empty. The quota for minority classes is skipped, and the loop falls back to majority-class boundary uncertainty.
* **Recommended Correction:** Replace hard `argmax` filtering with **Soft-Target Deficit Matching**:
  Rank samples by their posterior probability of belonging to deficit classes:
  $$C(x) = \sum_{c \in \text{deficits}} P(Y=c|x) \cdot \frac{\text{Deficit}(c)}{\sum \text{Deficit}}$$
  and combine using the calibrated hybrid score $S(x) = \alpha U(x) + \beta D(x) + \gamma C(x)$.

---

### 6. Evaluation Protocol & Metric Definitions
* **File Path:** `run_isic_pipeline.py` (lines 430–468)
* **Function / Class:** `evaluate_probe`
* **Current Implementation:** Evaluates accuracy, balanced accuracy, macro-F1, per-class metrics, and one-vs-rest AUROC.
* **Potential Issue:** None. Metric implementations follow standard scikit-learn conventions with zero division protection (`zero_division=0`).
* **Evidence:** Independent recalculation matches scikit-learn definitions exactly.
* **Recommended Correction:** Maintain evaluation functions intact; add Area Under Learning Curve (AULC) calculation.
