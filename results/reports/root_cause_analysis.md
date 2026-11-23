# Root-Cause Analysis: Why Current CSDA-AL Underperformed Random on ISIC 2019

**Executive Summary:**  
In the initial baseline run (`task-1351`), CSDA-AL yielded **24.36% Balanced Accuracy**, **0.2450 Macro-F1**, and **0.7382 AUROC**, while the naive Random baseline achieved **29.39% Balanced Accuracy**, **0.2804 Macro-F1**, and **0.7752 AUROC**.

This diagnosis identifies and ranks the root causes responsible for this degradation.

---

## Severity Ranking of Failure Modes

| Rank | Root Cause | Severity | Impact on CSDA-AL |
| :---: | :--- | :---: | :--- |
| **1** | **Argmax Quota Trap in Imbalanced Selection** | **CRITICAL** | Completely disables minority class acquisition |
| **2** | **Weight Singularity in Cross-Entropy Loss** | **HIGH** | Distorts gradients, damages probability calibration |
| **3** | **Cold-Start Clustering Density Bias** | **HIGH** | Leaves rare classes (AK, DF) with 0 samples in Round 0 |
| **4** | **Unnormalized Feature Embeddings** | **MEDIUM** | Distorts distance-based geometric diversity |
| **5** | **Single Seed Variance at 1.8% Budget** | **MEDIUM** | Results sensitive to stochastic sample initialization |

---

## Detailed Root Cause Profiles

### Root Cause 1 (Severity: CRITICAL): The "Argmax Quota Trap"
* **Mechanism:**
  Active learning methods designed for balanced benchmarks (like CIFAR-10) assume that $\arg\max_k P(Y=k|x)$ covers all classes across the unlabeled pool. On ISIC 2019, the class imbalance is **53.9:1**.
  In rounds 0 through 4, the linear probe is trained on fewer than 200 samples. Because Nevus (NV) makes up >50% and Melanoma (MEL) >17%, the model's posterior probabilities for Dermatofibroma (`DF`, 0.94%) or Vascular Lesion (`VASC`, 1.0%) rarely exceed 0.20–0.30.
  Consequently, `pred_c = np.argmax(probs, axis=1)` assigns **zero** unlabeled instances to class `DF`.
* **Empirical Consequence:**
  The algorithm allocated a quota of e.g. 5 samples to `DF`, searched for instances where `pred_c == DF`, found **0 instances**, and failed to acquire any `DF` samples. The remaining quota defaulted to high-uncertainty instances on the `NV`/`MEL` decision boundary, starving minority classes.
* **Why Random Performed Better:**
  Random sampling draws from the true underlying distribution without model bias. Over 320 draws, Random inevitably sampled several instances of minor classes purely by chance, whereas biased CSDA-AL hyper-focused on boundary majority noise.

---

### Root Cause 2 (Severity: HIGH): Loss Weight Singularity ($w \approx 8 \times 10^6$)
* **Mechanism:**
  The weighting formula $w_c = \frac{N}{N_c + 10^{-5}}$ was normalized by dividing by the sum of weights.
  When $N_c = 0$ for any class, $w_{\text{absent}} \approx \frac{80}{10^{-5}} = 8,000,000$.
  The sum of weights $\sum_c w_c \approx 8,000,000$, which scaled the weights of all actually present classes down to:
  $$w_{\text{present}} \approx \frac{O(1)}{8,000,000} \approx 10^{-7}$$
* **Empirical Consequence:**
  The learning signals for all classes present in the batch were crushed to near zero, destroying classifier convergence in early rounds.

---

### Root Cause 3 (Severity: HIGH): SUPClust Density Bias at Cold Start
* **Mechanism:**
  MiniBatchKMeans on raw feature space naturally allocates cluster centroids to regions of high density. In a 54:1 imbalanced distribution, high-density regions are overwhelmingly `NV` and `MEL`.
* **Empirical Evidence from Cold Start Inventory:**
  ```json
  "class_counts": [16, 42, 9, 0, 10, 0, 2, 1]
  ```
  Class index 3 (`AK`) and class index 5 (`DF`) received **zero instances** in the seed set. Starting Active Learning with missing classes created an unrecoverable deficit given Root Cause 1.

---

## Prescription: The 3 Scientific Fixes

1. **Soft-Target Deficit Matching:**
   Replace `argmax` filtering with soft-posterior expectation:
   $$C(x) = \sum_{c=0}^7 P(Y=c|x) \cdot \text{NormalizedDeficit}(c)$$
   This identifies instances with meaningful probability of being minority lesions, even if their argmax is not yet the top class.

2. **Capped Class-Balanced Loss:**
   Implement Cui et al. effective number of samples with a hard cap:
   $$w_c = \min\left( \frac{1 - \beta}{1 - \beta^{N_c + 1}}, 20.0 \right) \quad (\text{if } N_c > 0 \text{ else } 0.0)$$

3. **L2-Normalized Embedding Space:**
   Compute all dot products, centroids, and diversity distances on unit sphere $\|\mathbf{z}\|_2 = 1.0$.
