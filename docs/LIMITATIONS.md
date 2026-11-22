# CSDA-AL Research Limitations & Methodological Boundaries

**Document ID:** CSDA-AL-LIM-001  
**Project:** Cold-Start Difficulty-Aware Active Learning on ISIC 2019  
**Status:** Comprehensive Analysis of Computational and Scientific Boundaries  

---

## 1. Hardware & Computational Constraints
1. **CPU Execution Throughput:** In the benchmark environment, execution is CPU-bound (`torch.cuda.is_available() == False`). While active learning queries, clustering, and inference are fully operational and verified, training deep ResNet-18 networks over tens of thousands of images across 14 methods and 3 seeds requires substantial wall-clock time (~25–30 minutes per round for 3,800 validation images). Accelerated GPU execution (e.g., NVIDIA A100/V100) is recommended for massive-scale hyperparameter sweeps.
2. **Feature Extraction Amortization:** To prevent redundant backbone forward passes during acquisition, penultimate representations (512-D) are cached on disk via `FeatureStore`. Pre-computed feature extraction significantly accelerates subsequent acquisition rounds.

---

## 2. Algorithmic Boundaries & Methodological Deviations

### 2.1 Flexible Sampling (Ju et al., MICCAI 2022)
- **Paper Assumption:** The original code in `flexsampling/core/anchor.py` and `curriculum.py` accesses ground-truth class labels across the entire training dataset to compute class prototypes and sample per-class buckets.
- **Active Learning Constraint:** In a genuine active learning scenario, labels for the unlabelled pool do not exist prior to annotation.
- **CSDA-AL Adaptation:** Class membership is inferred strictly via classifier pseudo-labels ($\hat{y} = \arg\max p(y|x)$) or unsupervised cluster proximity. This preserves oracle integrity and avoids ground-truth data leakage.

### 2.2 Active Label Refinement (Khanal et al., MICCAI 2024)
- **Task Distinction:** Khanal et al. specifically addresses *label noise correction* in datasets with corrupted annotations, rather than pool-based sample acquisition from clean unlabelled pools.
- **Methodological Scope:** Khanal's pipeline is evaluated independently under synthetic label noise (20%, 40%) as an experimental extension, rather than conflated with clean-label imbalanced active learning.

### 2.3 Cold-Start Feature Geometry (SUPClust)
- **Semantic Alignment:** Clusters formed in unsupervised SimCLR representation space correspond to visual morphology and feature geometry, not necessarily diagnostic pathology. While typicality and boundary scoring (SUP) effectively diversify the initial sample batch, rare pathological subtypes that share visual features with common nevi remain challenging to separate without supervision.

---

## 3. Clinical & Evaluation Boundaries
1. **Validation Set Overhead:** While CSDA-AL cold-start operates without initial training labels, evaluating validation recall requires a labelled validation set (3,800 images). In clinical deployments, validation annotation overhead must be accounted for in total project cost.
2. **Patient Independence:** The ISIC 2019 archive contains lesions from multiple clinical centers. While patient and lesion grouping is enforced where metadata is present, missing patient IDs in public subsets mean complete absence of patient-level correlation cannot be mathematically proven for 100% of samples.
3. **Diagnostic Non-Equivalence:** In dermatology, false negatives on malignant lesions (Melanoma, Squamous Cell Carcinoma) carry far higher clinical severity than false positives on benign nevi. CSDA-AL treats all errors equally in standard Cross-Entropy loss; integrating cost-sensitive asymmetric clinical penalties is a promising avenue for future work.
4. **Clinical Tool Disclaimer:** This software and research framework is intended strictly for machine learning research and algorithm benchmarking. It is not an approved medical device and must not be used for diagnostic decision-making in clinical healthcare.
