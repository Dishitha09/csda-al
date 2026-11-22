# CSDA-AL Project Scope Resolution

**Document ID:** CSDA-AL-PSR-001  
**Date:** 2026-11-22  
**Status:** Approved & Binding  
**Authority:** Master Execution Specification & Latest Technical Directives (supersedes preliminary meeting transcripts)

---

## 1. Executive Summary

This document resolves the architectural discrepancies between early project discussion transcripts (which temporarily excluded transfer-learning initialization and stopping criteria) and the authoritative project specification (the slide specification, Master Codex prompt, and Starter Bundle directives). 

The authoritative project mandate dictates:
1. **Transfer-Learning Cold-Start Initialization** and **Class-Aware Stopping Criterion** are formally incorporated into the primary system architecture.
2. To preserve full scientific transparency and satisfy comparative baselines, **no-transfer** (self-supervised SimCLR cold-start) and **no-class-aware-stop** (fixed round/budget stopping) are retained as mandatory, explicit experimental ablation configurations.
3. The four primary research papers (Ju et al. *Flexible Sampling*, Ono et al. *SUPClust*, Bengar et al. *Class-Balanced AL*, and Khanal et al. *Active Label Refinement*) are methodologically mapped and attributed to their scientifically appropriate roles.

---

## 2. Chronological Scope Evolution

| Milestone / Source | Scope Definition | Key Inclusions | Key Exclusions / Caveats |
|---|---|---|---|
| **Original Problem Statement Slide** (`original_problem_statement_slide.png`) | Cold-Start Initialization, Class-Imbalance Active Learning, Difficulty-Aware Sampling, Transfer Learning, Stopping Rule | 1. Transfer-learning cold start<br>2. Imbalance-aware AL<br>3. Difficulty-aware sampling<br>4. Class-aware stopping rule | Early conceptual draft |
| **Interim Meeting Transcript** (`prompt.pdf` / audio discussion) | Scoped down due to initial GPU resource and timeline constraints | Core active learning on ISIC 2019, SimCLR self-supervised pretraining | Transfer learning and stopping criteria marked as deferred/out-of-scope |
| **Starter Bundle & Master Codex Directives** (`ANTIGRAVITY_MASTER_PROMPT_ISIC2019.md`, `INPUT_AUDIT_NOTES.md`) | Re-integrated full scope with rigorous ablation controls | Full 8-class ISIC 2019 pipeline: SUPClust, Bengar Class-Balanced AL, Flexible Sampling, Transfer Learning baseline, Class-Aware Stopping | Preserves clean oracle separation; Khanal re-labeling isolated to noisy-label extension |

---

## 3. Discrepancy Resolution Matrix

### 3.1 Cold-Start Initialization: SimCLR vs. Transfer Learning
* **Discrepancy:** The early discussion proposed self-supervised contrastive learning (SimCLR) without pre-trained ImageNet weights to study pure self-supervised initialization; the master problem statement requested transfer learning.
* **Resolution:** Both modalities are implemented:
  - **Core Baseline (SSL Cold-Start):** In-domain self-supervised SimCLR representation extraction on unlabelled ISIC 2019 followed by SUPClust / TypiClust clustering.
  - **Transfer-Learning Extension:** ImageNet-pretrained ResNet-18 feature extraction followed by SUPClust / TypiClust clustering and linear warm-up.
  - **Ablation Control:** Random initial acquisition (B0) establishes the lower bound.

### 3.2 Stopping Rule: Fixed Budget vs. Class-Aware Stopping
* **Discrepancy:** Early scripts terminated solely upon exhausting a fixed query round count or annotation budget; the research prompt demanded an active class-aware stopping criterion.
* **Resolution:** 
  - A mathematically rigorous **Class-Aware Stopping Criterion** is implemented in `csda_al.engine.stopping`. It evaluates:
    1. Minimum minority-class representation (all classes must have at least $k_{min}$ acquired samples).
    2. Per-class validation recall thresholds and recall stability ($\Delta \text{Recall}_c < \epsilon$).
    3. Macro-F1 plateau over consecutive rounds ($P$ patience rounds without improvement $> \delta$).
  - **Leakage Prevention:** Stopping evaluations strictly consume validation set predictions or acquired training counts; **never** the held-out test set.
  - **Ablation Control:** Fixed-budget / fixed-round termination remains available via configuration flag `stopping.strategy: "fixed_budget"` vs `"class_aware"`.

### 3.3 Active Label Refinement (Khanal et al.)
* **Discrepancy:** Khanal et al. (MICCAI 2024) specifically tackles *noisy label correction* (Co-teaching + Variance of Gradients) rather than unlabelled pool acquisition.
* **Resolution:** Khanal's pipeline is implemented as an independent noisy-label experimental module (`experiments/noisy_label_refinement/`). It is evaluated under controlled synthetic label noise (20%, 40%) rather than conflated with clean-label imbalanced active learning.

### 3.4 Flexible Sampling (Ju et al.)
* **Discrepancy:** The original Ju et al. code (`flexsampling`) utilizes full-dataset class labels to select class anchors, which violates pool-based active learning if applied naively to unlabelled pools.
* **Resolution:** 
  - **Faithful Baseline:** Replicates the original paper's supervised anchor pretraining on known seeds.
  - **Pool-Safe CSDA-AL Adaptation:** Class prototypes and difficulty estimation are dynamically updated only using the *acquired* labelled set and validation performance, preserving oracle sanctity.

---

## 4. Final Feature Classification

### 4.1 Mandatory Core Features
1. **Stratified ISIC 2019 Pipeline:** 70% train / 10% validation / 20% test split with cryptographically locked identity hashes (zero data leakage).
2. **Cold-Start Strategies:** Random, KMeans, TypiClust, and SUPClust (with SUP and typicality ablation modes).
3. **Imbalance-Aware Acquisition:**
   - Bengar et al. (WACV 2022) Class-Balanced Active Learning (entropy uncertainty penalized by pseudo-class distribution).
   - Entropy, BALD (MC-Dropout), and Core-Set baselines.
   - CSDA-AL Difficulty-Aware Acquisition (combining uncertainty with validation-derived class difficulty weights).
4. **Class-Aware Stopping:** Multi-metric validation-only stopping rule.
5. **Evaluation Suite:** Full classification metrics (Accuracy, Macro-F1, Balanced Accuracy, Per-Class Precision/Recall/F1, Confusion Matrix, and Macro-AUROC) and Active Learning efficiency (AULC, budget efficiency).

### 4.2 Research Extensions & Ablations
1. **Transfer-Learning Cold Start:** Pretrained ImageNet weights vs. in-domain SimCLR SSL weights.
2. **Khanal Label Refinement:** Co-teaching + VOG under 20% and 40% label noise.
3. **Stopping Ablation:** Class-aware early stopping vs. fixed budget budget-exhaustion.

---

## 5. Proposed Research Contribution

The resulting unified framework—**CSDA-AL (Cold-Start Difficulty-Aware Active Learning)**—delivers:
1. **Principled Cold Start:** Eliminates initial sample bias on long-tailed medical datasets via self-supervised boundary typicality (SUPClust).
2. **Dynamic Class-Difficulty Adaptation:** Dynamically shifts annotation queries toward hard minority diagnostic classes (e.g., DF, VASC, AK) without leaking unlabelled pool ground truth.
3. **Class-Balanced Regularization:** Adopts Bengar-style pseudo-label distribution balancing to prevent dominant classes (NV, MEL) from exhausting the annotation budget.
4. **Resource-Conscious Early Stopping:** Halts acquisition when minority classes attain diagnostic stability, maximizing clinical return-on-investment.
