# CSDA-AL Implementation and Verification Report

**Document ID:** CSDA-AL-IR-001  
**Date:** 2026-11-22  
**Status:** Software Architecture Complete & Verified  

---

## 1. Executive Summary

This report documents the implementation, integration, and software verification of the **CSDA-AL (Cold-Start Difficulty-Aware Active Learning)** research framework. All components mandated by the Master Execution Prompt have been implemented, tested, and integrated into the production package `src/csda_al/`.

The test suite contains **209 automated unit and integration tests**, achieving a **100% pass rate** in 25.49 seconds. End-to-end active learning execution has been confirmed on the actual ISIC 2019 prepared dataset across all 7 rounds without synthetic substitutions.

---

## 2. Implemented Research Components

### 2.1 Bengar Class-Balanced Active Learning (`src/csda_al/acquisition/bengar.py`)
- **Implemented:** Implements Bengar et al. (WACV 2022) with both constrained quota allocation (proportional to class representation deficits) and continuous penalized uncertainty formulations.
- **Contract:** Inherits `AcquisitionStrategy`, receives `labeled_counts` via `QueryContext`, uses pseudo-labels to partition candidates, and handles class underflow through global uncertainty redistribution.
- **Verification:** Unit tested in `tests/test_new_components.py` (both modes verified, quota deficits tested).

### 2.2 Flexible Sampling Acquisition (`src/csda_al/acquisition/flex_sampling.py`)
- **Implemented:** Implements Ju et al. (MICCAI 2022) curriculum sampling. Converts validation class recalls into difficulty-proportional sampling quotas, sampling candidates via entropy or BALD without ground-truth label leakage.
- **Verification:** Tested in `tests/test_new_components.py` with mock validation difficulty.

### 2.3 Class-Aware Early Stopping Criterion (`src/csda_al/engine/stopping.py`)
- **Implemented:** Multi-metric early stopping rule evaluating minimum minority-class representation ($k_{\min} \ge 2$), Macro-F1 plateau over consecutive patience rounds, and tail-class recall satiation.
- **Integration:** Wired into `src/csda_al/engine/active_loop.py` during validation pending phases and exported as round diagnostics.
- **Verification:** Unit tested in `tests/test_new_components.py` across all state transitions.

### 2.4 Macro-AUROC Multi-Class Metric (`src/csda_al/evaluation/metrics.py`)
- **Implemented:** One-vs-rest Macro-AUROC calculation via `compute_macro_auroc` using Scikit-Learn. Handles classes with zero support gracefully.
- **Integration:** Wired into `evaluate_predictions` in `src/csda_al/evaluation/report.py`.
- **Verification:** Tested on synthetic and multi-class test fixtures; verified in real ISIC 2019 outputs.

### 2.5 Khanal Noisy-Label Refinement Pipeline (`src/csda_al/imbalance/noisy_label_refinement.py` & `experiments/scripts/run_khanal_refinement.py`)
- **Implemented:** Co-teaching dual-model small-loss selection, synthetic label corruption (0%, 20%, 40%), and active label refinement via disagreement and predictive entropy.
- **Execution Evidence:** Executed on real ISIC 2019 samples, achieving noise identification and re-labeling precision up to 0.450 on 40% noise. Metrics saved to `results/metrics/m8_khanal_noisy_refinement.json`.

### 2.6 Windows Filesystem Robustness (`src/csda_al/utils/io.py`)
- **Fix:** Enhanced `atomic_json` with a 5-attempt retry-backoff loop to resolve transient Windows OS file locking (`PermissionError: [WinError 5] Access is denied`) during fast checkpointing and replay cycles.

---

## 3. Test Suite Verification Summary

```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-8.3.5, pluggy-1.7.0
rootdir: C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al
configfile: pyproject.toml
collected 209 items

tests/test_acquisition.py .........................                      [ 11%]
tests/test_archive_preparation.py ...........                             [ 17%]
tests/test_cli_smoke.py ............                                     [ 22%]
tests/test_dataset.py .................                                  [ 31%]
tests/test_dataset_analysis.py .......                                   [ 34%]
tests/test_difficulty.py .........                                       [ 38%]
tests/test_engine.py ..........                                          [ 43%]
tests/test_foundation_review.py ...........                              [ 48%]
tests/test_image_duplicates.py ........                                  [ 52%]
tests/test_metrics.py .....................                              [ 62%]
tests/test_models_features.py ..........                                 [ 67%]
tests/test_new_components.py .......                                     [ 70%]
tests/test_no_label_leakage.py ...........                               [ 76%]
tests/test_packaging.py ............                                     [ 81%]
tests/test_simclr.py ............                                        [ 87%]
tests/test_stage1_data.py ..........                                     [ 92%]
tests/test_supclust.py ................                                  [100%]

============================== 209 passed in 25.49s ===========================
```

---

## 4. Real ISIC 2019 Active Learning Execution Evidence

A full 7-round active learning run (`results/runs/test_b0`) was executed directly against real ISIC 2019 training images (17,732 pool) and evaluated on 3,800 validation images:
- **Rounds Completed:** 7 rounds (Round 1: 80 samples, Rounds 2–7: 40 samples/round = 320 total annotations).
- **Oracle Transactions:** All 320 labels acquired through `LabelOracle` and recorded in `oracle_ledger.json`.
- **Model Checkpoints:** Saved `initial.pt` and `round_0001.pt` through `round_0007.pt` with SHA-256 fingerprints.
- **Validation Artifacts:** Produced `round_0001.json` through `round_0007.json`, confusion matrices (`round_XXXX_confusion.csv`), and per-class reports.
- **Metric Verification:** Real output confirmed `accuracy: 0.457`, `balanced_accuracy: 0.165`, `macro_f1: 0.143`, and `macro_auroc: 0.519` (reflecting majority-class collapse in random sampling).
- **Integrity Properties:** `synthetic: false`, `no_real_training: false`, zero test leakage.
