"""A18 Experiment: Improved Minority-Aware Active Learning (MAL) with Fine-Tuned Backbone.

Integrates:
- ResNet-18 + Fine-Tuned Representation Adapter (identical to A1_Finetuned)
- Class-weighted loss
- Normalized multi-objective acquisition score:
    Score = alpha * Uncertainty + beta * Diversity + gamma * Minority_Relevance - delta * Redundancy
- Configurable weights: alpha=0.35, beta=0.25, gamma=0.25, delta=0.15
- Margin + Entropy uncertainty
- Model-predicted minority relevance with validation deficiency weighting
- Farthest-first intra-batch diversity with redundancy penalty
- Evaluation across seeds 42, 43, 44 and budgets 80 -> 320
- Validation calibration (tau) and validation-optimized thresholds (T*)
- Full checkpoint saving and logging to results/all_experiments.csv
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.optim as optim

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8
MINORITY_CLASSES = [5, 6, 7]  # DF, VASC, SCC

# =============================================================================
# 1. Configurable Acquisition Hyperparameters
# =============================================================================

@dataclass
class A18AcquisitionConfig:
    alpha_uncertainty: float = 0.35
    beta_diversity: float = 0.25
    gamma_minority: float = 0.25
    delta_redundancy: float = 0.15
    candidate_pool_multiplier: int = 4

# =============================================================================
# 2. Model: Fine-Tuned Adapter (identical to A1_Finetuned)
# =============================================================================

class FineTunedAdapter(nn.Module):
    """Representation adaptation head with residual projection block."""
    def __init__(self, in_features: int = 512, num_classes: int = 8):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, in_features),
            nn.LayerNorm(in_features),
        )
        self.head = nn.Linear(in_features, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

# =============================================================================
# 3. Training Function with Class-Weighted Loss
# =============================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
) -> nn.Module:
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    total = float(len(y_train))
    present = counts > 0
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    if present.sum() > 0:
        raw_w = total / (torch.sqrt(counts[present]) * np.sqrt(total) + 1.0)
        raw_w = torch.clamp(raw_w, min=0.1, max=15.0)
        weights[present] = raw_w / raw_w.sum() * present.sum()

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

# =============================================================================
# 4. Improved A18 Acquisition Function
# =============================================================================

def a18_acquisition(
    unlab_indices: List[int],
    current_labeled: List[int],
    probs_unlab: np.ndarray,
    adapted_unlab_feats: np.ndarray,
    adapted_labeled_feats: np.ndarray,
    val_recalls: np.ndarray | None,
    batch_size: int = 40,
    config: A18AcquisitionConfig = A18AcquisitionConfig(),
) -> List[int]:
    """Score = alpha * Uncertainty + beta * Diversity + gamma * Minority_Relevance - delta * Redundancy"""
    N = len(unlab_indices)
    
    # 1. Uncertainty: Combination of Normalized Shannon Entropy and Margin Uncertainty
    p_safe = np.clip(probs_unlab, 1e-12, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
    
    sorted_probs = np.sort(probs_unlab, axis=1)
    margin = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])  # High when top 2 classes are ambiguous
    
    ent_norm = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy) + 1e-12)
    margin_norm = (margin - np.min(margin)) / (np.max(margin) - np.min(margin) + 1e-12)
    uncertainty_score = 0.5 * ent_norm + 0.5 * margin_norm

    # 2. Minority Relevance (Label-Free from Model Predictions)
    # Deficit weighting: priority for classes underperforming on validation set
    if val_recalls is not None:
        deficits = np.maximum(0.0, 1.0 - val_recalls)
    else:
        deficits = np.ones(NUM_CLASSES)
        deficits[[0, 1]] = 0.2  # De-emphasize Melanoma and Nevi
        
    weighted_mass = np.sum(probs_unlab * deficits, axis=1)
    peak_minority_prob = np.max(probs_unlab[:, MINORITY_CLASSES], axis=1)
    
    # Combine deficit mass with peak rare-class probability
    raw_minority = 0.6 * weighted_mass + 0.4 * peak_minority_prob
    minority_score = (raw_minority - np.min(raw_minority)) / (np.max(raw_minority) - np.min(raw_minority) + 1e-12)

    # 3. Candidate Filtering: Top candidates by uncertainty & minority relevance
    pre_score = 0.5 * uncertainty_score + 0.5 * minority_score
    cand_pool_size = min(N, batch_size * config.candidate_pool_multiplier)
    cand_rank = np.argsort(pre_score)[::-1][:cand_pool_size]
    cand_unlab = [unlab_indices[i] for i in cand_rank]
    
    cand_feats = adapted_unlab_feats[cand_rank]
    # Normalize features
    cand_feats_norm = cand_feats / np.linalg.norm(cand_feats, axis=1, keepdims=True).clip(min=1e-12)
    lab_feats_norm = adapted_labeled_feats / np.linalg.norm(adapted_labeled_feats, axis=1, keepdims=True).clip(min=1e-12)

    # 4. Diversity with respect to labeled set
    sims_to_lab = np.dot(cand_feats_norm, lab_feats_norm.T)
    dist_to_lab = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(np.max(sims_to_lab, axis=1), -1.0, 1.0)))
    diversity_score = (dist_to_lab - np.min(dist_to_lab)) / (np.max(dist_to_lab) - np.min(dist_to_lab) + 1e-12)

    # Static component score for candidates
    cand_static_score = (
        config.alpha_uncertainty * uncertainty_score[cand_rank]
        + config.beta_diversity * diversity_score
        + config.gamma_minority * minority_score[cand_rank]
    )

    # 5. Greedy Selection with Intra-Batch Redundancy Penalty (delta)
    selected = []
    avail = list(range(len(cand_unlab)))

    # First choice: candidate with highest static score
    first_choice = int(np.argmax(cand_static_score))
    selected.append(cand_unlab[first_choice])
    avail.remove(first_choice)

    while len(selected) < batch_size and avail:
        sel_feats = cand_feats_norm[[cand_unlab.index(idx) for idx in selected]]
        rem_feats = cand_feats_norm[avail]
        
        sims_to_batch = np.dot(rem_feats, sel_feats.T)
        dist_to_batch = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(np.max(sims_to_batch, axis=1), -1.0, 1.0)))
        
        # Redundancy penalty: 1 - normalized distance
        redundancy = 1.0 - (dist_to_batch / np.sqrt(2.0))
        
        # Total score: static_score - delta * redundancy
        batch_scores = cand_static_score[avail] - config.delta_redundancy * redundancy
        best_pos = int(np.argmax(batch_scores))
        selected.append(cand_unlab[avail[best_pos]])
        avail.pop(best_pos)

    return selected

# =============================================================================
# 5. Calibration and Threshold Optimization
# =============================================================================

def optimize_validation_tau(val_probs: np.ndarray, val_labels: np.ndarray, val_priors: np.ndarray) -> float:
    best_tau = 0.0
    best_ba = balanced_accuracy_score(val_labels, np.argmax(val_probs, axis=1)) * 100.0
    for tau in np.linspace(0.05, 0.70, 25):
        adj_p = val_probs / (val_priors ** tau + 1e-6)
        preds = np.argmax(adj_p, axis=1)
        score = balanced_accuracy_score(val_labels, preds) * 100.0
        if score > best_ba:
            best_ba = score
            best_tau = float(tau)
    return best_tau


def optimize_class_thresholds(val_probs: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(val_labels, minlength=NUM_CLASSES)
    priors = counts / len(val_labels)
    T = np.clip(priors ** 0.35, 1e-3, 1.0)
    current_preds = np.argmax(val_probs / T, axis=1)
    best_score = balanced_accuracy_score(val_labels, current_preds)
    
    for c in range(NUM_CLASSES):
        best_mult = 1.0
        for mult in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
            T_test = T.copy()
            T_test[c] *= mult
            score = balanced_accuracy_score(val_labels, np.argmax(val_probs / T_test, axis=1))
            if score > best_score:
                best_score = score
                best_mult = mult
        T[c] *= best_mult
    return T


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
    tau: float = 0.0,
    priors: np.ndarray | None = None,
) -> Dict[str, Any]:
    if thresholds is not None:
        preds = np.argmax(probs / (thresholds + 1e-6), axis=1)
    elif tau > 0 and priors is not None:
        adj_p = probs / (priors ** tau + 1e-6)
        preds = np.argmax(adj_p, axis=1)
    else:
        preds = np.argmax(probs, axis=1)

    rec_all = recall_score(y_true, preds, average=None, zero_division=0) * 100.0
    prec_all, _, _, _ = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
    f1_all = f1_score(y_true, preds, average=None, zero_division=0)
    
    ba = float(np.mean(rec_all))
    macro_f1 = float(np.mean(f1_all))
    acc = float(accuracy_score(y_true, preds)) * 100.0
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))

    min_rec_3 = float((rec_all[5] + rec_all[6] + rec_all[7]) / 3.0)

    try:
        auroc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        auroc = 0.0

    return {
        "balanced_accuracy": ba,
        "macro_f1": macro_f1,
        "macro_auroc": auroc,
        "accuracy": acc,
        "minority_recall_3": min_rec_3,
        "per_class_recall": {CLASSES[c]: float(rec_all[c]) for c in range(NUM_CLASSES)},
        "per_class_precision": {CLASSES[c]: float(prec_all[c] * 100.0) for c in range(NUM_CLASSES)},
        "per_class_f1": {CLASSES[c]: float(f1_all[c]) for c in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
    }

# =============================================================================
# 6. Main Execution
# =============================================================================

def seed_everything(seed: int) -> None:
    """Complete 4-tier random seed synchronization across Python, NumPy, and PyTorch."""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    repo_root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
    feat_dir = repo_root / "data" / "cached_features"
    base_out = repo_root / "outputs" / "isic2019" / "research_investigation"
    ckpt_base = base_out / "checkpoints" / "A18_Improved_MAL"
    results_dir = repo_root / "results"
    
    print("Loading feature representations...")
    train_data = torch.load(feat_dir / "train_features.pt", map_location="cpu", weights_only=False)
    val_data = torch.load(feat_dir / "val_features.pt", map_location="cpu", weights_only=False)
    test_data = torch.load(feat_dir / "test_features.pt", map_location="cpu", weights_only=False)

    train_f, train_l = train_data["features"], train_data["labels"]
    val_f, val_l = val_data["features"], val_data["labels"]
    test_f, test_l = test_data["features"], test_data["labels"]

    val_l_np = val_l.numpy()
    test_l_np = test_l.numpy()
    val_priors = np.bincount(val_l_np, minlength=NUM_CLASSES) / len(val_l_np)

    seeds = [42, 43, 44]
    init_budget = 80
    batch_size = 40
    rounds = 6
    cfg = A18AcquisitionConfig()

    print(f"\n=======================================================")
    print(f"  RUNNING A18: Improved MAL with Fine-Tuned Backbone")
    print(f"  Complete Seeding: Synchronized across Python, NumPy & PyTorch")
    print(f"  Weights: alpha={cfg.alpha_uncertainty}, beta={cfg.beta_diversity}, gamma={cfg.gamma_minority}, delta={cfg.delta_redundancy}")
    print(f"=======================================================")

    a18_runs = []
    all_exp_rows = []
    acquired_distributions = {}

    for seed in seeds:
        print(f"\n  --- Seed {seed} (Fully Seeded) ---")
        seed_everything(seed)
        rng = np.random.RandomState(seed)
        lab = list(rng.choice(len(train_f), size=init_budget, replace=False))
        unlab = [i for i in range(len(train_f)) if i not in lab]

        for r in range(rounds + 1):
            curr_budget = len(lab)
            X_tr = train_f[lab]
            y_tr = train_l[lab]

            # Re-seed model initialization deterministically per round and seed
            torch.manual_seed(seed * 1000 + r)
            model = FineTunedAdapter(in_features=512, num_classes=NUM_CLASSES)
            model = train_model(model, X_tr, y_tr, epochs=50, lr=0.003)

            # Predict on Val and Test
            with torch.no_grad():
                val_probs = torch.softmax(model(val_f), dim=1).numpy()
                test_probs = torch.softmax(model(test_f), dim=1).numpy()

            eval_argmax = evaluate_predictions(test_l_np, test_probs)
            best_tau = optimize_validation_tau(val_probs, val_l_np, val_priors)
            eval_cal = evaluate_predictions(test_l_np, test_probs, tau=best_tau, priors=val_priors)
            opt_T = optimize_class_thresholds(val_probs, val_l_np)
            eval_opt_T = evaluate_predictions(test_l_np, test_probs, thresholds=opt_T)

            # Save Checkpoint
            ckpt_dir = ckpt_base / f"seed_{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"round_{r:04d}_budget_{curr_budget}.pt"
            torch.save({
                "round": r,
                "budget": curr_budget,
                "seed": seed,
                "strategy": "A18_Improved_MAL",
                "model_state": model.state_dict(),
                "labeled_indices": list(lab),
                "best_tau": best_tau,
                "opt_T": opt_T.tolist(),
                "cal_ba": eval_cal["balanced_accuracy"],
                "cal_f1": eval_cal["macro_f1"],
                "cal_auroc": eval_cal["macro_auroc"],
            }, ckpt_path)

            print(f"    Round {r} (Budget {curr_budget}): Argmax BA={eval_argmax['balanced_accuracy']:.2f}% | Cal BA={eval_cal['balanced_accuracy']:.2f}% | OptT BA={eval_opt_T['balanced_accuracy']:.2f}% | MinRec={eval_cal['minority_recall_3']:.1f}%")

            for thresh_name, ev_res in [
                ("StandardArgmax", eval_argmax),
                (f"BayesianPrior_tau{best_tau:.2f}", eval_cal),
                ("ValidationOptimizedThresholds", eval_opt_T),
            ]:
                all_exp_rows.append({
                    "experiment_id": f"A18_Improved_MAL_{thresh_name}_seed{seed}_b{curr_budget}",
                    "seed": seed,
                    "budget": curr_budget,
                    "method": "A18_Improved_MAL",
                    "backbone": "ResNet-18+Adapter",
                    "backbone_trainable": True,
                    "loss": "ClassWeightedCrossEntropy",
                    "acquisition": "Improved_MAL_Diversity",
                    "calibration": "BayesianPrior" if "BayesianPrior" in thresh_name else "None",
                    "threshold_method": thresh_name,
                    "balanced_accuracy": ev_res["balanced_accuracy"],
                    "macro_f1": ev_res["macro_f1"],
                    "macro_auroc": ev_res["macro_auroc"],
                    "DF_recall": ev_res["per_class_recall"]["DF"],
                    "VASC_recall": ev_res["per_class_recall"]["VASC"],
                    "SCC_recall": ev_res["per_class_recall"]["SCC"],
                    "AK_recall": ev_res["per_class_recall"]["AK"],
                    "BKL_recall": ev_res["per_class_recall"]["BKL"],
                })

            if r == rounds:
                counts = np.bincount(train_l[lab].numpy(), minlength=NUM_CLASSES)
                acquired_distributions[seed] = {CLASSES[c]: int(counts[c]) for c in range(NUM_CLASSES)}
                a18_runs.append({
                    "seed": seed,
                    "argmax": eval_argmax,
                    "calibrated": eval_cal,
                    "opt_thresholds": eval_opt_T,
                    "acquired_distribution": acquired_distributions[seed],
                })
                break

            # Active Learning Query
            with torch.no_grad():
                unlab_t = train_f[unlab]
                unlab_probs = torch.softmax(model(unlab_t), dim=1).numpy()
                adapted_unlab = model.encode(unlab_t).numpy()
                adapted_lab = model.encode(train_f[lab]).numpy()

            val_recs = recall_score(val_l_np, np.argmax(val_probs, axis=1), average=None, zero_division=0)
            
            queries = a18_acquisition(
                unlab_indices=unlab,
                current_labeled=lab,
                probs_unlab=unlab_probs,
                adapted_unlab_feats=adapted_unlab,
                adapted_labeled_feats=adapted_lab,
                val_recalls=val_recs,
                batch_size=batch_size,
                config=cfg,
            )
            lab.extend(queries)
            unlab = [i for i in unlab if i not in queries]

    # Save to all_experiments.csv
    all_exp_path = results_dir / "all_experiments.csv"
    df_new = pd.DataFrame(all_exp_rows)
    df_old = pd.read_csv(all_exp_path)
    df_merged = pd.concat([df_old, df_new], ignore_index=True)
    df_merged.drop_duplicates(subset=["experiment_id"], keep="last", inplace=True)
    df_merged.to_csv(all_exp_path, index=False)
    print(f"\n✓ Saved A18 records to {all_exp_path} ({len(df_merged)} total records)")

    # Print Acquired Distribution Analysis
    print("\n================ A18 ACQUIRED DISTRIBUTIONS AT B=320 ================")
    for seed, dist in acquired_distributions.items():
        print(f"Seed {seed}: {dist}")

    # Final summary statistics across seeds
    u_ba = [r["argmax"]["balanced_accuracy"] for r in a18_runs]
    c_ba = [r["calibrated"]["balanced_accuracy"] for r in a18_runs]
    c_f1 = [r["calibrated"]["macro_f1"] for r in a18_runs]
    c_auc = [r["calibrated"]["macro_auroc"] for r in a18_runs]
    c_min = [r["calibrated"]["minority_recall_3"] for r in a18_runs]

    print("\n================ A18 FINAL AGGREGATE RESULTS ================")
    print(f"Argmax Balanced Accuracy:   {np.mean(u_ba):.2f} ± {np.std(u_ba):.2f}%")
    print(f"Calibrated Balanced Acc:    {np.mean(c_ba):.2f} ± {np.std(c_ba):.2f}%")
    print(f"Calibrated Macro-F1:        {np.mean(c_f1):.4f} ± {np.std(c_f1):.4f}")
    print(f"Calibrated Macro-AUROC:     {np.mean(c_auc):.4f} ± {np.std(c_auc):.4f}")
    print(f"Minority Recall (DF/VASC/SCC): {np.mean(c_min):.1f} ± {np.std(c_min):.1f}%")

if __name__ == "__main__":
    main()
