"""Comprehensive Loss Function Comparison Benchmark on ISIC 2019.

Compares 7 loss functions under identical frozen split & seed conditions:
1. Standard Cross Entropy (CE)
2. Class-Weighted Cross Entropy (CW-CE)
3. Effective-Number Class Weighting (Cui et al., CVPR 2019)
4. Focal Loss (gamma=2.0)
5. Class-Balanced Focal Loss (CB-Focal)
6. Balanced Softmax (Ren et al., NeurIPS 2020)
7. Logit Adjustment (Menon et al., ICLR 2021)

Saves EVERY checkpoint for every method, seed, and round into:
outputs/isic2019/research_investigation/checkpoints/loss_comparison/
"""
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    recall_score,
    confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8
MINORITY_CLASSES = [5, 6, 7]  # Strictly DF, VASC, SCC: (DF + VASC + SCC) / 3

# =============================================================================
# Custom Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=1)
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, sample_counts: torch.Tensor):
        super().__init__()
        # pi_c is empirical frequency in the training batch
        counts = torch.clamp(sample_counts, min=1.0)
        self.log_pi = torch.log(counts / counts.sum()).to(sample_counts.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_pi.unsqueeze(0)
        return nn.functional.cross_entropy(adjusted_logits, targets)

class LogitAdjustedLoss(nn.Module):
    def __init__(self, priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.log_priors = torch.log(torch.clamp(priors, min=1e-6)).to(priors.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.tau * self.log_priors.unsqueeze(0)
        return nn.functional.cross_entropy(adjusted_logits, targets)

def compute_effective_number_weights(counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """Class-Balanced Effective Number weights (Cui et al., CVPR 2019): (1 - beta) / (1 - beta^n)"""
    effective_num = 1.0 - torch.pow(beta, counts.clamp(min=1.0))
    weights = (1.0 - beta) / (effective_num + 1e-8)
    present = counts > 0
    if present.sum() > 0:
        weights[~present] = 0.0
        weights = weights / weights[present].sum() * present.sum()
    return weights

# =============================================================================
# Training with Loss Options
# =============================================================================

def train_with_loss(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_name: str,
    epochs: int = 40,
    lr: float = 0.01,
    val_priors: torch.Tensor = None
) -> Tuple[nn.Module, Dict]:
    model = nn.Linear(X_train.shape[1], NUM_CLASSES)
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    total = float(len(y_train))
    present = counts > 0

    if loss_name == "standard_ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "class_weighted_ce":
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if present.sum() > 0:
            raw_w = total / (torch.sqrt(counts[present]) * np.sqrt(total) + 1.0)
            raw_w = torch.clamp(raw_w, min=0.1, max=15.0)
            weights[present] = raw_w / raw_w.sum() * present.sum()
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_name == "effective_number_ce":
        weights = compute_effective_number_weights(counts, beta=0.99)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_name == "focal_loss":
        criterion = FocalLoss(gamma=2.0)
    elif loss_name == "class_balanced_focal":
        weights = compute_effective_number_weights(counts, beta=0.99)
        criterion = FocalLoss(gamma=2.0, weight=weights)
    elif loss_name == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(counts)
    elif loss_name == "logit_adjustment":
        # Logit adjustment using empirical training batch priors
        batch_priors = torch.clamp(counts / total, min=1e-6)
        criterion = LogitAdjustedLoss(batch_priors, tau=1.0)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    history = []
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    model.eval()
    return model, {"loss_history": history, "final_loss": history[-1]}

def evaluate_model(y_true: np.ndarray, probs: np.ndarray, tau: float = 0.0, priors: np.ndarray = None) -> Dict:
    if tau > 0 and priors is not None:
        adj_probs = probs / (priors ** tau + 1e-6)
        preds = np.argmax(adj_probs, axis=1)
    else:
        preds = np.argmax(probs, axis=1)

    ba = float(balanced_accuracy_score(y_true, preds)) * 100.0
    acc = float(accuracy_score(y_true, preds)) * 100.0
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    rec_all = recall_score(y_true, preds, average=None, zero_division=0) * 100.0
    prec_all, _, _, _ = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))

    try:
        auroc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        auroc = 0.0

    return {
        "balanced_accuracy": ba,
        "accuracy": acc,
        "macro_f1": f1,
        "macro_auroc": auroc,
        "minority_recall": float(np.mean([rec_all[c] for c in MINORITY_CLASSES])),
        "per_class_recall": {CLASSES[c]: float(rec_all[c]) for c in range(NUM_CLASSES)},
        "per_class_precision": {CLASSES[c]: float(prec_all[c] * 100.0) for c in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }

def main():
    repo_root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
    feat_dir = repo_root / "data" / "cached_features"
    base_out = repo_root / "outputs" / "isic2019" / "research_investigation"
    out_dir = base_out / "loss_comparison"
    ckpt_dir = base_out / "checkpoints" / "loss_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pre-extracted feature tensors...")
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
    budget = 320

    losses = [
        ("standard_ce", "Standard Cross Entropy"),
        ("class_weighted_ce", "Class-Weighted Cross Entropy (A1 Baseline)"),
        ("effective_number_ce", "Effective-Number Weighting (Cui et al.)"),
        ("focal_loss", "Focal Loss (gamma=2.0)"),
        ("class_balanced_focal", "Class-Balanced Focal Loss"),
        ("balanced_softmax", "Balanced Softmax (Ren et al.)"),
        ("logit_adjustment", "Logit Adjustment (Menon et al.)"),
    ]

    results = {}
    summary_rows = []

    # Generate canonical seed-matched labeled indices (budget=320)
    seed_matched_indices = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        seed_matched_indices[seed] = list(rng.choice(len(train_f), size=budget, replace=False))

    all_exp_rows = []

    for loss_id, loss_name in losses:
        print(f"\n=================================================================")
        print(f"  Evaluating Loss: [{loss_id}] {loss_name}")
        print(f"=================================================================")
        loss_runs = []

        for seed in seeds:
            lab_indices = seed_matched_indices[seed]
            
            X_tr = train_f[lab_indices]
            y_tr = train_l[lab_indices]

            # Fit model with specified loss
            model, train_info = train_with_loss(
                X_tr, y_tr,
                loss_name=loss_id,
                epochs=40,
                lr=0.01,
                val_priors=torch.tensor(val_priors, dtype=torch.float32)
            )

            # SAVE EVERY CHECKPOINT!
            ckpt_path = ckpt_dir / f"{loss_id}_seed_{seed}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "loss_name": loss_id,
                "seed": seed,
                "labeled_indices": lab_indices,
                "train_info": train_info,
                "counts": np.bincount(y_tr.numpy(), minlength=NUM_CLASSES).tolist()
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint: {ckpt_path.name}")

            # Predict on Val and Test
            with torch.no_grad():
                val_logits = model(val_f)
                val_probs = torch.softmax(val_logits, dim=1).numpy()
                test_logits = model(test_f)
                test_probs = torch.softmax(test_logits, dim=1).numpy()

            # Tune calibration threshold tau on Validation set ONLY
            best_tau = 0.0
            best_val_ba = balanced_accuracy_score(val_l_np, np.argmax(val_probs, axis=1)) * 100.0
            for tau_cand in np.linspace(0.05, 0.70, 25):
                adj_p = val_probs / (val_priors ** tau_cand + 1e-6)
                score = balanced_accuracy_score(val_l_np, np.argmax(adj_p, axis=1)) * 100.0
                if score > best_val_ba:
                    best_val_ba = score
                    best_tau = tau_cand

            # Evaluate on locked Test set
            uncal_test = evaluate_model(test_l_np, test_probs, tau=0.0)
            cal_test = evaluate_model(test_l_np, test_probs, tau=best_tau, priors=val_priors)

            loss_runs.append({
                "seed": seed,
                "checkpoint": str(ckpt_path),
                "best_tau": best_tau,
                "uncalibrated": uncal_test,
                "calibrated": cal_test
            })
            print(f"    Seed {seed}: Uncal BA={uncal_test['balanced_accuracy']:.2f}% | Cal BA={cal_test['balanced_accuracy']:.2f}% | Cal F1={cal_test['macro_f1']:.4f} | MinRec(DF/VASC/SCC)={cal_test['minority_recall']:.1f}%")

            # Log to all_experiments row (both uncalibrated and calibrated)
            all_exp_rows.append({
                "experiment_id": f"loss_{loss_id}_uncal_seed{seed}",
                "seed": seed,
                "budget": budget,
                "method": f"Loss_{loss_id}",
                "backbone": "ResNet-18",
                "backbone_trainable": False,
                "loss": loss_id,
                "acquisition": "Random",
                "calibration": "None",
                "threshold_method": "Argmax",
                "balanced_accuracy": uncal_test["balanced_accuracy"],
                "macro_f1": uncal_test["macro_f1"],
                "macro_auroc": uncal_test["macro_auroc"],
                "DF_recall": uncal_test["per_class_recall"]["DF"],
                "VASC_recall": uncal_test["per_class_recall"]["VASC"],
                "SCC_recall": uncal_test["per_class_recall"]["SCC"],
                "AK_recall": uncal_test["per_class_recall"]["AK"],
                "BKL_recall": uncal_test["per_class_recall"]["BKL"],
            })
            all_exp_rows.append({
                "experiment_id": f"loss_{loss_id}_cal_seed{seed}",
                "seed": seed,
                "budget": budget,
                "method": f"Loss_{loss_id}",
                "backbone": "ResNet-18",
                "backbone_trainable": False,
                "loss": loss_id,
                "acquisition": "Random",
                "calibration": f"BayesianPriorAdjustment_tau{best_tau:.2f}",
                "threshold_method": "ValidationCalibrated",
                "balanced_accuracy": cal_test["balanced_accuracy"],
                "macro_f1": cal_test["macro_f1"],
                "macro_auroc": cal_test["macro_auroc"],
                "DF_recall": cal_test["per_class_recall"]["DF"],
                "VASC_recall": cal_test["per_class_recall"]["VASC"],
                "SCC_recall": cal_test["per_class_recall"]["SCC"],
                "AK_recall": cal_test["per_class_recall"]["AK"],
                "BKL_recall": cal_test["per_class_recall"]["BKL"],
            })

        results[loss_id] = {"desc": loss_name, "runs": loss_runs}

        # Calculate summary across seeds
        u_ba = [r["uncalibrated"]["balanced_accuracy"] for r in loss_runs]
        c_ba = [r["calibrated"]["balanced_accuracy"] for r in loss_runs]
        c_f1 = [r["calibrated"]["macro_f1"] for r in loss_runs]
        c_auc = [r["calibrated"]["macro_auroc"] for r in loss_runs]
        c_min = [r["calibrated"]["minority_recall"] for r in loss_runs]

        summary_rows.append({
            "Loss Function": loss_name,
            "Uncal Bal-Acc (%)": f"{np.mean(u_ba):.2f} ± {np.std(u_ba):.2f}",
            "Cal Bal-Acc (%)": f"{np.mean(c_ba):.2f} ± {np.std(c_ba):.2f}",
            "Cal Macro-F1": f"{np.mean(c_f1):.4f} ± {np.std(c_f1):.4f}",
            "Cal Macro-AUROC": f"{np.mean(c_auc):.4f} ± {np.std(c_auc):.4f}",
            "Minority Recall (%)": f"{np.mean(c_min):.1f} ± {np.std(c_min):.1f}%"
        })

    # Save detailed outputs and summary
    with open(out_dir / "loss_comparison_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "loss_comparison_summary.csv", index=False)
    # Also save to results/tables for git tracking
    df_summary.to_csv(repo_root / "results" / "tables" / "loss_comparison_summary.csv", index=False)

    # Save/append to results/all_experiments.csv
    all_exp_path = repo_root / "results" / "all_experiments.csv"
    df_all_new = pd.DataFrame(all_exp_rows)
    if all_exp_path.exists():
        df_existing = pd.read_csv(all_exp_path)
        df_combined = pd.concat([df_existing, df_all_new], ignore_index=True)
    else:
        df_combined = df_all_new
    df_combined.drop_duplicates(subset=["experiment_id"], keep="last", inplace=True)
    df_combined.to_csv(all_exp_path, index=False)
    print(f"  ✓ Updated {all_exp_path} ({len(df_combined)} total experiment records)")

    print("\n================ FINAL LOSS COMPARISON SUMMARY ================\n")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
