"""Fast Active Learning Benchmark Suite for CSDA-AL on ISIC 2019.

Executes all benchmark methods on cached ResNet-18 representations:
- B0: Random Baseline (both unweighted and class-weighted)
- B1: Supervised Upper Bound (all 17,732 training samples)
- M1: SUPClust (Cold-start clustering + density + boundary scoring)
- M2: Flexible Sampling (Ju et al., MICCAI 2022 validation difficulty curriculum)
- M3: Bengar Class-Balanced Active Learning (Bengar et al., WACV 2022 quota deficits)
- M4: Entropy Sampling (Predictive Shannon Entropy)
- M5: Core-Set (k-Center greedy geometric coverage)
- M6: BALD (Bayesian Active Learning by Disagreement)
- M7: Difficulty-Aware Sampling
- C1: SUPClust + Flexible Sampling
- C2: Bengar + Flexible Sampling
- C3: SUPClust + Bengar
- C4: Full CSDA-AL Combined (SUPClust + Bengar Quotas + Difficulty Curriculum + Class-Aware Stopping)
- C4 Ablations:
    - no_stopping
    - no_bengar
    - no_difficulty
"""
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8

class LinearClassifier(nn.Module):
    def __init__(self, in_dim=512, num_classes=8):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_classifier(X_train: torch.Tensor, y_train: torch.Tensor, num_classes=8, epochs=40, lr=0.01, class_weighted=True):
    model = LinearClassifier(in_dim=X_train.shape[1], num_classes=num_classes)
    
    if class_weighted:
        counts = torch.bincount(y_train, minlength=num_classes).float()
        total = len(y_train)
        weights = total / (counts + 1e-5)
        weights = weights / weights.sum() * num_classes
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model

def evaluate_classifier(model: nn.Module, X_eval: torch.Tensor, y_eval: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_eval)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    y_true = y_eval.cpu().numpy()
    
    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    
    # AUROC
    try:
        # Multi-class One-vs-Rest AUROC
        y_onehot = np.zeros((len(y_true), NUM_CLASSES))
        for c in range(NUM_CLASSES):
            y_onehot[y_true == c, c] = 1.0
        aurocs = []
        for c in range(NUM_CLASSES):
            if np.sum(y_onehot[:, c]) > 0:
                aurocs.append(roc_auc_score(y_onehot[:, c], probs[:, c]))
            else:
                aurocs.append(0.5)
        macro_auroc = float(np.mean(aurocs))
    except Exception:
        macro_auroc = 0.5
        aurocs = [0.5] * NUM_CLASSES

    # Confusion matrix
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, preds):
        conf_mat[t, p] += 1

    per_class = []
    for c in range(NUM_CLASSES):
        per_class.append({
            "class_index": c,
            "class_name": CLASS_NAMES[c],
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "support": int(supp[c]),
            "predicted_count": int(np.sum(preds == c)),
            "auroc": float(aurocs[c])
        })
        
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_auroc": macro_auroc,
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "probabilities": probs.tolist(),
        "predictions": preds.tolist()
    }

# ----------------- ACQUISITION STRATEGIES ----------------- #

def get_supclust_cold_start(features: np.ndarray, n_init: int = 80, seed: int = 42) -> List[int]:
    """SUPClust cold start: k-means clustering in representation space,
    selecting cluster centers (typicality) and boundary instances."""
    rng = np.random.RandomState(seed)
    kmeans = MiniBatchKMeans(n_clusters=n_init // 2, random_state=seed, batch_size=256, n_init=3)
    kmeans.fit(features)
    centers = kmeans.cluster_centers_
    
    selected_indices = []
    # For each cluster, pick the sample closest to the centroid (density/typicality)
    for c in range(len(centers)):
        cluster_mask = np.where(kmeans.labels_ == c)[0]
        if len(cluster_mask) == 0:
            continue
        dists = np.linalg.norm(features[cluster_mask] - centers[c], axis=1)
        best_idx = cluster_mask[np.argmin(dists)]
        selected_indices.append(int(best_idx))
        
    # Pick boundary instances (furthest from centroid in large clusters)
    remaining_budget = n_init - len(selected_indices)
    candidate_boundary = []
    for idx in range(len(features)):
        if idx not in selected_indices:
            dist_to_center = np.linalg.norm(features[idx] - centers[kmeans.labels_[idx]])
            candidate_boundary.append((dist_to_center, idx))
    candidate_boundary.sort(key=lambda x: x[0], reverse=True)
    for _, idx in candidate_boundary[:remaining_budget]:
        selected_indices.append(int(idx))
        
    return selected_indices

def select_coreset(features: np.ndarray, unlabelled_indices: List[int], labelled_indices: List[int], batch_size: int) -> List[int]:
    """Core-Set greedy k-center."""
    unlabelled_feats = features[unlabelled_indices]
    labelled_feats = features[labelled_indices]
    
    # Distance from each unlabelled point to closest labelled point
    min_dists = np.min(np.linalg.norm(unlabelled_feats[:, None, :] - labelled_feats[None, :, :], axis=2), axis=1)
    
    chosen = []
    for _ in range(batch_size):
        idx = int(np.argmax(min_dists))
        chosen.append(unlabelled_indices[idx])
        # Update distances
        new_dist = np.linalg.norm(unlabelled_feats - unlabelled_feats[idx], axis=1)
        min_dists = np.minimum(min_dists, new_dist)
        min_dists[idx] = -1.0
    return chosen

def acquire_samples(
    strategy_name: str,
    probs_unlabelled: np.ndarray,
    features: np.ndarray,
    unlabelled_indices: List[int],
    labelled_indices: List[int],
    labelled_labels: List[int],
    val_metrics: Dict,
    batch_size: int = 40,
    seed: int = 42
) -> List[int]:
    rng = np.random.RandomState(seed)
    
    if strategy_name == "random":
        return list(rng.choice(unlabelled_indices, size=batch_size, replace=False))
        
    elif strategy_name == "entropy":
        # Shannon Entropy H(x) = -sum p * log(p)
        eps = 1e-12
        entropy = -np.sum(probs_unlabelled * np.log(probs_unlabelled + eps), axis=1)
        ranked = np.argsort(-entropy)
        return [unlabelled_indices[i] for i in ranked[:batch_size]]
        
    elif strategy_name == "bald":
        # Simulated BALD via pseudo-Bayesian variation
        eps = 1e-12
        entropy = -np.sum(probs_unlabelled * np.log(probs_unlabelled + eps), axis=1)
        confidence = np.max(probs_unlabelled, axis=1)
        bald_scores = entropy * (1.0 - confidence)
        ranked = np.argsort(-bald_scores)
        return [unlabelled_indices[i] for i in ranked[:batch_size]]

    elif strategy_name == "coreset":
        return select_coreset(features, unlabelled_indices, labelled_indices, batch_size)

    elif strategy_name == "flex_sampling":
        # Flexible Sampling: Weight unlabelled samples inversely by validation per-class recall
        val_recalls = np.array([pc["recall"] for pc in val_metrics["per_class"]])
        diff_weights = 1.0 - val_recalls
        diff_weights = np.maximum(diff_weights, 0.05)
        diff_weights /= np.sum(diff_weights)
        
        # Candidate difficulty is dot product of predicted probs and class difficulty weights
        candidate_diff = np.dot(probs_unlabelled, diff_weights)
        eps = 1e-12
        entropy = -np.sum(probs_unlabelled * np.log(probs_unlabelled + eps), axis=1)
        scores = entropy * candidate_diff
        ranked = np.argsort(-scores)
        return [unlabelled_indices[i] for i in ranked[:batch_size]]

    elif strategy_name == "bengar":
        # Bengar Class-Balanced: Allocate quota deficits T_c = max(0, Target - Count_c)
        counts = np.bincount(labelled_labels, minlength=NUM_CLASSES)
        target = (len(labelled_labels) + batch_size) / NUM_CLASSES
        deficits = np.maximum(0, target - counts)
        if np.sum(deficits) == 0:
            quotas = np.full(NUM_CLASSES, batch_size // NUM_CLASSES)
        else:
            quotas = np.round(batch_size * (deficits / np.sum(deficits))).astype(int)
        # Fix rounding
        while np.sum(quotas) < batch_size:
            quotas[np.argmin(counts)] += 1
        while np.sum(quotas) > batch_size:
            quotas[np.argmax(quotas)] -= 1
            
        pred_classes = np.argmax(probs_unlabelled, axis=1)
        confidences = np.max(probs_unlabelled, axis=1)
        eps = 1e-12
        entropy = -np.sum(probs_unlabelled * np.log(probs_unlabelled + eps), axis=1)
        
        selected = []
        for c in range(NUM_CLASSES):
            q_c = quotas[c]
            c_mask = np.where(pred_classes == c)[0]
            if len(c_mask) == 0:
                continue
            # Rank by entropy within predicted class
            c_ranked = c_mask[np.argsort(-entropy[c_mask])]
            for idx in c_ranked[:q_c]:
                selected.append(unlabelled_indices[idx])
                
        # Fill underflow with top remaining entropy
        remaining = [idx for idx in unlabelled_indices if idx not in selected]
        if len(selected) < batch_size:
            rem_entropy = -np.sum(probs_unlabelled[[unlabelled_indices.index(i) for i in remaining]] * np.log(probs_unlabelled[[unlabelled_indices.index(i) for i in remaining]] + eps), axis=1)
            rem_ranked = [remaining[i] for i in np.argsort(-rem_entropy)]
            selected.extend(rem_ranked[:batch_size - len(selected)])
        return selected[:batch_size]

    elif strategy_name == "difficulty_aware":
        val_recalls = np.array([pc["recall"] for pc in val_metrics["per_class"]])
        class_error = 1.0 - val_recalls
        sample_diff = np.dot(probs_unlabelled, class_error)
        ranked = np.argsort(-sample_diff)
        return [unlabelled_indices[i] for i in ranked[:batch_size]]

    elif strategy_name in ["c1_supclust_flex", "c2_bengar_flex", "c3_supclust_bengar", "c4_csda_al", "c4_no_stop", "c4_no_bengar", "c4_no_diff"]:
        # CSDA-AL Combined Strategy
        use_bengar = strategy_name not in ["c1_supclust_flex", "c4_no_bengar"]
        use_diff = strategy_name not in ["c3_supclust_bengar", "c4_no_diff"]
        
        eps = 1e-12
        uncertainty = -np.sum(probs_unlabelled * np.log(probs_unlabelled + eps), axis=1)
        uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-8)
        
        if use_diff:
            val_recalls = np.array([pc["recall"] for pc in val_metrics["per_class"]])
            diff_weights = 1.0 - val_recalls
            diff_weights = np.maximum(diff_weights, 0.05)
            diff_weights /= np.sum(diff_weights)
            candidate_diff = np.dot(probs_unlabelled, diff_weights)
            candidate_diff = (candidate_diff - np.min(candidate_diff)) / (np.max(candidate_diff) - np.min(candidate_diff) + 1e-8)
            combined_score = 0.5 * uncertainty + 0.5 * candidate_diff
        else:
            combined_score = uncertainty
            
        if use_bengar:
            counts = np.bincount(labelled_labels, minlength=NUM_CLASSES)
            target = (len(labelled_labels) + batch_size) / NUM_CLASSES
            deficits = np.maximum(0, target - counts)
            if np.sum(deficits) == 0:
                quotas = np.full(NUM_CLASSES, batch_size // NUM_CLASSES)
            else:
                quotas = np.round(batch_size * (deficits / np.sum(deficits))).astype(int)
            while np.sum(quotas) < batch_size:
                quotas[np.argmin(counts)] += 1
            while np.sum(quotas) > batch_size:
                quotas[np.argmax(quotas)] -= 1
                
            pred_classes = np.argmax(probs_unlabelled, axis=1)
            selected = []
            for c in range(NUM_CLASSES):
                q_c = quotas[c]
                c_mask = np.where(pred_classes == c)[0]
                if len(c_mask) == 0:
                    continue
                c_ranked = c_mask[np.argsort(-combined_score[c_mask])]
                for idx in c_ranked[:q_c]:
                    selected.append(unlabelled_indices[idx])
                    
            remaining = [idx for idx in unlabelled_indices if idx not in selected]
            if len(selected) < batch_size:
                rem_indices_in_unlab = [unlabelled_indices.index(i) for i in remaining]
                rem_scores = combined_score[rem_indices_in_unlab]
                rem_ranked = [remaining[i] for i in np.argsort(-rem_scores)]
                selected.extend(rem_ranked[:batch_size - len(selected)])
            return selected[:batch_size]
        else:
            ranked = np.argsort(-combined_score)
            return [unlabelled_indices[i] for i in ranked[:batch_size]]

    else:
        return list(rng.choice(unlabelled_indices, size=batch_size, replace=False))


def run_experiment(
    method_name: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    seed: int = 42,
    num_rounds: int = 7,
    n_init: int = 80,
    batch_size: int = 40,
    class_weighted: bool = True
) -> Dict:
    rng = np.random.RandomState(seed)
    N = len(features)
    all_indices = list(range(N))
    
    # Cold-start initialization
    if method_name in ["m1_supclust", "c1_supclust_flex", "c3_supclust_bengar", "c4_csda_al", "c4_no_stop", "c4_no_bengar", "c4_no_diff"]:
        initial_indices = get_supclust_cold_start(features.numpy(), n_init=n_init, seed=seed)
    else:
        initial_indices = list(rng.choice(all_indices, size=n_init, replace=False))
        
    labelled_indices = list(initial_indices)
    unlabelled_indices = [i for i in all_indices if i not in labelled_indices]
    
    history = []
    stopped_early = False
    stop_round = num_rounds
    
    # Class-aware stopping state
    patience = 2
    no_improve_count = 0
    best_macro_f1 = 0.0
    
    for r in range(num_rounds):
        # 1. Train classifier on labelled set
        X_train = features[labelled_indices]
        y_train = labels[labelled_indices]
        model = train_classifier(X_train, y_train, epochs=40, lr=0.01, class_weighted=class_weighted)
        
        # 2. Evaluate on validation set
        val_eval = evaluate_classifier(model, val_features, val_labels)
        
        # 3. Predict on unlabelled pool
        model.eval()
        with torch.no_grad():
            unlab_logits = model(features[unlabelled_indices])
            unlab_probs = torch.softmax(unlab_logits, dim=1).numpy()
            
        cur_labels = labels[labelled_indices].tolist()
        class_counts = np.bincount(cur_labels, minlength=NUM_CLASSES).tolist()
        
        round_record = {
            "round": r,
            "budget": len(labelled_indices),
            "class_counts": class_counts,
            "accuracy": val_eval["accuracy"],
            "balanced_accuracy": val_eval["balanced_accuracy"],
            "macro_f1": val_eval["macro_f1"],
            "macro_precision": val_eval["macro_precision"],
            "macro_recall": val_eval["macro_recall"],
            "macro_auroc": val_eval["macro_auroc"],
            "per_class": val_eval["per_class"],
            "confusion_matrix": val_eval["confusion_matrix"]
        }
        history.append(round_record)
        
        # Check Class-Aware Stopping
        if "c4" in method_name and "no_stop" not in method_name and r >= 4:
            f1 = val_eval["macro_f1"]
            min_support = min([c for c in class_counts])
            if f1 > best_macro_f1 + 0.005:
                best_macro_f1 = f1
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience and min_support >= 3:
                stopped_early = True
                stop_round = r + 1
                print(f"  [{method_name} seed={seed}] Class-aware stopping triggered at Round {r+1} (Budget={len(labelled_indices)})")
                break
                
        if r < num_rounds - 1:
            # Acquire next batch
            new_queries = acquire_samples(
                strategy_name=method_name,
                probs_unlabelled=unlab_probs,
                features=features.numpy(),
                unlabelled_indices=unlabelled_indices,
                labelled_indices=labelled_indices,
                labelled_labels=cur_labels,
                val_metrics=val_eval,
                batch_size=batch_size,
                seed=seed + r
            )
            labelled_indices.extend(new_queries)
            unlabelled_indices = [i for i in unlabelled_indices if i not in new_queries]

    # Final test evaluation on test set
    test_eval = evaluate_classifier(model, test_features, test_labels)
    
    return {
        "method": method_name,
        "seed": seed,
        "stopped_early": stopped_early,
        "stop_round": stop_round,
        "final_budget": len(labelled_indices),
        "val_history": history,
        "final_val_metrics": history[-1],
        "final_test_metrics": {
            "accuracy": test_eval["accuracy"],
            "balanced_accuracy": test_eval["balanced_accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "macro_precision": test_eval["macro_precision"],
            "macro_recall": test_eval["macro_recall"],
            "macro_auroc": test_eval["macro_auroc"],
            "per_class": test_eval["per_class"],
            "confusion_matrix": test_eval["confusion_matrix"]
        }
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    feat_dir = repo_root / "outputs" / "cached_features"
    results_dir = repo_root / "results"
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_pt = feat_dir / "train_features.pt"
    if not train_pt.exists():
        print(f"[Notice] Cached features file not found at {train_pt}.")
        print("All benchmark comparison matrices and figures are already pre-computed in results/tables/ and results/figures/!")
        print("To re-run extraction, place ISIC 2019 images in data/ and run scripts/cache_all_features.py.")
        return

    print("Loading cached features...")
    train_cache = torch.load(feat_dir / "train_features.pt", map_location="cpu")
    val_cache = torch.load(feat_dir / "val_features.pt", map_location="cpu")
    test_cache = torch.load(feat_dir / "test_features.pt", map_location="cpu")

    features = train_cache["features"]
    labels = train_cache["labels"].long()
    val_features = val_cache["features"]
    val_labels = val_cache["labels"].long()
    test_features = test_cache["features"]
    test_labels = test_cache["labels"].long()

    print(f"Loaded train: {features.shape}, val: {val_features.shape}, test: {test_features.shape}")

    # Methods to benchmark
    methods = [
        "random",               # B0 Baseline (class-weighted)
        "b0_unweighted",        # B0 Baseline (unweighted, exact reproduction)
        "m1_supclust",          # M1: SUPClust
        "m2_flex_sampling",     # M2: Flexible Sampling
        "m3_bengar",            # M3: Bengar Class-Balanced
        "m4_entropy",           # M4: Entropy
        "m5_coreset",           # M5: Core-Set
        "m6_bald",              # M6: BALD
        "m7_difficulty_aware",  # M7: Difficulty-Aware
        "c1_supclust_flex",     # C1: SUPClust + Flex
        "c2_bengar_flex",       # C2: Bengar + Flex
        "c3_supclust_bengar",   # C3: SUPClust + Bengar
        "c4_csda_al",           # C4: Full CSDA-AL
        "c4_no_stop",           # Ablation: C4 without stopping
        "c4_no_bengar",         # Ablation: C4 without Bengar
        "c4_no_diff",           # Ablation: C4 without difficulty
    ]

    seeds = [42, 123, 999]
    all_results = {}
    summary_rows = []

    print("\n================ STARTING BENCHMARK SUITE ================\n")
    t0 = time.time()

    # First run B1: Supervised Upper Bound (all 17,732 training samples)
    print("Running B1: Supervised Upper Bound (Full 17,732 training set)...")
    b1_model = train_classifier(features, labels, epochs=50, lr=0.01, class_weighted=True)
    b1_val = evaluate_classifier(b1_model, val_features, val_labels)
    b1_test = evaluate_classifier(b1_model, test_features, test_labels)
    b1_record = {
        "method": "b1_upper_bound",
        "seed": 42,
        "stopped_early": False,
        "stop_round": 1,
        "final_budget": len(features),
        "val_history": [{"round": 0, "budget": len(features), **b1_val}],
        "final_val_metrics": b1_val,
        "final_test_metrics": b1_test
    }
    all_results["b1_upper_bound"] = [b1_record]
    summary_rows.append({
        "Method": "B1: Upper Bound (All 17,732)",
        "Budget": len(features),
        "Val Bal-Acc": f"{b1_val['balanced_accuracy']*100:.2f}%",
        "Val Macro-F1": f"{b1_val['macro_f1']:.4f}",
        "Val Macro-AUROC": f"{b1_val['macro_auroc']:.4f}",
        "Test Bal-Acc": f"{b1_test['balanced_accuracy']*100:.2f}%",
        "Test Macro-F1": f"{b1_test['macro_f1']:.4f}",
        "Test Macro-AUROC": f"{b1_test['macro_auroc']:.4f}"
    })
    print(f"  B1 Upper Bound -> Val Bal-Acc: {b1_val['balanced_accuracy']*100:.2f}%, Macro-F1: {b1_val['macro_f1']:.4f}, AUROC: {b1_val['macro_auroc']:.4f}")

    # Run all AL methods across seeds
    for m in methods:
        all_results[m] = []
        seed_val_bal_accs = []
        seed_val_f1s = []
        seed_val_aurocs = []
        seed_test_bal_accs = []
        seed_test_f1s = []
        seed_test_aurocs = []
        budgets = []

        cw = (m != "b0_unweighted")
        actual_m = "random" if m == "b0_unweighted" else m

        print(f"\n--- Running Method: {m} across {len(seeds)} seeds ---")
        for s in seeds:
            res = run_experiment(
                method_name=actual_m,
                features=features,
                labels=labels,
                val_features=val_features,
                val_labels=val_labels,
                test_features=test_features,
                test_labels=test_labels,
                seed=s,
                class_weighted=cw
            )
            res["method"] = m
            all_results[m].append(res)
            
            vf = res["final_val_metrics"]
            tf = res["final_test_metrics"]
            seed_val_bal_accs.append(vf["balanced_accuracy"])
            seed_val_f1s.append(vf["macro_f1"])
            seed_val_aurocs.append(vf["macro_auroc"])
            seed_test_bal_accs.append(tf["balanced_accuracy"])
            seed_test_f1s.append(tf["macro_f1"])
            seed_test_aurocs.append(tf["macro_auroc"])
            budgets.append(res["final_budget"])
            
            print(f"  [Seed {s}] Budget: {res['final_budget']} | Val Bal-Acc: {vf['balanced_accuracy']*100:.2f}% | Val Macro-F1: {vf['macro_f1']:.4f} | Val AUROC: {vf['macro_auroc']:.4f}")

        # Summary for method
        mean_v_ba = np.mean(seed_val_bal_accs) * 100
        std_v_ba = np.std(seed_val_bal_accs) * 100
        mean_v_f1 = np.mean(seed_val_f1s)
        std_v_f1 = np.std(seed_val_f1s)
        mean_v_auc = np.mean(seed_val_aurocs)
        std_v_auc = np.std(seed_val_aurocs)

        mean_t_ba = np.mean(seed_test_bal_accs) * 100
        std_t_ba = np.std(seed_test_bal_accs) * 100
        mean_t_f1 = np.mean(seed_test_f1s)
        std_t_f1 = np.std(seed_test_f1s)
        mean_t_auc = np.mean(seed_test_aurocs)
        std_t_auc = np.std(seed_test_aurocs)

        summary_rows.append({
            "Method": m,
            "Budget": f"{int(np.mean(budgets))}",
            "Val Bal-Acc": f"{mean_v_ba:.2f} ± {std_v_ba:.2f}%",
            "Val Macro-F1": f"{mean_v_f1:.4f} ± {std_v_f1:.4f}",
            "Val Macro-AUROC": f"{mean_v_auc:.4f} ± {std_v_auc:.4f}",
            "Test Bal-Acc": f"{mean_t_ba:.2f} ± {std_t_ba:.2f}%",
            "Test Macro-F1": f"{mean_t_f1:.4f} ± {std_t_f1:.4f}",
            "Test Macro-AUROC": f"{mean_t_auc:.4f} ± {std_t_auc:.4f}"
        })

    total_time = time.time() - t0
    print(f"\nAll experiments finished in {total_time:.1f}s ({total_time/60:.2f} minutes)!")

    # Save summary table
    df_summary = pd.DataFrame(summary_rows)
    csv_path = tables_dir / "benchmark_comparison_matrix.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved benchmark summary table to {csv_path}")
    print("\nBenchmark Comparison Matrix:")
    print(df_summary.to_string())

    # Save individual run files
    for m, runs in all_results.items():
        m_dir = runs_dir / m
        m_dir.mkdir(parents=True, exist_ok=True)
        # Save run 0 as reference
        ref_run = runs[0]
        with open(m_dir / "round_0007.json", "w") as f:
            json.dump(ref_run["final_val_metrics"], f, indent=2)
        with open(m_dir / "full_experiment.json", "w") as f:
            json.dump(runs, f, indent=2)

        # Save per-class and confusion csv for seed 42
        df_pc = pd.DataFrame(ref_run["final_val_metrics"]["per_class"])
        df_pc.to_csv(m_dir / "round_0007_per_class.csv", index=False)
        df_cm = pd.DataFrame(ref_run["final_val_metrics"]["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
        df_cm.to_csv(m_dir / "round_0007_confusion.csv")

    # Save overall json metrics
    with open(results_dir / "metrics" / "all_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved all raw experiment records to results/metrics/all_benchmark_results.json")

if __name__ == "__main__":
    main()
