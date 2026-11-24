"""Advanced Minority-Aware Active Learning (MAL) with Budget Tracking & Validation Calibration.

Implements Phases 5, 6, 7, 8:
- Minority-probability awareness (P_minority)
- Soft quota candidate filtering without true label access
- Cosine diversity within candidate region
- Loss function comparison (Class-Weighted vs Logit-Adjusted CE)
- Budget-vs-Performance curves (80, 120, 160, 200, 240, 280, 320)
- Validation-tuned Bayesian threshold calibration evaluated on locked test set
- Compares against seed-matched A1 and A9 baselines across seeds 42, 43, 44
"""
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    recall_score
)
import torch
import torch.nn as nn
import torch.optim as optim

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8
MINORITY_CLASSES = [3, 5, 6, 7]  # AK, DF, VASC, SCC

def get_calibrated_cold_start(norm_feats: np.ndarray, n_init: int = 80, seed: int = 42) -> List[int]:
    n_clusters = max(n_init // 2, NUM_CLASSES)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=256, n_init=3)
    kmeans.fit(norm_feats)
    centers = kmeans.cluster_centers_

    selected = []
    for c in range(len(centers)):
        c_mask = np.where(kmeans.labels_ == c)[0]
        if len(c_mask) > 0:
            dists = np.linalg.norm(norm_feats[c_mask] - centers[c], axis=1)
            selected.append(int(c_mask[np.argmin(dists)]))

    rem = n_init - len(selected)
    if rem > 0:
        sel_mat = norm_feats[selected]
        sims = np.dot(norm_feats, sel_mat.T)
        min_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(sims, axis=1)))
        for s in selected:
            min_dists[s] = -1.0
        order = np.argsort(min_dists)[::-1]
        selected.extend([int(idx) for idx in order[:rem]])
    return selected

def train_linear_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 40,
    lr: float = 0.01,
    loss_type: str = "class_weighted",
    val_priors: np.ndarray = None
) -> nn.Module:
    model = nn.Linear(X_train.shape[1], NUM_CLASSES)
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    total = float(len(y_train))
    present = counts > 0

    if loss_type == "class_weighted":
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if present.sum() > 0:
            raw_w = total / (torch.sqrt(counts[present]) * np.sqrt(total) + 1.0)
            raw_w = torch.clamp(raw_w, min=0.1, max=15.0)
            weights[present] = raw_w / raw_w.sum() * present.sum()
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_type == "logit_adjusted":
        # Logit adjustment: loss(y, z + tau * log(pi))
        weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        if loss_type == "logit_adjusted" and val_priors is not None:
            # Shift training logits by empirical log priors
            log_prior = torch.tensor(np.log(val_priors + 1e-6), dtype=torch.float32)
            logits = logits + log_prior
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, tau: float = 0.0, priors: np.ndarray = None) -> Dict:
    if tau > 0 and priors is not None:
        adj_probs = probs / (priors ** tau + 1e-6)
        preds = np.argmax(adj_probs, axis=1)
    else:
        preds = np.argmax(probs, axis=1)

    ba = float(balanced_accuracy_score(y_true, preds)) * 100.0
    acc = float(accuracy_score(y_true, preds)) * 100.0
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    rec_all = recall_score(y_true, preds, average=None, zero_division=0) * 100.0

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
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }

def minority_aware_acquisition(
    unlab_indices: List[int],
    norm_feats: np.ndarray,
    probs_unlab: np.ndarray,
    batch_size: int = 40,
    val_recalls: np.ndarray = None,
    current_labeled: List[int] = None
) -> List[int]:
    """Acquisition policy that targets minority mass + boundary uncertainty with diversity."""
    # 1. Predictive entropy
    p_safe = np.clip(probs_unlab, 1e-12, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
    ent_norm = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy) + 1e-12)

    # 2. Probability mass assigned to underperforming minority classes
    if val_recalls is not None:
        # Weight by deficiency: (1 - recall)
        deficits = np.maximum(0.0, 1.0 - val_recalls)
    else:
        deficits = np.ones(NUM_CLASSES)
        deficits[[0, 1]] = 0.2  # De-emphasize Melanoma & Nevus

    minority_mass = np.sum(probs_unlab * deficits, axis=1)
    min_norm = (minority_mass - np.min(minority_mass)) / (np.max(minority_mass) - np.min(minority_mass) + 1e-12)

    # 3. Combined score
    score = 0.4 * ent_norm + 0.6 * min_norm

    # 4. Soft diversity selection: Take top candidates, then run farthest-first on features
    candidate_pool_size = min(len(unlab_indices), batch_size * 4)
    top_candidate_order = np.argsort(score)[::-1][:candidate_pool_size]
    cand_unlab = [unlab_indices[i] for i in top_candidate_order]

    cand_feats = norm_feats[cand_unlab]
    labeled_feats = norm_feats[current_labeled]

    # Select batch_size samples from candidates maximizing diversity to labelled set
    sims = np.dot(cand_feats, labeled_feats.T)
    min_dist_to_labeled = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(sims, axis=1)))

    selected = []
    avail = list(range(len(cand_unlab)))
    # Pick first: highest score among most distant
    combined_cand_score = 0.7 * score[top_candidate_order] + 0.3 * min_dist_to_labeled
    first_idx = int(np.argmax(combined_cand_score))
    selected.append(cand_unlab[first_idx])
    avail.remove(first_idx)

    # Greedy farthest selection for remaining batch
    while len(selected) < batch_size and avail:
        sel_mat = norm_feats[selected]
        rem_feats = norm_feats[[cand_unlab[i] for i in avail]]
        sims_sel = np.dot(rem_feats, sel_mat.T)
        dist_to_batch = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(sims_sel, axis=1)))
        
        # Balance diversity with minority score
        rem_scores = score[[top_candidate_order[i] for i in avail]]
        step_score = 0.5 * rem_scores + 0.5 * dist_to_batch
        best_pos = int(np.argmax(step_score))
        best_cand_idx = avail.pop(best_pos)
        selected.append(cand_unlab[best_cand_idx])

    return selected

def run_experiment():
    repo_root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
    feat_dir = repo_root / "data" / "cached_features"
    out_dir = repo_root / "outputs" / "isic2019" / "research_investigation" / "advanced_al"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pre-extracted feature embeddings...")
    train_data = torch.load(feat_dir / "train_features.pt", map_location="cpu", weights_only=False)
    val_data = torch.load(feat_dir / "val_features.pt", map_location="cpu", weights_only=False)
    test_data = torch.load(feat_dir / "test_features.pt", map_location="cpu", weights_only=False)

    train_f, train_l = train_data["features"], train_data["labels"]
    val_f, val_l = val_data["features"], val_data["labels"]
    test_f, test_l = test_data["features"], test_data["labels"]

    # Unit L2 normalize
    norm_train_f = (train_f / torch.norm(train_f, dim=1, keepdim=True)).numpy()
    val_l_np = val_l.numpy()
    test_l_np = test_l.numpy()
    val_priors = np.bincount(val_l_np, minlength=NUM_CLASSES) / len(val_l_np)

    seeds = [42, 43, 44]
    init_budget = 80
    batch_size = 40
    budget_ceiling = 320
    rounds = (budget_ceiling - init_budget) // batch_size

    methods = [
        {"id": "A1_Random", "name": "Class-Weighted Random Baseline", "strategy": "random", "loss": "class_weighted"},
        {"id": "A9_ColdStart", "name": "Cold-Start + Class-Weighted Loss", "strategy": "cold_random", "loss": "class_weighted"},
        {"id": "A15_MAL", "name": "Minority-Aware Active Learning (Proposed)", "strategy": "mal", "loss": "class_weighted"},
        {"id": "A16_MAL_LogitAdj", "name": "MAL + Logit-Adjusted Loss", "strategy": "mal", "loss": "logit_adjusted"},
    ]

    all_results = {}
    budget_trajectories = {}

    for m in methods:
        print(f"\n=======================================================")
        print(f"  Running Strategy: [{m['id']}] {m['name']}")
        print(f"=======================================================")
        method_runs = []
        method_traj = []

        for seed in seeds:
            print(f"  --- Seed {seed} ---")
            rng = np.random.RandomState(seed)

            # Initialize seed-matched pool
            if m["strategy"] == "cold_random":
                lab = get_calibrated_cold_start(norm_train_f, n_init=init_budget, seed=seed)
            else:
                lab = list(rng.choice(len(train_f), size=init_budget, replace=False))

            unlab = [i for i in range(len(train_f)) if i not in lab]
            seed_traj = []

            # 7 interactive rounds
            for r in range(rounds + 1):
                curr_budget = len(lab)
                
                # Fit model
                model = train_linear_model(
                    train_f[lab],
                    train_l[lab],
                    epochs=40,
                    lr=0.01,
                    loss_type=m["loss"],
                    val_priors=val_priors
                )

                # Predict on Validation
                with torch.no_grad():
                    val_logits = model(val_f)
                    val_probs = torch.softmax(val_logits, dim=1).numpy()
                    
                    test_logits = model(test_f)
                    test_probs = torch.softmax(test_logits, dim=1).numpy()

                # Tune threshold tau on validation set
                best_tau = 0.0
                best_val_ba = balanced_accuracy_score(val_l_np, np.argmax(val_probs, axis=1)) * 100.0
                for tau_cand in np.linspace(0.05, 0.70, 20):
                    adj_p = val_probs / (val_priors ** tau_cand + 1e-6)
                    score = balanced_accuracy_score(val_l_np, np.argmax(adj_p, axis=1)) * 100.0
                    if score > best_val_ba:
                        best_val_ba = score
                        best_tau = tau_cand

                # Evaluate on Test set (uncalibrated & calibrated)
                uncal_test = evaluate_predictions(test_l_np, test_probs, tau=0.0)
                cal_test = evaluate_predictions(test_l_np, test_probs, tau=best_tau, priors=val_priors)

                # Class counts in acquired training pool
                cur_counts = np.bincount(train_l[lab].numpy(), minlength=NUM_CLASSES).tolist()

                traj_point = {
                    "round": r,
                    "budget": curr_budget,
                    "acquired_class_counts": {CLASSES[c]: cur_counts[c] for c in range(NUM_CLASSES)},
                    "uncalibrated_test_ba": uncal_test["balanced_accuracy"],
                    "uncalibrated_test_f1": uncal_test["macro_f1"],
                    "calibrated_test_ba": cal_test["balanced_accuracy"],
                    "calibrated_test_f1": cal_test["macro_f1"],
                    "calibrated_test_auroc": cal_test["macro_auroc"],
                    "calibrated_minority_recall": cal_test["minority_recall"],
                    "best_tau": best_tau
                }
                seed_traj.append(traj_point)
                print(f"    Round {r} (Budget {curr_budget}): Uncal BA: {uncal_test['balanced_accuracy']:.2f}% | Cal BA: {cal_test['balanced_accuracy']:.2f}% | Cal F1: {cal_test['macro_f1']:.4f} | MinRec: {cal_test['minority_recall']:.1f}%")

                if r == rounds:
                    break

                # Query next batch
                with torch.no_grad():
                    unlab_t = train_f[unlab]
                    unlab_probs = torch.softmax(model(unlab_t), dim=1).numpy()

                val_recs = recall_score(val_l_np, np.argmax(val_probs, axis=1), average=None, zero_division=0)

                if m["strategy"] == "random" or m["strategy"] == "cold_random":
                    queries = list(rng.choice(unlab, size=batch_size, replace=False))
                elif "mal" in m["strategy"]:
                    queries = minority_aware_acquisition(
                        unlab_indices=unlab,
                        norm_feats=norm_train_f,
                        probs_unlab=unlab_probs,
                        batch_size=batch_size,
                        val_recalls=val_recs,
                        current_labeled=lab
                    )

                lab.extend(queries)
                unlab = [i for i in unlab if i not in queries]

            method_runs.append({
                "seed": seed,
                "final_uncal": uncal_test,
                "final_cal": cal_test,
                "final_acquired_counts": cur_counts
            })
            method_traj.append(seed_traj)

        all_results[m["id"]] = {"desc": m["name"], "runs": method_runs}
        budget_trajectories[m["id"]] = method_traj

    # Save detailed runs and trajectories
    with open(out_dir / "advanced_al_detailed_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_dir / "budget_trajectories.json", "w") as f:
        json.dump(budget_trajectories, f, indent=2)

    # Compile Final Summary Table
    summary_rows = []
    for m_id, data in all_results.items():
        uncal_ba = [r["final_uncal"]["balanced_accuracy"] for r in data["runs"]]
        cal_ba = [r["final_cal"]["balanced_accuracy"] for r in data["runs"]]
        cal_f1 = [r["final_cal"]["macro_f1"] for r in data["runs"]]
        cal_auroc = [r["final_cal"]["macro_auroc"] for r in data["runs"]]
        cal_min_rec = [r["final_cal"]["minority_recall"] for r in data["runs"]]

        summary_rows.append({
            "Method": m_id,
            "Description": data["desc"],
            "Uncalibrated Bal-Acc (%)": f"{np.mean(uncal_ba):.2f} ± {np.std(uncal_ba):.2f}",
            "Calibrated Bal-Acc (%)": f"{np.mean(cal_ba):.2f} ± {np.std(cal_ba):.2f}",
            "Calibrated Macro-F1": f"{np.mean(cal_f1):.4f} ± {np.std(cal_f1):.4f}",
            "Calibrated AUROC": f"{np.mean(cal_auroc):.4f} ± {np.std(cal_auroc):.4f}",
            "Minority Recall (%)": f"{np.mean(cal_min_rec):.1f} ± {np.std(cal_min_rec):.1f}%"
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "advanced_al_summary.csv", index=False)
    print("\n================ FINAL INVESTIGATION SUMMARY TABLE ================\n")
    print(df_summary.to_string(index=False))

    return df_summary

if __name__ == "__main__":
    run_experiment()
