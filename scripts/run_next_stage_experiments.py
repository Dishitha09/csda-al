"""Master runner for Next-Stage CSDA-AL Controlled Experiments A9 through A13.

Enforces strictly identical dataset splits, identical annotation budgets,
and seed-matched initial labeled pools across seeds 42, 43, and 44.
"""
import copy
import json
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
)
import torch
import torch.nn as nn
import torch.optim as optim

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8

def evaluate_model(model: nn.Module, X_eval: torch.Tensor, y_eval: torch.Tensor) -> Dict:
    model.eval()
    with torch.no_grad():
        logits = model(X_eval)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    y_true = y_eval.cpu().numpy()
    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    # One-vs-Rest AUROC
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

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    per_class = []
    for c in range(NUM_CLASSES):
        per_class.append({
            "class_name": CLASS_NAMES[c],
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "support": int(supp[c])
        })

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
        "per_class": per_class,
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }

def train_probe_safe(X_train: torch.Tensor, y_train: torch.Tensor, epochs=40, lr=0.01, class_weighted=True) -> nn.Module:
    model = nn.Linear(X_train.shape[1], NUM_CLASSES)
    if class_weighted:
        counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
        total = float(len(y_train))
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        present = counts > 0
        if present.sum() > 0:
            raw_w = total / (torch.sqrt(counts[present]) * np.sqrt(total) + 1.0)
            raw_w = torch.clamp(raw_w, min=0.1, max=15.0)
            weights[present] = raw_w / raw_w.sum() * present.sum()
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

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
        for _ in range(rem):
            best = int(np.argmax(min_dists))
            selected.append(best)
            sim_to_best = np.dot(norm_feats, norm_feats[best])
            dist_to_best = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim_to_best))
            min_dists = np.minimum(min_dists, dist_to_best)
            min_dists[best] = -1.0

    return selected[:n_init]

def select_queries(
    strategy: str,
    probs_unlab: np.ndarray,
    norm_feats: np.ndarray,
    unlab_indices: List[int],
    lab_indices: List[int],
    lab_labels: List[int],
    batch_size: int = 40,
    seed: int = 42,
    alpha: float = 0.40,
    beta: float = 0.30,
    gamma: float = 0.30
) -> List[int]:
    rng = np.random.RandomState(seed)
    eps = 1e-12

    if strategy == "random":
        return list(rng.choice(unlab_indices, size=batch_size, replace=False))

    # Uncertainty U(x): Normalized Predictive Entropy
    ent = -np.sum(probs_unlab * np.log(probs_unlab + eps), axis=1)
    u_score = (ent - ent.min()) / (ent.max() - ent.min() + eps)

    if strategy == "entropy":
        ranked = np.argsort(-u_score)
        return [unlab_indices[i] for i in ranked[:batch_size]]

    # Revised Unit-Sphere Cosine Diversity D(x)
    unlab_mat = norm_feats[unlab_indices]
    lab_mat = norm_feats[lab_indices]
    sims = np.dot(unlab_mat, lab_mat.T)
    min_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(sims, axis=1)))
    d_score = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min() + eps)

    # Class-Sensitivity C(x): Soft-Target Deficit Matching
    counts = np.bincount(lab_labels, minlength=NUM_CLASSES)
    target = (len(lab_labels) + batch_size) / NUM_CLASSES
    deficits = np.maximum(0, target - counts)
    norm_deficits = deficits / (np.sum(deficits) + eps)
    exp_deficit = np.dot(probs_unlab, norm_deficits)
    c_score = (exp_deficit - exp_deficit.min()) / (exp_deficit.max() - exp_deficit.min() + eps)

    # Calibrated Hybrid Score
    comb_score = alpha * u_score + beta * d_score + gamma * c_score

    # Greedy Selection with intra-batch redundancy repulsion
    chosen = []
    scores = np.copy(comb_score)
    for _ in range(batch_size):
        best = int(np.argmax(scores))
        chosen.append(unlab_indices[best])
        scores[best] = -1e9
        sim_to_best = np.dot(unlab_mat, unlab_mat[best])
        near = sim_to_best > 0.85
        scores[near] *= 0.5

    return chosen

def run_experiment_suite():
    repo_root = Path(__file__).resolve().parent.parent
    feat_dir = repo_root / "data" / "cached_features"
    out_dir = repo_root / "outputs" / "isic2019"
    exp_base_dir = out_dir / "experiments"
    exp_base_dir.mkdir(parents=True, exist_ok=True)

    print("Loading verified cached features...")
    train_cache = torch.load(feat_dir / "train_features.pt", map_location="cpu")
    val_cache = torch.load(feat_dir / "val_features.pt", map_location="cpu")
    test_cache = torch.load(feat_dir / "test_features.pt", map_location="cpu")

    train_f, train_l = train_cache["features"], train_cache["labels"].long()
    val_f, val_l = val_cache["features"], val_cache["labels"].long()
    test_f, test_l = test_cache["features"], test_cache["labels"].long()

    # Precompute L2-normalized representations
    norm_train_f = (train_f / (torch.norm(train_f, dim=1, keepdim=True) + 1e-8)).numpy()

    seeds = [42, 43, 44]
    budget = 320
    init_budget = 80
    batch_size = 40
    num_rounds = (budget - init_budget) // batch_size + 1

    # Precompute seed-matched initial labeled sets for reproducible fair comparisons
    random_inits = {
        s: list(np.random.RandomState(s).choice(len(train_f), size=init_budget, replace=False))
        for s in seeds
    }
    cold_inits = {
        s: get_calibrated_cold_start(norm_train_f, n_init=init_budget, seed=s)
        for s in seeds
    }

    # -------------------------------------------------------------
    # 1. RUN A9: Class-Weighted Training + Calibrated Cold-Start
    # -------------------------------------------------------------
    print("\n--- Running Experiment A9: Class-Weighted + Calibrated Cold-Start ---")
    a9_results = []
    for s in seeds:
        lab = list(cold_inits[s])
        unlab = [i for i in range(len(train_f)) if i not in lab]
        for r in range(num_rounds):
            model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=True)
            if r == num_rounds - 1:
                break
            with torch.no_grad():
                probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
            queries = select_queries("random", probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(), batch_size=batch_size, seed=s+r)
            lab.extend(queries)
            unlab = [i for i in unlab if i not in queries]

        val_ev = evaluate_model(model, val_f, val_l)
        test_ev = evaluate_model(model, test_f, test_l)
        a9_results.append({"seed": s, "val": val_ev, "test": test_ev, "final_labeled_count": len(lab)})

    # -------------------------------------------------------------
    # 2. RUN A10: Class-Weighted Training + Uncertainty Acquisition
    # -------------------------------------------------------------
    print("\n--- Running Experiment A10: Class-Weighted + Uncertainty Acquisition ---")
    a10_results = []
    for s in seeds:
        lab = list(random_inits[s])
        unlab = [i for i in range(len(train_f)) if i not in lab]
        for r in range(num_rounds):
            model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=True)
            if r == num_rounds - 1:
                break
            with torch.no_grad():
                probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
            queries = select_queries("entropy", probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(), batch_size=batch_size, seed=s+r)
            lab.extend(queries)
            unlab = [i for i in unlab if i not in queries]

        val_ev = evaluate_model(model, val_f, val_l)
        test_ev = evaluate_model(model, test_f, test_l)
        a10_results.append({"seed": s, "val": val_ev, "test": test_ev, "final_labeled_count": len(lab)})

    # -------------------------------------------------------------
    # 3. RUN A11: Class-Weighted + Uncertainty + Revised Diversity
    # -------------------------------------------------------------
    print("\n--- Running Experiment A11: Class-Weighted + Uncertainty + Revised Diversity ---")
    a11_results = []
    for s in seeds:
        lab = list(random_inits[s])
        unlab = [i for i in range(len(train_f)) if i not in lab]
        for r in range(num_rounds):
            model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=True)
            if r == num_rounds - 1:
                break
            with torch.no_grad():
                probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
            # 70% Uncertainty, 30% Revised Diversity
            queries = select_queries("hybrid", probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(),
                                     batch_size=batch_size, seed=s+r, alpha=0.70, beta=0.30, gamma=0.0)
            lab.extend(queries)
            unlab = [i for i in unlab if i not in queries]

        val_ev = evaluate_model(model, val_f, val_l)
        test_ev = evaluate_model(model, test_f, test_l)
        a11_results.append({"seed": s, "val": val_ev, "test": test_ev, "final_labeled_count": len(lab)})

    # -------------------------------------------------------------
    # 4. RUN A12: Validation-Tuned Acquisition Weights Grid Search
    # -------------------------------------------------------------
    print("\n--- Running Experiment A12: Tuning Acquisition Weights on Validation Set ---")
    grid_candidates = [
        {"name": "W1_Balanced", "alpha": 0.40, "beta": 0.30, "gamma": 0.30},
        {"name": "W2_Uncertainty_Heavy", "alpha": 0.60, "beta": 0.20, "gamma": 0.20},
        {"name": "W3_No_Diversity", "alpha": 0.70, "beta": 0.00, "gamma": 0.30},
        {"name": "W4_High_Class_Sensitivity", "alpha": 0.50, "beta": 0.10, "gamma": 0.40},
        {"name": "W5_Equal_Unc_Class", "alpha": 0.50, "beta": 0.00, "gamma": 0.50},
    ]

    a12_validation_scores = []
    best_weights = None
    best_val_score = -1.0

    for cand in grid_candidates:
        cand_val_bas = []
        cand_val_f1s = []
        for s in seeds:
            lab = list(cold_inits[s])
            unlab = [i for i in range(len(train_f)) if i not in lab]
            for r in range(num_rounds):
                model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=True)
                if r == num_rounds - 1:
                    break
                with torch.no_grad():
                    probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
                queries = select_queries("hybrid", probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(),
                                         batch_size=batch_size, seed=s+r,
                                         alpha=cand["alpha"], beta=cand["beta"], gamma=cand["gamma"])
                lab.extend(queries)
                unlab = [i for i in unlab if i not in queries]

            val_ev = evaluate_model(model, val_f, val_l)
            cand_val_bas.append(val_ev["balanced_accuracy"])
            cand_val_f1s.append(val_ev["macro_f1"])

        mean_val_ba = float(np.mean(cand_val_bas))
        mean_val_f1 = float(np.mean(cand_val_f1s))
        composite_val_score = 0.5 * mean_val_ba + 0.5 * mean_val_f1

        print(f"  Candidate {cand['name']:24s} | Val Bal-Acc: {mean_val_ba*100:.2f}% | Val Macro-F1: {mean_val_f1:.4f}")
        a12_validation_scores.append({
            "candidate": cand["name"],
            "weights": {"alpha": cand["alpha"], "beta": cand["beta"], "gamma": cand["gamma"]},
            "mean_val_bal_acc": mean_val_ba,
            "mean_val_macro_f1": mean_val_f1,
            "composite_val_score": composite_val_score
        })

        if composite_val_score > best_val_score:
            best_val_score = composite_val_score
            best_weights = cand

    print(f"\nBest Validation-Selected Weight Configuration: {best_weights['name']} ({best_weights})")

    # Evaluate best A12 configuration on locked test set
    a12_results = []
    for s in seeds:
        lab = list(cold_inits[s])
        unlab = [i for i in range(len(train_f)) if i not in lab]
        for r in range(num_rounds):
            model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=True)
            if r == num_rounds - 1:
                break
            with torch.no_grad():
                probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
            queries = select_queries("hybrid", probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(),
                                     batch_size=batch_size, seed=s+r,
                                     alpha=best_weights["alpha"], beta=best_weights["beta"], gamma=best_weights["gamma"])
            lab.extend(queries)
            unlab = [i for i in unlab if i not in queries]

        test_ev = evaluate_model(model, test_f, test_l)
        a12_results.append({"seed": s, "test": test_ev})

    # -------------------------------------------------------------
    # 5. RUN A13: Final Integrated CSDA-AL
    # -------------------------------------------------------------
    print("\n--- Running Experiment A13: Final Integrated CSDA-AL ---")
    # A13 integrates: Calibrated Cold-Start + Safe Class-Balanced Loss + Validation-Selected Hybrid Weights
    a13_results = []
    for s in seeds:
        lab = list(cold_inits[s])
        unlab = [i for i in range(len(train_f)) if i not in lab]
        for r in range(num_rounds):
            model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=True)
            if r == num_rounds - 1:
                break
            with torch.no_grad():
                probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
            queries = select_queries("hybrid", probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(),
                                     batch_size=batch_size, seed=s+r,
                                     alpha=best_weights["alpha"], beta=best_weights["beta"], gamma=best_weights["gamma"])
            lab.extend(queries)
            unlab = [i for i in unlab if i not in queries]

        test_ev = evaluate_model(model, test_f, test_l)
        a13_results.append({"seed": s, "test": test_ev, "acquired_labels": train_l[lab].tolist()})

    # Save all results to disk
    experiments = {
        "A9": {"desc": "Class-Weighted Training + Calibrated Cold-Start", "runs": a9_results},
        "A10": {"desc": "Class-Weighted Training + Uncertainty Acquisition", "runs": a10_results},
        "A11": {"desc": "Class-Weighted + Uncertainty + Revised Diversity", "runs": a11_results},
        "A12": {"desc": f"Validation-Tuned CSDA-AL ({best_weights['name']})", "validation_grid": a12_validation_scores, "runs": a12_results},
        "A13": {"desc": "Final Integrated CSDA-AL", "runs": a13_results}
    }

    for exp_id, data in experiments.items():
        exp_dir = exp_base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "results.json", "w") as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------
    # 6. COMPILE MASTER COMPARISON TABLE
    # -------------------------------------------------------------
    # Load A0, A1, A2, A3, A7, A8 from previous verified runs
    baseline_records = [
        {"id": "A0", "name": "Random Baseline (Unweighted)", "ba": (26.06, 0.27), "f1": (0.2606, 0.0048), "auc": (0.7436, 0.0218)},
        {"id": "A1", "name": "Class-Weighted Random", "ba": (28.11, 1.57), "f1": (0.2798, 0.0175), "auc": (0.7376, 0.0203)},
        {"id": "A2", "name": "Calibrated Cold-Start Only", "ba": (27.49, 0.56), "f1": (0.2769, 0.0069), "auc": (0.7503, 0.0075)},
        {"id": "A3", "name": "Uncertainty Only (Entropy)", "ba": (26.75, 1.06), "f1": (0.2707, 0.0117), "auc": (0.7542, 0.0211)},
        {"id": "A7", "name": "CSDA-AL without Diversity", "ba": (26.53, 0.65), "f1": (0.2623, 0.0095), "auc": (0.7603, 0.0150)},
        {"id": "A8", "name": "Original CSDA-AL Combined", "ba": (26.36, 1.34), "f1": (0.2675, 0.0238), "auc": (0.7594, 0.0128)},
    ]

    def summarize_runs(runs):
        bas = [r["test"]["balanced_accuracy"] * 100 for r in runs]
        f1s = [r["test"]["macro_f1"] for r in runs]
        aucs = [r["test"]["macro_auroc"] for r in runs]
        return (np.mean(bas), np.std(bas)), (np.mean(f1s), np.std(f1s)), (np.mean(aucs), np.std(aucs))

    new_records = [
        {"id": "A9", "name": "Class-Weighted + Calibrated Cold-Start", **dict(zip(["ba", "f1", "auc"], summarize_runs(a9_results)))},
        {"id": "A10", "name": "Class-Weighted + Uncertainty Acquisition", **dict(zip(["ba", "f1", "auc"], summarize_runs(a10_results)))},
        {"id": "A11", "name": "Class-Weighted + Uncertainty + Revised Diversity", **dict(zip(["ba", "f1", "auc"], summarize_runs(a11_results)))},
        {"id": "A12", "name": f"Validation-Tuned CSDA-AL ({best_weights['name']})", **dict(zip(["ba", "f1", "auc"], summarize_runs(a12_results)))},
        {"id": "A13", "name": "Final Improved CSDA-AL", **dict(zip(["ba", "f1", "auc"], summarize_runs(a13_results)))},
    ]

    all_records = baseline_records + new_records
    table_rows = []
    for r in all_records:
        table_rows.append({
            "Configuration": f"[{r['id']}] {r['name']}",
            "Balanced Accuracy": f"{r['ba'][0]:.2f} ± {r['ba'][1]:.2f}%",
            "Macro-F1": f"{r['f1'][0]:.4f} ± {r['f1'][1]:.4f}",
            "Macro-AUROC": f"{r['auc'][0]:.4f} ± {r['auc'][1]:.4f}"
        })

    df_master = pd.DataFrame(table_rows)
    master_csv = out_dir / "metrics" / "next_stage_benchmark_comparison.csv"
    df_master.to_csv(master_csv, index=False)

    def df_to_markdown(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join([str(val) for val in row.values]) + " |")
        return "\n".join(lines) + "\n"

    print("\n================ MASTER COMPARISON TABLE (A0 to A13) ================\n")
    print(df_to_markdown(df_master))

    return df_master, experiments

if __name__ == "__main__":
    run_experiment_suite()
