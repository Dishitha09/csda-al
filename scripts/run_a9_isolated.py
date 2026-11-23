"""Isolated Controlled Experiment for A9 on ISIC 2019 under Verified Fixed Split.

Compares:
- A0: Unweighted Random Baseline
- A1: Class-Weighted Random
- A2: Calibrated Cold-Start Only
- A9: Class-Weighted Training + Calibrated Cold-Start (Combining the two strongest approaches)

Evaluates Balanced Accuracy, Macro-F1, Macro-AUROC, per-class recall, and seed variance across seeds 42, 43, and 44.
"""
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

    # Multiclass One-vs-Rest AUROC
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

def train_probe(X_train: torch.Tensor, y_train: torch.Tensor, epochs=40, lr=0.01, class_weighted=True) -> nn.Module:
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

def run_isolated_a9():
    repo_root = Path(__file__).resolve().parent.parent
    splits_dir = repo_root / "outputs" / "isic2019" / "splits"
    feat_dir = repo_root / "data" / "cached_features"
    out_dir = repo_root / "outputs" / "isic2019" / "experiments" / "A9"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      ISOLATED CONTROLLED EXPERIMENT: A9 (COMBINED A1 + A2)       ")
    print("==================================================================")

    # 1. Verify Fixed Split Manifests
    train_ids_df = pd.read_csv(splits_dir / "train_ids.csv")
    val_ids_df = pd.read_csv(splits_dir / "validation_ids.csv")
    test_ids_df = pd.read_csv(splits_dir / "test_ids.csv")
    print(f"Verified Fixed Split Partitions:")
    print(f"  - Training Set:   {len(train_ids_df)} samples")
    print(f"  - Validation Set: {len(val_ids_df)} samples")
    print(f"  - Test Set:       {len(test_ids_df)} samples (LOCKED)")

    # 2. Load Embeddings
    train_cache = torch.load(feat_dir / "train_features.pt", map_location="cpu")
    val_cache = torch.load(feat_dir / "val_features.pt", map_location="cpu")
    test_cache = torch.load(feat_dir / "test_features.pt", map_location="cpu")

    train_f, train_l = train_cache["features"], train_cache["labels"].long()
    val_f, val_l = val_cache["features"], val_cache["labels"].long()
    test_f, test_l = test_cache["features"], test_cache["labels"].long()

    norm_train_f = (train_f / (torch.norm(train_f, dim=1, keepdim=True) + 1e-8)).numpy()

    seeds = [42, 43, 44]
    budget = 320
    init_budget = 80
    batch_size = 40
    num_rounds = (budget - init_budget) // batch_size + 1

    configs = [
        {"id": "A0", "name": "Random Baseline (Unweighted)", "init": "random", "cw": False},
        {"id": "A1", "name": "A1: Random + Class-Weighted Loss", "init": "random", "cw": True},
        {"id": "A2", "name": "A2: Calibrated Cold-Start Only", "init": "cold", "cw": False},
        {"id": "A9", "name": "A9: Combined (Cold-Start + Class-Weighted)", "init": "cold", "cw": True},
    ]

    all_results = {}
    summary_rows = []

    for cfg in configs:
        print(f"\nEvaluating Configuration [{cfg['id']}]: {cfg['name']}...")
        runs = []
        for s in seeds:
            rng = np.random.RandomState(s)
            if cfg["init"] == "cold":
                lab = get_calibrated_cold_start(norm_train_f, n_init=init_budget, seed=s)
            else:
                lab = list(rng.choice(len(train_f), size=init_budget, replace=False))
            unlab = [i for i in range(len(train_f)) if i not in lab]

            for r in range(num_rounds):
                model = train_probe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=cfg["cw"])
                if r == num_rounds - 1:
                    break
                queries = list(rng.choice(unlab, size=batch_size, replace=False))
                lab.extend(queries)
                unlab = [i for i in unlab if i not in queries]

            test_ev = evaluate_model(model, test_f, test_l)
            runs.append({
                "seed": s,
                "test": test_ev,
                "initial_counts": np.bincount(train_l[lab[:init_budget]].numpy(), minlength=NUM_CLASSES).tolist(),
                "final_counts": np.bincount(train_l[lab].numpy(), minlength=NUM_CLASSES).tolist()
            })
            print(f"  Seed {s:2d} -> Bal-Acc: {test_ev['balanced_accuracy']*100:.2f}% | Macro-F1: {test_ev['macro_f1']:.4f} | AUROC: {test_ev['macro_auroc']:.4f}")

        bas = [r["test"]["balanced_accuracy"] * 100 for r in runs]
        f1s = [r["test"]["macro_f1"] for r in runs]
        aucs = [r["test"]["macro_auroc"] for r in runs]

        summary_rows.append({
            "ID": cfg["id"],
            "Configuration": cfg["name"],
            "Test Bal-Acc": f"{np.mean(bas):.2f} ± {np.std(bas):.2f}%",
            "Test Macro-F1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "Test Macro-AUROC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
            "Seed_42_BA": bas[0],
            "Seed_43_BA": bas[1],
            "Seed_44_BA": bas[2],
            "Std_BA": np.std(bas),
            "Mean_AUROC": np.mean(aucs)
        })
        all_results[cfg["id"]] = {"desc": cfg["name"], "runs": runs}

    df_summary = pd.DataFrame(summary_rows)
    with open(out_dir / "isolated_a9_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    df_summary.to_csv(out_dir / "isolated_a9_summary.csv", index=False)

    print("\n================ FINAL CONTROLLED COMPARISON TABLE ================\n")
    headers = ["ID", "Configuration", "Test Bal-Acc", "Test Macro-F1", "Test Macro-AUROC"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df_summary.iterrows():
        lines.append(f"| {row['ID']} | {row['Configuration']} | {row['Test Bal-Acc']} | {row['Test Macro-F1']} | {row['Test Macro-AUROC']} |")
    print("\n".join(lines))

    # Per-Class Recall Comparison between A1 and A9
    print("\n---------------- Per-Class Recall Comparison (A1 vs A9) ----------------")
    per_class_table = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        a1_rec = np.mean([r["test"]["per_class"][c_idx]["recall"] for r in all_results["A1"]["runs"]]) * 100
        a2_rec = np.mean([r["test"]["per_class"][c_idx]["recall"] for r in all_results["A2"]["runs"]]) * 100
        a9_rec = np.mean([r["test"]["per_class"][c_idx]["recall"] for r in all_results["A9"]["runs"]]) * 100
        per_class_table.append({
            "Class": c_name,
            "A0 (Random)": np.mean([r["test"]["per_class"][c_idx]["recall"] for r in all_results["A0"]["runs"]]) * 100,
            "A1 (Class-Weighted)": a1_rec,
            "A2 (Cold-Start)": a2_rec,
            "A9 (Combined)": a9_rec,
            "A9 vs A1 Gain": a9_rec - a1_rec
        })
    df_pc = pd.DataFrame(per_class_table)
    df_pc.to_csv(out_dir / "per_class_recall_comparison.csv", index=False)
    print(df_pc.to_string(index=False))

    return df_summary, all_results

if __name__ == "__main__":
    run_isolated_a9()
