"""Controlled Head-to-Head Validation: A1 vs A11 on ISIC 2019.

Enforces strictly identical initial labeled IDs per seed, identical canonical splits,
and identical annotation budgets across seeds 42, 43, 44.
Saves detailed round-by-round acquired IDs, probability matrices, and per-class metrics.
"""
import copy
import json
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
    confusion_matrix
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
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))

    per_class = []
    for c in range(NUM_CLASSES):
        per_class.append({
            "class_name": CLASS_NAMES[c],
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "support": int(supp[c]),
            "auroc": float(aurocs[c])
        })

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }

def train_probe_safe(X_train: torch.Tensor, y_train: torch.Tensor, epochs=40, lr=0.01) -> nn.Module:
    model = nn.Linear(X_train.shape[1], NUM_CLASSES)
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    total = float(len(y_train))
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    present = counts > 0
    if present.sum() > 0:
        raw_w = total / (torch.sqrt(counts[present]) * np.sqrt(total) + 1.0)
        raw_w = torch.clamp(raw_w, min=0.1, max=15.0)
        weights[present] = raw_w / raw_w.sum() * present.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

def select_a11_queries(
    probs_unlab: np.ndarray,
    norm_feats: np.ndarray,
    unlab_indices: List[int],
    lab_indices: List[int],
    batch_size: int = 40,
    seed: int = 42,
    alpha: float = 0.70,
    beta: float = 0.30
) -> List[int]:
    eps = 1e-12
    # Uncertainty: Normalized Shannon Entropy
    ent = -np.sum(probs_unlab * np.log(probs_unlab + eps), axis=1)
    u_score = (ent - ent.min()) / (ent.max() - ent.min() + eps)

    # Revised Cosine Diversity: Distance on unit sphere
    unlab_mat = norm_feats[unlab_indices]
    lab_mat = norm_feats[lab_indices]
    sims = np.dot(unlab_mat, lab_mat.T)
    min_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(sims, axis=1)))
    d_score = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min() + eps)

    # Hybrid Score (A11: 70% U + 30% D)
    comb_score = alpha * u_score + beta * d_score

    # Greedy Selection with intra-batch repulsion (cosine similarity > 0.85)
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

def run_a14():
    repo_root = Path(__file__).resolve().parent.parent
    splits_dir = repo_root / "outputs" / "isic2019" / "splits"
    feat_dir = repo_root / "data" / "cached_features"
    out_dir = repo_root / "outputs" / "isic2019" / "experiments" / "A14"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      PHASE 2: A14 CONTROLLED COMPARISON (A1 vs A11)              ")
    print("==================================================================")

    # Verify Splits
    train_ids_df = pd.read_csv(splits_dir / "train_ids.csv")
    val_ids_df = pd.read_csv(splits_dir / "validation_ids.csv")
    test_ids_df = pd.read_csv(splits_dir / "test_ids.csv")
    train_ids = train_ids_df["image_id"].tolist()
    val_ids = val_ids_df["image_id"].tolist()
    test_ids = test_ids_df["image_id"].tolist()

    # Load representations
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

    # Pre-sample and FREEZE the EXACT same initial labeled IDs for each seed
    seed_matched_initial_ids = {}
    seed_matched_initial_indices = {}
    for s in seeds:
        rng = np.random.RandomState(s)
        init_idx = list(rng.choice(len(train_f), size=init_budget, replace=False))
        seed_matched_initial_indices[s] = init_idx
        seed_matched_initial_ids[s] = [train_ids[i] for i in init_idx]

    # Save initial labeled sets for audit verification
    with open(out_dir / "seed_matched_initial_labeled_ids.json", "w") as f:
        json.dump(seed_matched_initial_ids, f, indent=2)

    results = {"A1": [], "A11": []}
    acquisition_histories = {"A1": {}, "A11": {}}

    for method in ["A1", "A11"]:
        print(f"\n--- Running Method [{method}] across seeds {seeds} ---")
        acquisition_histories[method] = {}
        for s in seeds:
            rng = np.random.RandomState(s)
            # Use identical initial labeled indices!
            lab = list(seed_matched_initial_indices[s])
            unlab = [i for i in range(len(train_f)) if i not in lab]

            round_records = []
            for r in range(num_rounds):
                model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01)
                current_counts = np.bincount(train_l[lab].numpy(), minlength=NUM_CLASSES).tolist()

                if r == num_rounds - 1:
                    round_records.append({
                        "round": r,
                        "labeled_count": len(lab),
                        "class_distribution": current_counts
                    })
                    break

                with torch.no_grad():
                    probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()

                if method == "A1":
                    queries = list(rng.choice(unlab, size=batch_size, replace=False))
                elif method == "A11":
                    queries = select_a11_queries(probs_unlab, norm_train_f, unlab, lab, batch_size=batch_size, seed=s+r)

                round_records.append({
                    "round": r,
                    "labeled_count": len(lab),
                    "acquired_ids": [train_ids[q] for q in queries],
                    "class_distribution": current_counts
                })

                lab.extend(queries)
                unlab = [i for i in unlab if i not in queries]

            val_ev = evaluate_model(model, val_f, val_l)
            test_ev = evaluate_model(model, test_f, test_l)
            results[method].append({
                "seed": s,
                "val": val_ev,
                "test": test_ev,
                "final_labeled_ids": [train_ids[i] for i in lab],
                "final_class_counts": np.bincount(train_l[lab].numpy(), minlength=NUM_CLASSES).tolist()
            })
            acquisition_histories[method][s] = round_records
            print(f"  [{method}] Seed {s:2d} -> Bal-Acc: {test_ev['balanced_accuracy']*100:.2f}% | Macro-F1: {test_ev['macro_f1']:.4f} | AUROC: {test_ev['macro_auroc']:.4f}")

    # Save full run records
    with open(out_dir / "a14_detailed_runs.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "acquisition_histories.json", "w") as f:
        json.dump(acquisition_histories, f, indent=2)

    # Master Head-to-Head Table
    summary_data = []
    for m in ["A1", "A11"]:
        bas = [r["test"]["balanced_accuracy"] * 100 for r in results[m]]
        f1s = [r["test"]["macro_f1"] for r in results[m]]
        aucs = [r["test"]["macro_auroc"] for r in results[m]]
        summary_data.append({
            "Method": m,
            "Description": "Random + Class-Weighted" if m == "A1" else "Class-Weighted + Unc + Revised Div (A11)",
            "Test Bal-Acc": f"{np.mean(bas):.2f} ± {np.std(bas):.2f}%",
            "Test Macro-F1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "Test Macro-AUROC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
            "Seed_42_BA": bas[0],
            "Seed_43_BA": bas[1],
            "Seed_44_BA": bas[2],
            "Seed_42_AUROC": aucs[0],
            "Seed_43_AUROC": aucs[1],
            "Seed_44_AUROC": aucs[2],
            "Mean_Bal_Acc": np.mean(bas),
            "Std_Bal_Acc": np.std(bas),
            "Mean_F1": np.mean(f1s),
            "Std_F1": np.std(f1s),
            "Mean_AUROC": np.mean(aucs),
            "Std_AUROC": np.std(aucs),
        })

    df_a14 = pd.DataFrame(summary_data)
    df_a14.to_csv(out_dir / "a14_head_to_head_summary.csv", index=False)

    def df_to_markdown(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join([str(val) for val in row.values]) + " |")
        return "\n".join(lines) + "\n"

    print("\n================ HEAD-TO-HEAD SUMMARY (A1 vs A11) ================\n")
    print(df_to_markdown(df_a14[["Method", "Description", "Test Bal-Acc", "Test Macro-F1", "Test Macro-AUROC"]]))

    return results, df_a14

if __name__ == "__main__":
    run_a14()
