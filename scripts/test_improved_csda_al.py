"""Benchmark script to test and validate improved CSDA-AL vs Random across seeds 42, 43, 44."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8

def evaluate_model(model, X_eval, y_eval):
    model.eval()
    with torch.no_grad():
        logits = model(X_eval)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y_eval.cpu().numpy()
    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    # One-vs-rest macro AUROC
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
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class = [{"class": CLASS_NAMES[c], "recall": float(rec[c]), "f1": float(f1[c])} for c in range(NUM_CLASSES)]
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "macro_auroc": macro_auroc, "per_class": per_class}

def train_probe_safe(X_train, y_train, epochs=40, lr=0.01):
    model = nn.Linear(X_train.shape[1], NUM_CLASSES)
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    total = float(len(y_train))
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    present = counts > 0
    if present.sum() > 0:
        # Smoothed inverse square root with upper cap
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

def get_calibrated_cold_start(features: np.ndarray, n_init: int = 80, seed: int = 42):
    norm_feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
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

def run_experiment(method, train_f, train_l, val_f, val_l, test_f, test_l, seeds=[42, 43, 44], budget=320, init_budget=80, batch_size=40):
    norm_train_f = train_f / (torch.norm(train_f, dim=1, keepdim=True) + 1e-8)
    norm_feats_np = norm_train_f.numpy()
    
    results = []
    for s in seeds:
        rng = np.random.RandomState(s)
        if method in ["improved_csda_al", "cold_start_only"]:
            lab = get_calibrated_cold_start(norm_feats_np, n_init=init_budget, seed=s)
        else:
            lab = list(rng.choice(len(train_f), size=init_budget, replace=False))
        unlab = [i for i in range(len(train_f)) if i not in lab]

        num_rounds = (budget - init_budget) // batch_size + 1
        for r in range(num_rounds):
            model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01)
            if r == num_rounds - 1:
                break
            with torch.no_grad():
                probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()

            if method == "random":
                queries = list(rng.choice(unlab, size=batch_size, replace=False))
            elif method == "entropy":
                ent = -np.sum(probs_unlab * np.log(probs_unlab + 1e-12), axis=1)
                ranked = np.argsort(-ent)
                queries = [unlab[i] for i in ranked[:batch_size]]
            elif method == "improved_csda_al":
                # Soft-target deficit matching + diversity + entropy
                ent = -np.sum(probs_unlab * np.log(probs_unlab + 1e-12), axis=1)
                u_score = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)

                # Class sensitivity: soft expected deficit
                lab_counts = np.bincount(train_l[lab].numpy(), minlength=NUM_CLASSES)
                target = (len(lab) + batch_size) / NUM_CLASSES
                deficits = np.maximum(0, target - lab_counts)
                norm_deficits = deficits / (np.sum(deficits) + 1e-8)
                c_score = np.dot(probs_unlab, norm_deficits)
                c_score = (c_score - c_score.min()) / (c_score.max() - c_score.min() + 1e-8)

                # Diversity: distance to nearest labeled in feature space
                unlab_mat = norm_feats_np[unlab]
                lab_mat = norm_feats_np[lab]
                sims = np.dot(unlab_mat, lab_mat.T)
                min_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(sims, axis=1)))
                d_score = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min() + 1e-8)

                # Hybrid score: alpha=0.4, beta=0.3, gamma=0.3
                hybrid = 0.40 * u_score + 0.30 * d_score + 0.30 * c_score
                scores = np.copy(hybrid)
                queries = []
                for _ in range(batch_size):
                    best = int(np.argmax(scores))
                    queries.append(unlab[best])
                    scores[best] = -1e9
                    sim_to_best = np.dot(unlab_mat, unlab_mat[best])
                    near = sim_to_best > 0.85  # cosine similarity > 0.85
                    scores[near] *= 0.5
            else:
                queries = list(rng.choice(unlab, size=batch_size, replace=False))

            lab.extend(queries)
            unlab = [i for i in unlab if i not in queries]

        eval_res = evaluate_model(model, test_f, test_l)
        results.append(eval_res)

    mean_ba = np.mean([r["balanced_accuracy"] for r in results]) * 100
    std_ba = np.std([r["balanced_accuracy"] for r in results]) * 100
    mean_f1 = np.mean([r["macro_f1"] for r in results])
    std_f1 = np.std([r["macro_f1"] for r in results])
    mean_auc = np.mean([r["macro_auroc"] for r in results])
    std_auc = np.std([r["macro_auroc"] for r in results])

    return {
        "method": method,
        "mean_ba": mean_ba, "std_ba": std_ba,
        "mean_f1": mean_f1, "std_f1": std_f1,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "raw": results
    }

if __name__ == "__main__":
    feat_dir = Path("data/cached_features")
    train_cache = torch.load(feat_dir / "train_features.pt", map_location="cpu")
    val_cache = torch.load(feat_dir / "val_features.pt", map_location="cpu")
    test_cache = torch.load(feat_dir / "test_features.pt", map_location="cpu")

    train_f, train_l = train_cache["features"], train_cache["labels"].long()
    val_f, val_l = val_cache["features"], val_cache["labels"].long()
    test_f, test_l = test_cache["features"], test_cache["labels"].long()

    print("Running 3-Seed Comparison (Seeds: 42, 43, 44)...")
    res_rand = run_experiment("random", train_f, train_l, val_f, val_l, test_f, test_l)
    print(f"RANDOM:       Bal-Acc: {res_rand['mean_ba']:.2f} ± {res_rand['std_ba']:.2f}% | Macro-F1: {res_rand['mean_f1']:.4f} ± {res_rand['std_f1']:.4f} | AUROC: {res_rand['mean_auc']:.4f}")

    res_ent = run_experiment("entropy", train_f, train_l, val_f, val_l, test_f, test_l)
    print(f"ENTROPY:      Bal-Acc: {res_ent['mean_ba']:.2f} ± {res_ent['std_ba']:.2f}% | Macro-F1: {res_ent['mean_f1']:.4f} ± {res_ent['std_f1']:.4f} | AUROC: {res_ent['mean_auc']:.4f}")

    res_imp = run_experiment("improved_csda_al", train_f, train_l, val_f, val_l, test_f, test_l)
    print(f"IMPROVED CSDA: Bal-Acc: {res_imp['mean_ba']:.2f} ± {res_imp['std_ba']:.2f}% | Macro-F1: {res_imp['mean_f1']:.4f} ± {res_imp['std_f1']:.4f} | AUROC: {res_imp['mean_auc']:.4f}")
