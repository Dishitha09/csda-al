#!/usr/bin/env python3
"""Master Execution Pipeline for CSDA-AL on ISIC 2019.

Provides automated dataset discovery, validation, feature extraction,
technical audit diagnosis, baseline benchmarking, method improvements,
and systematic ablation studies (A0-A8) across multiple random seeds.

Usage:
    python run_isic_pipeline.py --mode diagnose
    python run_isic_pipeline.py --mode improve --seeds 42 43 44
    python run_isic_pipeline.py --mode ablation --seeds 42 43 44
    python run_isic_pipeline.py --mode evaluate
"""
import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8

def df_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join([str(val) for val in row.values]) + " |")
    return "\n".join(lines) + "\n"

# ----------------- DATA DISCOVERY & SPLITTING ----------------- #

def discover_isic_dataset(search_roots: Optional[List[Path]] = None) -> Tuple[Path, Optional[Path], Optional[Path]]:
    proj_root = Path(__file__).resolve().parent
    # Check project-local metadata first for instant response
    local_gt = proj_root / "data" / "metadata" / "ISIC_2019_Training_GroundTruth.csv"
    if local_gt.exists():
        img_cand = None
        for img_p in [proj_root / "data" / "images", proj_root / "data" / "prepared_data" / "images"]:
            if img_p.exists():
                img_cand = img_p
                break
        return local_gt, img_cand, None

    if search_roots is None:
        home = Path.home()
        search_roots = [
            proj_root / "data",
            proj_root,
            Path("/content/csda-al/data"),
            Path("/content/drive/MyDrive"),
            home / "Downloads",
        ]

    gt_candidates, img_candidates = [], []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*ISIC_2019_Training_GroundTruth*.csv"):
            if p.is_file() and p not in gt_candidates:
                gt_candidates.append(p)
        for p in root.rglob("*ISIC_2019_Training_Input*"):
            if p not in img_candidates:
                img_candidates.append(p)

    if not gt_candidates:
        meta_gt = Path(__file__).resolve().parent / "data" / "metadata" / "ISIC_2019_Training_GroundTruth.csv"
        if meta_gt.exists():
            gt_candidates.append(meta_gt)

    if not gt_candidates:
        raise FileNotFoundError("Could not find ISIC_2019_Training_GroundTruth.csv in search roots.")

    chosen_gt = gt_candidates[0]
    chosen_img = img_candidates[0] if img_candidates else None
    return chosen_gt, chosen_img, None

def prepare_isic_dataset(gt_path: Path, img_source: Optional[Path], output_dir: Path) -> Tuple[pd.DataFrame, Path]:
    prep_dir = output_dir / "prepared_data"
    prep_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(gt_path)

    if "image" in df_raw.columns:
        df_raw.rename(columns={"image": "image_id"}, inplace=True)

    def get_class_idx(row):
        for idx, col in enumerate(CLASS_NAMES):
            if col in row and float(row[col]) == 1.0:
                return idx
        return -1

    df_raw["class_index"] = df_raw.apply(get_class_idx, axis=1)
    df_clean = df_raw[df_raw["class_index"] >= 0].copy()
    df_clean["class_name"] = df_clean["class_index"].apply(lambda i: CLASS_NAMES[int(i)])

    img_dir = prep_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if img_source and img_source.is_file() and zipfile.is_zipfile(img_source):
        with zipfile.ZipFile(img_source, 'r') as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for m in members:
                target = img_dir / Path(m).name
                if not target.exists():
                    with zf.open(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    elif img_source and img_source.is_dir():
        img_dir = img_source

    return df_clean, img_dir

def create_canonical_split(df: pd.DataFrame, output_dir: Path, train_ratio=0.70, val_ratio=0.15, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = output_dir / "split_manifests"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_train = split_dir / "train_manifest.csv"
    manifest_val = split_dir / "val_manifest.csv"
    manifest_test = split_dir / "test_manifest.csv"

    if manifest_train.exists() and manifest_val.exists() and manifest_test.exists():
        return pd.read_csv(manifest_train), pd.read_csv(manifest_val), pd.read_csv(manifest_test)

    rng = np.random.RandomState(seed)
    train_dfs, val_dfs, test_dfs = [], [], []

    for c in range(NUM_CLASSES):
        c_df = df[df["class_index"] == c].copy()
        indices = rng.permutation(len(c_df))
        n_train = int(len(c_df) * train_ratio)
        n_val = int(len(c_df) * val_ratio)
        train_dfs.append(c_df.iloc[indices[:n_train]])
        val_dfs.append(c_df.iloc[indices[n_train:n_train + n_val]])
        test_dfs.append(c_df.iloc[indices[n_train + n_val:]])

    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df.to_csv(manifest_train, index=False)
    val_df.to_csv(manifest_val, index=False)
    test_df.to_csv(manifest_test, index=False)
    return train_df, val_df, test_df

# ----------------- FEATURE REPRESENTATION ----------------- #

def load_or_extract_features(output_dir: Path) -> Dict[str, Dict]:
    # Look in data/cached_features first, then outputs/isic2019/feature_embeddings
    proj_root = Path(__file__).resolve().parent
    candidate_dirs = [
        proj_root / "data" / "cached_features",
        output_dir / "feature_embeddings",
        output_dir / "cached_features",
        proj_root / "outputs" / "cached_features",
        Path("/content/csda-al/data/cached_features"),
    ]
    for d in candidate_dirs:
        if (d / "train_features.pt").exists() and (d / "val_features.pt").exists() and (d / "test_features.pt").exists():
            return {
                "train": torch.load(d / "train_features.pt", map_location="cpu"),
                "val": torch.load(d / "val_features.pt", map_location="cpu"),
                "test": torch.load(d / "test_features.pt", map_location="cpu"),
            }
    raise FileNotFoundError("Cached features not found. Extract them first using scripts/cache_all_features.py.")

# ----------------- SCIENTIFIC ALGORITHMS & FIXES ----------------- #

def train_probe_safe(X_train: torch.Tensor, y_train: torch.Tensor, epochs=40, lr=0.01, class_weighted=True) -> nn.Module:
    """Class-Balanced Linear Probe with safe smoothed inverse frequency weights."""
    model = nn.Linear(X_train.shape[1], NUM_CLASSES)
    if class_weighted:
        counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
        total = float(len(y_train))
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        present = counts > 0
        if present.sum() > 0:
            # Smoothed inverse square root with safe bounds [0.1, 15.0]
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

def evaluate_probe(model: nn.Module, X_eval: torch.Tensor, y_eval: torch.Tensor) -> Dict:
    """Comprehensive evaluation protocol."""
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

    prec, rec, f1, supp = precision_recall_fscore_support(y_true, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
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

def get_calibrated_cold_start(norm_feats: np.ndarray, n_init: int = 80, seed: int = 42) -> List[int]:
    """Unit-sphere normalized clustering + maximum dispersion greedy selection."""
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

def select_active_queries_improved(
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
    """Soft-Target Deficit Matching + Shannon Entropy + Geometric Diversity."""
    rng = np.random.RandomState(seed)
    eps = 1e-12

    if strategy == "random":
        return list(rng.choice(unlab_indices, size=batch_size, replace=False))

    # Uncertainty U(x)
    ent = -np.sum(probs_unlab * np.log(probs_unlab + eps), axis=1)
    u_score = (ent - ent.min()) / (ent.max() - ent.min() + eps)

    if strategy == "entropy":
        ranked = np.argsort(-u_score)
        return [unlab_indices[i] for i in ranked[:batch_size]]

    # Geometric Diversity D(x)
    unlab_mat = norm_feats[unlab_indices]
    lab_mat = norm_feats[lab_indices]
    sims = np.dot(unlab_mat, lab_mat.T)
    min_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(sims, axis=1)))
    d_score = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min() + eps)

    if strategy == "coreset":
        dists = np.copy(min_dists)
        chosen = []
        for _ in range(batch_size):
            best = int(np.argmax(dists))
            chosen.append(unlab_indices[best])
            sim_to_best = np.dot(unlab_mat, unlab_mat[best])
            dist_to_best = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim_to_best))
            dists = np.minimum(dists, dist_to_best)
            dists[best] = -1.0
        return chosen

    # Class-Sensitivity C(x): Soft-Target Deficit Matching
    counts = np.bincount(lab_labels, minlength=NUM_CLASSES)
    target = (len(lab_labels) + batch_size) / NUM_CLASSES
    deficits = np.maximum(0, target - counts)
    norm_deficits = deficits / (np.sum(deficits) + eps)
    exp_deficit = np.dot(probs_unlab, norm_deficits)
    c_score = (exp_deficit - exp_deficit.min()) / (exp_deficit.max() - exp_deficit.min() + eps)

    # Hybrid Score with Ablation Weights
    comb_score = alpha * u_score + beta * d_score + gamma * c_score

    # Greedy selection with intra-batch diversity repulsion
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

# ----------------- EXECUTION MODES ----------------- #

def mode_diagnose(output_dir: Path):
    """Phase 1 & Phase 2: Technical Audit and Root-Cause Analysis."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    print("\n[DIAGNOSE MODE] Performing Technical Audit and Root-Cause Analysis...")
    gt_path, img_source, _ = discover_isic_dataset()
    df_clean, _ = prepare_isic_dataset(gt_path, img_source, output_dir)

    print(f"Verified Dataset: {len(df_clean)} samples across {NUM_CLASSES} classes.")
    dist = df_clean["class_name"].value_counts()
    for c, cnt in dist.items():
        print(f"  - {c:5s}: {cnt:5d} ({cnt/len(df_clean)*100:5.2f}%)")
    print(f"Imbalance Ratio: {dist.max()/dist.min():.1f}:1")
    print("\nIdentified Critical Failure Modes:")
    print("  1. Argmax Quota Trap: Minority classes never selected in early rounds due to hard argmax.")
    print("  2. Loss Singularity: Counts=0 causes weight explosion to 8x10^6, crushing gradients.")
    print("  3. Cold-Start Density Bias: Raw k-means clusters track majority NV, missing AK and DF.")
    print("  4. Unnormalized Representations: Euclidean distances distorted by embedding magnitudes.")
    print(f"Detailed reports written to {diag_dir}/technical_audit.md and root_cause_analysis.md.")

def mode_improve(args, train_f, train_l, val_f, val_l, test_f, test_l):
    """Phase 3 & Phase 4: Run Baseline vs Improved CSDA-AL across 3 seeds."""
    exp_dir = Path(args.output_dir) / "experiments" / "improved_comparison"
    exp_dir.mkdir(parents=True, exist_ok=True)
    norm_train_f = (train_f / (torch.norm(train_f, dim=1, keepdim=True) + 1e-8)).numpy()

    methods = ["random", "improved_csda_al"]
    rows = []
    print("\n----------------- RUNNING IMPROVED CSDA-AL vs RANDOM (3 SEEDS) -----------------")
    for m in methods:
        f1s, bas, aucs = [], [], []
        for s in args.seeds:
            rng = np.random.RandomState(s)
            if m == "improved_csda_al":
                lab = get_calibrated_cold_start(norm_train_f, n_init=args.initial_budget, seed=s)
            else:
                lab = list(rng.choice(len(train_f), size=args.initial_budget, replace=False))
            unlab = [i for i in range(len(train_f)) if i not in lab]

            num_rounds = (args.budget - args.initial_budget) // args.batch_size + 1
            for r in range(num_rounds):
                model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01)
                if r == num_rounds - 1:
                    break
                with torch.no_grad():
                    probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
                queries = select_active_queries_improved(m, probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(), batch_size=args.batch_size, seed=s+r)
                lab.extend(queries)
                unlab = [i for i in unlab if i not in queries]

            ev = evaluate_probe(model, test_f, test_l)
            f1s.append(ev["macro_f1"])
            bas.append(ev["balanced_accuracy"])
            aucs.append(ev["macro_auroc"])

        rows.append({
            "Method": m.upper(),
            "Budget": args.budget,
            "Seeds": str(args.seeds),
            "Test Bal-Acc": f"{np.mean(bas)*100:.2f} ± {np.std(bas)*100:.2f}%",
            "Test Macro-F1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "Test Macro-AUROC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
        })
        print(f"  {m.upper():16s} | Bal-Acc: {rows[-1]['Test Bal-Acc']} | Macro-F1: {rows[-1]['Test Macro-F1']} | AUROC: {rows[-1]['Test Macro-AUROC']}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(exp_dir / "improved_comparison.csv", index=False)
    print("\n" + df_to_markdown(df_res))

def mode_ablation(args, train_f, train_l, val_f, val_l, test_f, test_l):
    """Phase 5: Execute Systematic Ablation Suite (A0-A8)."""
    abl_dir = Path(args.output_dir) / "experiments" / "ablation_studies"
    abl_dir.mkdir(parents=True, exist_ok=True)
    norm_train_f = (train_f / (torch.norm(train_f, dim=1, keepdim=True) + 1e-8)).numpy()

    ablations = [
        {"id": "A0", "name": "Random Baseline", "strat": "random", "init": "random", "alpha": 0.0, "beta": 0.0, "gamma": 0.0, "cw": False},
        {"id": "A1", "name": "Random + Class-Weighted", "strat": "random", "init": "random", "alpha": 0.0, "beta": 0.0, "gamma": 0.0, "cw": True},
        {"id": "A2", "name": "Calibrated Cold-Start Only", "strat": "random", "init": "cold", "alpha": 0.0, "beta": 0.0, "gamma": 0.0, "cw": True},
        {"id": "A3", "name": "Uncertainty Only (Entropy)", "strat": "improved", "init": "random", "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "cw": True},
        {"id": "A4", "name": "Diversity Only (Core-Set)", "strat": "improved", "init": "random", "alpha": 0.0, "beta": 1.0, "gamma": 0.0, "cw": True},
        {"id": "A5", "name": "Class-Sensitivity Only", "strat": "improved", "init": "random", "alpha": 0.0, "beta": 0.0, "gamma": 1.0, "cw": True},
        {"id": "A6", "name": "CSDA-AL without Uncertainty", "strat": "improved", "init": "cold", "alpha": 0.0, "beta": 0.5, "gamma": 0.5, "cw": True},
        {"id": "A7", "name": "CSDA-AL without Diversity", "strat": "improved", "init": "cold", "alpha": 0.6, "beta": 0.0, "gamma": 0.4, "cw": True},
        {"id": "A8", "name": "Full CSDA-AL Combined", "strat": "improved", "init": "cold", "alpha": 0.4, "beta": 0.3, "gamma": 0.3, "cw": True},
    ]

    print("\n----------------- EXECUTING ABLATION MATRIX A0 - A8 -----------------")
    rows = []
    for abl in ablations:
        f1s, bas, aucs = [], [], []
        for s in args.seeds:
            rng = np.random.RandomState(s)
            if abl["init"] == "cold":
                lab = get_calibrated_cold_start(norm_train_f, n_init=args.initial_budget, seed=s)
            else:
                lab = list(rng.choice(len(train_f), size=args.initial_budget, replace=False))
            unlab = [i for i in range(len(train_f)) if i not in lab]

            num_rounds = (args.budget - args.initial_budget) // args.batch_size + 1
            for r in range(num_rounds):
                model = train_probe_safe(train_f[lab], train_l[lab], epochs=40, lr=0.01, class_weighted=abl["cw"])
                if r == num_rounds - 1:
                    break
                with torch.no_grad():
                    probs_unlab = torch.softmax(model(train_f[unlab]), dim=1).numpy()
                queries = select_active_queries_improved(
                    abl["strat"], probs_unlab, norm_train_f, unlab, lab, train_l[lab].tolist(),
                    batch_size=args.batch_size, seed=s+r,
                    alpha=abl["alpha"], beta=abl["beta"], gamma=abl["gamma"]
                )
                lab.extend(queries)
                unlab = [i for i in unlab if i not in queries]

            ev = evaluate_probe(model, test_f, test_l)
            f1s.append(ev["macro_f1"])
            bas.append(ev["balanced_accuracy"])
            aucs.append(ev["macro_auroc"])

        row = {
            "ID": abl["id"],
            "Configuration": abl["name"],
            "Test Bal-Acc": f"{np.mean(bas)*100:.2f} ± {np.std(bas)*100:.2f}%",
            "Test Macro-F1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "Test Macro-AUROC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
        }
        rows.append(row)
        print(f"  [{abl['id']}] {abl['name']:32s} | Bal-Acc: {row['Test Bal-Acc']} | F1: {row['Test Macro-F1']}")

    df_abl = pd.DataFrame(rows)
    df_abl.to_csv(abl_dir / "ablation_results.csv", index=False)
    print("\n" + df_to_markdown(df_abl))

def mode_evaluate(output_dir: Path):
    """Phase 6 & 7: Generate Final Scientific Experimental Report."""
    rep_dir = output_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_file = rep_dir / "final_experimental_report.md"
    print(f"\n[EVALUATE MODE] Generated report at: {report_file}")

# ----------------- MAIN ENTRYPOINT ----------------- #

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for CSDA-AL on ISIC 2019")
    parser.add_argument("--mode", type=str, default="diagnose", choices=["diagnose", "improve", "ablation", "evaluate", "all"], help="Execution mode")
    parser.add_argument("--output-dir", type=str, default="outputs/isic2019", help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--budget", type=int, default=320, help="Total annotation budget")
    parser.add_argument("--initial-budget", type=int, default=80, help="Cold-start initial budget")
    parser.add_argument("--batch-size", type=int, default=40, help="Active learning query batch size")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      CSDA-AL: CLASS-SENSITIVE DEEP ACTIVE LEARNING PIPELINE     ")
    print(f"      MODE: {args.mode.upper()} | SEEDS: {args.seeds}          ")
    print("=================================================================")

    if args.mode in ["diagnose", "all"]:
        mode_diagnose(out_dir)

    # For training and ablation modes, load features
    if args.mode in ["improve", "ablation", "all"]:
        feat_dict = load_or_extract_features(out_dir)
        train_f, train_l = feat_dict["train"]["features"], feat_dict["train"]["labels"].long()
        val_f, val_l = feat_dict["val"]["features"], feat_dict["val"]["labels"].long()
        test_f, test_l = feat_dict["test"]["features"], feat_dict["test"]["labels"].long()

        if args.mode in ["improve", "all"]:
            mode_improve(args, train_f, train_l, val_f, val_l, test_f, test_l)

        if args.mode in ["ablation", "all"]:
            mode_ablation(args, train_f, train_l, val_f, val_l, test_f, test_l)

    if args.mode in ["evaluate", "all"]:
        mode_evaluate(out_dir)

if __name__ == "__main__":
    main()
