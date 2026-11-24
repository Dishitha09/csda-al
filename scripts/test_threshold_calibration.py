import numpy as np
import json
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from pathlib import Path

repo_root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")

with open(repo_root / "outputs" / "isic2019" / "experiments" / "A14" / "a14_detailed_runs.json") as f:
    d = json.load(f)

test_feat = torch.load(repo_root / "data" / "cached_features" / "test_features.pt", map_location="cpu", weights_only=False)
y_true = np.array(test_feat["labels"])

val_feat = torch.load(repo_root / "data" / "cached_features" / "val_features.pt", map_location="cpu", weights_only=False)
y_val_true = np.array(val_feat["labels"])

classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

print("=====================================================================")
print("  PHASE 7 TEST: VALIDATION-TUNED THRESHOLD CALIBRATION & LOGIT PRIOR")
print("=====================================================================")

val_priors = np.bincount(y_val_true, minlength=8) / len(y_val_true)

for m in ["A1", "A11"]:
    bal_accs = []
    f1s = []
    recs = []
    print(f"\n--- Model {m} ---")
    for run in d[m]:
        seed = run["seed"]
        val_probs = np.array(run["val"]["probabilities"])
        test_probs = np.array(run["test"]["probabilities"])
        
        # Uncalibrated baseline
        uncal_test_preds = np.argmax(test_probs, axis=1)
        uncal_ba = balanced_accuracy_score(y_true, uncal_test_preds) * 100
        uncal_f1 = f1_score(y_true, uncal_test_preds, average="macro")

        # Tune tau strictly on VALIDATION set
        best_tau = 0.0
        best_val_ba = balanced_accuracy_score(y_val_true, np.argmax(val_probs, axis=1))
        for tau in np.linspace(0.05, 0.70, 25):
            adj_val_probs = val_probs / (val_priors ** tau + 1e-6)
            adj_val_preds = np.argmax(adj_val_probs, axis=1)
            score = balanced_accuracy_score(y_val_true, adj_val_preds)
            if score > best_val_ba:
                best_val_ba = score
                best_tau = tau
        
        # Apply optimal tau to LOCKED TEST SET
        adj_test_probs = test_probs / (val_priors ** best_tau + 1e-6)
        adj_test_preds = np.argmax(adj_test_probs, axis=1)
        
        cal_ba = balanced_accuracy_score(y_true, adj_test_preds) * 100
        cal_f1 = f1_score(y_true, adj_test_preds, average="macro")
        r_all = recall_score(y_true, adj_test_preds, average=None) * 100
        
        bal_accs.append(cal_ba)
        f1s.append(cal_f1)
        recs.append(r_all)
        print(f"Seed {seed}:")
        print(f"  Uncalibrated: Bal-Acc = {uncal_ba:.2f}%, Macro-F1 = {uncal_f1:.4f}")
        print(f"  Calibrated (tau={best_tau:.2f}): Bal-Acc = {cal_ba:.2f}%, Macro-F1 = {cal_f1:.4f} (Delta BA: {cal_ba - uncal_ba:+.2f}%)")
        print(f"  Minority Recalls: DF={r_all[5]:.1f}%, VASC={r_all[6]:.1f}%, SCC={r_all[7]:.1f}%, AK={r_all[3]:.1f}%")

    print(f"\n{m} SUMMARY (Calibrated across 3 seeds):")
    print(f"  Balanced Accuracy: {np.mean(bal_accs):.2f} ± {np.std(bal_accs):.2f}%")
    print(f"  Macro-F1:          {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    mean_recs = np.mean(recs, axis=0)
    for c_i, c_n in enumerate(classes):
        print(f"    {c_n}: {mean_recs[c_i]:.1f}% recall")
