"""Phase 3 Diagnostic Investigation: Why Minority Classes (DF, VASC, SCC, AK) Fail.

Examines:
1. Confusion matrices for A1, A9, A11
2. Per-class Precision, Recall, F1
3. Predicted class distribution vs True class distribution
4. Acquired sample counts per class across AL rounds
5. Confidence distributions per class
6. False-positive and false-negative breakdown for minority classes
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8

def run_diagnostics():
    repo_root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
    out_dir = repo_root / "outputs" / "isic2019" / "research_investigation" / "phase3_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load A14 runs (A1 vs A11)
    a14_path = repo_root / "outputs" / "isic2019" / "experiments" / "A14" / "a14_detailed_runs.json"
    with open(a14_path, "r") as f:
        a14_data = json.load(f)

    # 2. Load A9 runs
    a9_path = repo_root / "outputs" / "isic2019" / "experiments" / "A9" / "isolated_a9_comparison.json"
    with open(a9_path, "r") as f:
        a9_data = json.load(f)

    # 3. Load acquisition histories
    acq_path = repo_root / "outputs" / "isic2019" / "experiments" / "A14" / "acquisition_histories.json"
    with open(acq_path, "r") as f:
        acq_data = json.load(f)

    # 1. Load test labels
    test_feat = torch.load(repo_root / "data" / "cached_features" / "test_features.pt", map_location="cpu", weights_only=False)
    y_true = np.array(test_feat["labels"])

    # Compute average confusion matrices across the 3 seeds
    def get_cms(runs):
        cms = []
        for r in runs:
            if "confusion_matrix" in r["test"]:
                cms.append(np.array(r["test"]["confusion_matrix"]))
            else:
                from sklearn.metrics import confusion_matrix
                cms.append(confusion_matrix(y_true, r["test"]["predictions"], labels=list(range(NUM_CLASSES))))
        return cms

    models = {
        "A1 (Class-Weighted Random)": get_cms(a14_data["A1"]),
        "A9 (Cold-Start + Class-Weighted)": get_cms(a9_data["A9"]["runs"]),
        "A11 (CSDA-AL Candidate)": get_cms(a14_data["A11"])
    }

    avg_cm = {}
    for m_name, cms in models.items():
        arr = np.mean([np.array(cm) for cm in cms], axis=0)
        avg_cm[m_name] = arr
        df_cm = pd.DataFrame(arr, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])
        df_cm.to_csv(out_dir / f"cm_{m_name.split()[0]}.csv")

    # Per-class metrics
    per_class_summary = {}
    for m_name in ["A1", "A9", "A11"]:
        if m_name in ["A1", "A11"]:
            runs = a14_data[m_name]
        else:
            runs = a9_data["A9"]["runs"]

        p_list, r_list, f1_list, pred_counts = [], [], [], []
        for r in runs:
            p_list.append([r["test"]["per_class"][c]["precision"] for c in range(NUM_CLASSES)])
            r_list.append([r["test"]["per_class"][c]["recall"] for c in range(NUM_CLASSES)])
            f1_list.append([r["test"]["per_class"][c]["f1"] for c in range(NUM_CLASSES)])
            preds = np.array(r["test"]["predictions"])
            counts = [int(np.sum(preds == c)) for c in range(NUM_CLASSES)]
            pred_counts.append(counts)

        per_class_summary[m_name] = {
            "precision": np.mean(p_list, axis=0),
            "recall": np.mean(r_list, axis=0),
            "f1": np.mean(f1_list, axis=0),
            "pred_counts": np.mean(pred_counts, axis=0)
        }

    # True test counts
    test_cm = np.array(models["A1 (Class-Weighted Random)"][0])
    true_counts = np.sum(test_cm, axis=1)

    # Build comparison table
    rows = []
    for c_idx, c_name in enumerate(CLASSES):
        rows.append({
            "Class": c_name,
            "True_Test_Count": int(true_counts[c_idx]),
            "True_Pct": f"{true_counts[c_idx] / np.sum(true_counts) * 100:.1f}%",
            "A1_Pred_Count": f"{per_class_summary['A1']['pred_counts'][c_idx]:.1f}",
            "A1_Rec(%)": f"{per_class_summary['A1']['recall'][c_idx]*100:.2f}%",
            "A1_Prec(%)": f"{per_class_summary['A1']['precision'][c_idx]*100:.2f}%",
            "A1_F1": f"{per_class_summary['A1']['f1'][c_idx]:.4f}",
            "A9_Rec(%)": f"{per_class_summary['A9']['recall'][c_idx]*100:.2f}%",
            "A9_F1": f"{per_class_summary['A9']['f1'][c_idx]:.4f}",
            "A11_Pred_Count": f"{per_class_summary['A11']['pred_counts'][c_idx]:.1f}",
            "A11_Rec(%)": f"{per_class_summary['A11']['recall'][c_idx]*100:.2f}%",
            "A11_Prec(%)": f"{per_class_summary['A11']['precision'][c_idx]*100:.2f}%",
            "A11_F1": f"{per_class_summary['A11']['f1'][c_idx]:.4f}",
        })

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(out_dir / "per_class_detailed_breakdown.csv", index=False)

    # 4. Minority False Positive / False Negative Breakdown
    error_analysis = {}
    for m_name in ["A1", "A9", "A11"]:
        cm = avg_cm[f"{m_name} (Class-Weighted Random)" if m_name == "A1" else (f"{m_name} (Cold-Start + Class-Weighted)" if m_name == "A9" else f"{m_name} (CSDA-AL Candidate)")]
        m_errors = {}
        for c_idx in [3, 5, 6, 7]: # AK, DF, VASC, SCC
            c_name = CLASSES[c_idx]
            total_true = np.sum(cm[c_idx, :])
            tp = cm[c_idx, c_idx]
            fn = total_true - tp
            # Where did false negatives go?
            fn_targets = {CLASSES[j]: cm[c_idx, j] for j in range(NUM_CLASSES) if j != c_idx}
            # Where did false positives come from?
            total_pred = np.sum(cm[:, c_idx])
            fp = total_pred - tp
            fp_sources = {CLASSES[i]: cm[i, c_idx] for i in range(NUM_CLASSES) if i != c_idx}
            m_errors[c_name] = {
                "Total_True": total_true,
                "TP": tp,
                "FN": fn,
                "Recall": tp / total_true if total_true > 0 else 0,
                "Top_FN_Confusion": sorted(fn_targets.items(), key=lambda x: x[1], reverse=True)[:3],
                "Total_Pred": total_pred,
                "FP": fp,
                "Precision": tp / total_pred if total_pred > 0 else 0,
                "Top_FP_Source": sorted(fp_sources.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        error_analysis[m_name] = m_errors

    with open(out_dir / "minority_error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2)

    # 5. Acquired class counts per round across AL rounds
    # Let's inspect final_class_counts in A1 and A11
    acq_counts_summary = {
        "A1_Final_Acquired_Labels": [r["final_class_counts"] for r in a14_data["A1"]],
        "A11_Final_Acquired_Labels": [r["final_class_counts"] for r in a14_data["A11"]],
    }
    with open(out_dir / "acquired_label_counts.json", "w") as f:
        json.dump(acq_counts_summary, f, indent=2)

    # Print summary to stdout
    print("=== Phase 3 Diagnostic Table ===")
    print(df_comp[["Class", "True_Test_Count", "A1_Rec(%)", "A1_Prec(%)", "A1_F1", "A9_Rec(%)", "A11_Rec(%)", "A11_Prec(%)", "A11_F1"]].to_string(index=False))

    print("\n=== Minority Error Breakdown for A11 ===")
    for c_name in ["DF", "VASC", "SCC", "AK"]:
        err = error_analysis["A11"][c_name]
        print(f"Class {c_name} (True count: {err['Total_True']}):")
        print(f"  TP: {err['TP']:.1f}, FN: {err['FN']:.1f} (Recall: {err['Recall']*100:.2f}%)")
        print(f"  Top FN Misclassifications (where true {c_name} went): {err['Top_FN_Confusion']}")
        print(f"  FP: {err['FP']:.1f} (Precision: {err['Precision']*100:.2f}%)")
        print(f"  Top FP Sources (who got mistakenly predicted as {c_name}): {err['Top_FP_Source']}")

if __name__ == "__main__":
    run_diagnostics()
