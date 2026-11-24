import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
df = pd.read_csv(repo_root / "results" / "all_experiments.csv")

methods = ["A1_Frozen", "A1_Finetuned", "A9_ColdStart", "A17_MAL_Diversity", "A18_Improved_MAL"]
budgets = [80, 120, 160, 200, 240, 280, 320]

# 1. Budget Trajectory
traj_rows = []
for b in budgets:
    row = {"Budget": b}
    for m in methods:
        sub = df[(df["method"] == m) & (df["budget"] == b) & (df["calibration"].str.contains("BayesianPrior"))]
        vals = sub["balanced_accuracy"].values
        row[m] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}" if len(vals) else "N/A"
    traj_rows.append(row)

df_traj = pd.DataFrame(traj_rows)
df_traj.to_csv(repo_root / "results" / "tables" / "budget_trajectory_summary.csv", index=False)
print("=== BUDGET TRAJECTORY TABLE (Calibrated Balanced Accuracy %) ===")
print(df_traj.to_string(index=False))

# 2. Final Per-Class Recalls at Budget 320
final_rows = []
for m in methods:
    sub = df[(df["method"] == m) & (df["budget"] == 320) & (df["calibration"].str.contains("BayesianPrior"))]
    row = {
        "Method": m,
        "Bal_Acc": f"{sub['balanced_accuracy'].mean():.2f} ± {sub['balanced_accuracy'].std():.2f}",
        "Macro_F1": f"{sub['macro_f1'].mean():.4f} ± {sub['macro_f1'].std():.4f}",
        "Macro_AUROC": f"{sub['macro_auroc'].mean():.4f} ± {sub['macro_auroc'].std():.4f}",
        "DF_Rec(%)": f"{sub['DF_recall'].mean():.1f} ± {sub['DF_recall'].std():.1f}",
        "VASC_Rec(%)": f"{sub['VASC_recall'].mean():.1f} ± {sub['VASC_recall'].std():.1f}",
        "SCC_Rec(%)": f"{sub['SCC_recall'].mean():.1f} ± {sub['SCC_recall'].std():.1f}",
        "AK_Rec(%)": f"{sub['AK_recall'].mean():.1f} ± {sub['AK_recall'].std():.1f}",
        "BKL_Rec(%)": f"{sub['BKL_recall'].mean():.1f} ± {sub['BKL_recall'].std():.1f}",
    }
    final_rows.append(row)

df_final = pd.DataFrame(final_rows)
df_final.to_csv(repo_root / "results" / "tables" / "final_per_class_recalls.csv", index=False)
print("\n=== FINAL PER-CLASS RECALLS AT B=320 (Calibrated) ===")
print(df_final.to_string(index=False))

# 3. Decision Thresholding Impact (Argmax vs Calibration vs Opt-Thresholds)
thresh_rows = []
for m in methods:
    sub_arg = df[(df["method"] == m) & (df["budget"] == 320) & (df["threshold_method"] == "StandardArgmax")]
    sub_cal = df[(df["method"] == m) & (df["budget"] == 320) & (df["threshold_method"].str.contains("BayesianPrior"))]
    sub_opt = df[(df["method"] == m) & (df["budget"] == 320) & (df["threshold_method"] == "ValidationOptimizedThresholds")]

    row = {
        "Method": m,
        "Argmax_BalAcc": f"{sub_arg['balanced_accuracy'].mean():.2f} ± {sub_arg['balanced_accuracy'].std():.2f}",
        "Calibrated_BalAcc": f"{sub_cal['balanced_accuracy'].mean():.2f} ± {sub_cal['balanced_accuracy'].std():.2f}",
        "OptThreshold_BalAcc": f"{sub_opt['balanced_accuracy'].mean():.2f} ± {sub_opt['balanced_accuracy'].std():.2f}",
        "Argmax_MinorityRec": f"{((sub_arg['DF_recall'] + sub_arg['VASC_recall'] + sub_arg['SCC_recall'])/3.0).mean():.1f}%",
        "Calibrated_MinorityRec": f"{((sub_cal['DF_recall'] + sub_cal['VASC_recall'] + sub_cal['SCC_recall'])/3.0).mean():.1f}%",
        "OptThreshold_MinorityRec": f"{((sub_opt['DF_recall'] + sub_opt['VASC_recall'] + sub_opt['SCC_recall'])/3.0).mean():.1f}%",
    }
    thresh_rows.append(row)

df_thresh = pd.DataFrame(thresh_rows)
df_thresh.to_csv(repo_root / "results" / "tables" / "threshold_strategy_comparison.csv", index=False)
print("\n=== DECISION THRESHOLDING STRATEGY COMPARISON ===")
print(df_thresh.to_string(index=False))
