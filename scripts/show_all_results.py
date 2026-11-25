"""Comprehensive Results Viewer for VS Code Terminal.

Displays the complete chronological progression of all experiments from start (A0)
to final champion model (A23 Ensemble) across all seeds and milestones.
"""
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
csv_path = repo_root / "results" / "all_experiments.csv"

def show_timeline():
    print("=" * 95)
    print("   ISIC 2019 COMPLETE EXPERIMENTAL RESULTS TIMELINE (FROM INITIAL BASELINE TO FINAL CHAMPION)")
    print("=" * 95)

    stages = [
        ("Milestone 1: A0 Unweighted Baseline", "A0_Random", "Frozen ResNet-18", "Standard CE", "26.06%", "0.7436", "0.0% (Complete Collapse)"),
        ("Milestone 2: A1 Bounded Loss Baseline", "A1_Frozen", "Frozen ResNet-18", "Class-Weighted CE", "28.28%", "0.7488", "5.3%"),
        ("Milestone 3: Bayesian Prior Calibration", "A1_Frozen (Cal)", "Frozen ResNet-18", "Class-Weighted CE", "31.01%", "0.7626", "11.7%"),
        ("Milestone 4: Loss Benchmark (Balanced Softmax)", "Balanced_Softmax", "Frozen ResNet-18", "Balanced Softmax", "31.59%", "0.7540", "24.0%"),
        ("Milestone 5: Backbone Fine-Tuning (Champion)", "A1_Finetuned", "ResNet-18+Adapter", "Class-Weighted CE", "31.99%", "0.7719", "18.3% (Peak Seed: 36.68%)"),
        ("Milestone 6: Active Learning Evolution (A18)", "A18_Improved_MAL", "ResNet-18+Adapter", "Class-Weighted CE", "29.07%", "0.7486", "14.3% (Thresh: 29.87%)"),
        ("Milestone 7: Scaling Law (14% Data Sweet Spot)", "B=2560 Scaling", "ResNet-18+Adapter", "Class-Weighted CE", "42.60%", "0.8313", "57.3% (94% Ceiling)"),
        ("Milestone 8: Clinical Safety Triage Mode", "A21_ClinicalTriage", "ResNet-18+Adapter", "Class-Weighted CE", "23.82%", "0.7690", "89.5% Melanoma Sensitivity"),
        ("Milestone 9: Semi-Supervised Self-Training", "A22_SemiSupervised", "ResNet-18+Adapter", "CW-CE + PseudoLabel", "32.13%", "0.7374", "22.0% (Thresh: 33.19%)"),
        ("Milestone 9: Deep Ensemble (FINAL RECORD)", "A23_Ensemble", "Multi-Seed Ensemble", "Class-Weighted CE", "39.50%", "0.8081", "29.12% (Thresh: 39.65%)"),
    ]

    df_summary = pd.DataFrame(stages, columns=[
        "Project Milestone", "Method Code", "Backbone Type", "Loss Function", "Bal-Acc", "AUROC", "Minority Recall"
    ])
    print(df_summary.to_string(index=False))

    if csv_path.exists():
        df_all = pd.read_csv(csv_path)
        print("\n" + "=" * 95)
        print(f"   MASTER EXPERIMENT REGISTRY: {len(df_all)} TOTAL VERIFIED EXPERIMENTAL RUNS")
        print("=" * 95)
        print(f"Unique methods evaluated: {df_all['method'].nunique()}")
        print(f"Random seeds covered:     {sorted([s for s in df_all['seed'].unique() if s != -1])}")
        print(f"Label budgets evaluated:  {sorted(df_all['budget'].unique())}")
        print("\nTop 5 Performing Models by Balanced Accuracy in results/all_experiments.csv:")
        top5 = df_all.sort_values(by="balanced_accuracy", ascending=False).head(5)
        cols_to_show = ["experiment_id", "method", "seed", "budget", "balanced_accuracy", "macro_auroc", "DF_recall", "VASC_recall", "SCC_recall"]
        print(top5[cols_to_show].to_string(index=False))

    print("\n" + "=" * 95)
    print("HOW TO VIEW THESE IN VS CODE:")
    print("  1. Press Ctrl + Shift + V on results/reports/MODEL_IMPROVEMENT_PROGRESS_REPORT.md for full report")
    print("  2. Open notebooks/CSDA_AL_Presentation_Demo.ipynb to run interactive charts")
    print("  3. Open results/all_experiments.csv to view all 386 raw experiment records")
    print("=" * 95)

if __name__ == "__main__":
    show_timeline()
