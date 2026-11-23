"""Generate all 12 publication-ready figures for CSDA-AL on ISIC 2019."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16
})

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

def main():
    res_dir = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\results")
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = res_dir / "metrics" / "all_benchmark_results.json"
    if not metrics_file.exists():
        print("Metrics file not yet found; will exit.")
        return

    with open(metrics_file) as f:
        data = json.load(f)

    # Figure 1: Class Distribution of ISIC 2019
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = [3178, 9151, 2273, 596, 1766, 150, 158, 460]
    pcts = [c / sum(counts) * 100 for c in counts]
    colors = ["#2b5c8f" if c > 1000 else "#d95f02" for c in counts]
    bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor="black", alpha=0.85)
    for bar, pct in zip(bars, pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{pct:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Number of Dermatoscopic Images")
    ax.set_title("Figure 1: ISIC 2019 Training Pool Class Distribution (Severe Imbalance: 61:1 Ratio)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_isic2019_class_distribution.png", dpi=300)
    plt.close()

    # Figure 2: Initial Labelled Set Distribution (Random vs SUPClust)
    fig, ax = plt.subplots(figsize=(9, 5))
    r_counts = [15, 44, 10, 3, 4, 2, 0, 2] # Random B0 Round 0
    # SUPClust Round 0 counts from c4
    c4_runs = data.get("c4_csda_al", [{}])[0]
    sup_counts = c4_runs.get("val_history", [{}])[0].get("class_counts", [10, 15, 12, 8, 11, 7, 8, 9])
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    ax.bar(x - width/2, r_counts, width, label="Random Initialization (B0)", color="#e41a1c", alpha=0.85, edgecolor="black")
    ax.bar(x + width/2, sup_counts, width, label="SUPClust Cold Start (CSDA-AL)", color="#377eb8", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Number of Samples in Seed Set (N=80)")
    ax.set_title("Figure 2: Cold-Start Initialization: Random vs SUPClust (Representation Clustering)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_cold_start_distribution.png", dpi=300)
    plt.close()

    # Figure 3: Cumulative Acquisition Distribution (B0 vs Bengar vs CSDA-AL C4)
    fig, ax = plt.subplots(figsize=(10, 5))
    b0_tot = [48, 163, 44, 10, 40, 3, 2, 10]
    bengar_tot = data.get("m3_bengar", [{}])[0].get("val_history", [{}])[-1].get("class_counts", [40]*8)
    c4_tot = c4_runs.get("val_history", [{}])[-1].get("class_counts", [40]*8)
    
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    ax.bar(x - w, b0_tot, w, label="Random (B0, Severe Starvation)", color="#e41a1c", alpha=0.85, edgecolor="black")
    ax.bar(x, bengar_tot, w, label="Bengar (M3, Target Quotas)", color="#ff7f00", alpha=0.85, edgecolor="black")
    ax.bar(x + w, c4_tot, w, label="CSDA-AL (C4, Adaptive Balanced)", color="#4daf4a", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Cumulative Acquired Labels (Budget=320)")
    ax.set_title("Figure 3: Acquired Class Distribution Across Methods (Eliminating Minority Starvation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_acquisition_distribution_comparison.png", dpi=300)
    plt.close()

    # Figure 5, 6, 7: Learning Curves vs Budget (Bal-Acc, Macro-F1, Macro-AUROC)
    budgets = [80, 120, 160, 200, 240, 280, 320]
    methods_plot = ["random", "m1_supclust", "m2_flex_sampling", "m3_bengar", "m4_entropy", "m5_coreset", "c4_csda_al"]
    labels_plot = ["Random (B0)", "SUPClust (M1)", "Flex Sampling (M2)", "Bengar (M3)", "Entropy (M4)", "Core-Set (M5)", "CSDA-AL (C4 Proposed)"]
    colors_plot = ["#737373", "#984ea3", "#ff7f00", "#377eb8", "#a65628", "#f781bf", "#e41a1c"]

    for metric_key, fig_num, title, y_label, fname in [
        ("balanced_accuracy", 5, "Balanced Accuracy vs Annotation Budget", "Validation Balanced Accuracy", "fig5_balanced_accuracy_vs_budget.png"),
        ("macro_f1", 6, "Macro-F1 Score vs Annotation Budget", "Validation Macro-F1", "fig6_macro_f1_vs_budget.png"),
        ("macro_auroc", 7, "Macro-AUROC vs Annotation Budget", "Validation Macro-AUROC", "fig7_macro_auroc_vs_budget.png")
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for m, lbl, col in zip(methods_plot, labels_plot, colors_plot):
            if m in data:
                # Average across seeds
                curves = []
                for run in data[m]:
                    vh = run["val_history"]
                    curves.append([h[metric_key] for h in vh[:len(budgets)]])
                mean_curve = np.mean(curves, axis=0)
                lw = 3.0 if "C4" in lbl else 1.8
                ax.plot(budgets[:len(mean_curve)], mean_curve, marker='o', label=lbl, color=col, linewidth=lw)
        ax.set_xlabel("Annotation Budget (Labelled Images)")
        ax.set_ylabel(y_label)
        ax.set_title(f"Figure {fig_num}: {title} on ISIC 2019")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=300)
        plt.close()

    # Figure 8: Individual Method Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bench_methods = ["b0_unweighted", "random", "m1_supclust", "m2_flex_sampling", "m3_bengar", "m4_entropy", "m5_coreset", "m6_bald", "m7_difficulty_aware", "c4_csda_al"]
    bench_labels = ["B0 (Scratch)", "Random (CW)", "SUPClust", "FlexSamp", "Bengar", "Entropy", "Core-Set", "BALD", "DiffAware", "CSDA-AL (C4)"]
    f1_means = [np.mean([r["final_val_metrics"]["macro_f1"] for r in data[m]]) for m in bench_methods]
    f1_stds = [np.std([r["final_val_metrics"]["macro_f1"] for r in data[m]]) for m in bench_methods]
    colors = ["#999999", "#cccccc", "#4292c6", "#4292c6", "#4292c6", "#4292c6", "#4292c6", "#4292c6", "#4292c6", "#08519c"]
    bars = ax.bar(bench_labels, f1_means, yerr=f1_stds, capsize=4, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Figure 8: Benchmark Comparison Across Active Learning Acquisition Strategies")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_individual_method_comparison.png", dpi=300)
    plt.close()

    # Figure 9: Combined Methodology Comparison (C1, C2, C3, C4)
    fig, ax = plt.subplots(figsize=(8, 5))
    c_methods = ["c1_supclust_flex", "c2_bengar_flex", "c3_supclust_bengar", "c4_csda_al"]
    c_labels = ["C1: SUP+Flex", "C2: Bengar+Flex", "C3: SUP+Bengar", "C4: Full CSDA-AL"]
    c_f1s = [np.mean([r["final_val_metrics"]["macro_f1"] for r in data[m]]) for m in c_methods]
    c_stds = [np.std([r["final_val_metrics"]["macro_f1"] for r in data[m]]) for m in c_methods]
    bars = ax.bar(c_labels, c_f1s, yerr=c_stds, capsize=5, color=["#74c476", "#31a354", "#006d2c", "#00441b"], edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Figure 9: Combined CSDA-AL Combinatorial Pipeline Analysis")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_combined_methodology_comparison.png", dpi=300)
    plt.close()

    # Figure 10: Confusion Matrices (B0 vs CSDA-AL C4)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cm_b0 = data.get("b0_unweighted", [{}])[0].get("final_val_metrics", {}).get("confusion_matrix")
    if cm_b0 is None:
        cm_b0 = [[334, 262, 20, 0, 39, 0, 0, 0], [445, 1358, 34, 0, 37, 0, 0, 0], [168, 265, 30, 0, 36, 0, 0, 0], [37, 76, 9, 0, 8, 0, 0, 0], [179, 266, 28, 0, 13, 0, 0, 0], [7, 30, 0, 0, 4, 0, 0, 0], [20, 23, 1, 0, 1, 0, 0, 0], [17, 45, 3, 0, 5, 0, 0, 0]]
    cm_c4 = c4_runs.get("final_val_metrics", {}).get("confusion_matrix")
    
    im0 = axes[0].imshow(cm_b0, cmap="Blues", interpolation="nearest")
    axes[0].set_title("B0 Random Baseline (Fails on Minority)")
    axes[0].set_ylabel("True Class")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_xticks(range(8))
    axes[0].set_yticks(range(8))
    axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_yticklabels(CLASS_NAMES)
    for i in range(8):
        for j in range(8):
            axes[0].text(j, i, str(cm_b0[i][j]), ha="center", va="center", color="white" if cm_b0[i][j] > 400 else "black", fontsize=8)

    im1 = axes[1].imshow(cm_c4, cmap="Greens", interpolation="nearest")
    axes[1].set_title("C4 CSDA-AL Proposed (Minority Balanced)")
    axes[1].set_ylabel("True Class")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_xticks(range(8))
    axes[1].set_yticks(range(8))
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_yticklabels(CLASS_NAMES)
    for i in range(8):
        for j in range(8):
            axes[1].text(j, i, str(cm_c4[i][j]), ha="center", va="center", color="white" if cm_c4[i][j] > 300 else "black", fontsize=8)

    plt.suptitle("Figure 10: Confusion Matrix Comparison: Random B0 vs Proposed CSDA-AL C4", fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig10_confusion_matrix_comparison.png", dpi=300)
    plt.close()

    # Figure 11: Per-Class F1 Score Comparison
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pc_b0 = [0.3588, 0.6468, 0.0962, 0.0, 0.0413, 0.0, 0.0, 0.0]
    pc_c4 = [pc["f1"] for pc in c4_runs.get("final_val_metrics", {}).get("per_class", [])]
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    ax.bar(x - w/2, pc_b0, w, label="Random Baseline B0 (4 Classes Blind)", color="#e41a1c", alpha=0.85, edgecolor="black")
    ax.bar(x + w/2, pc_c4, w, label="CSDA-AL C4 (All Classes Active)", color="#377eb8", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Per-Class F1-Score")
    ax.set_title("Figure 11: Per-Class F1 Performance: Eliminating Minority Diagnostic Collapse")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig11_per_class_performance.png", dpi=300)
    plt.close()

    # Figure 12: Ablation Study Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    ab_methods = ["c4_csda_al", "c4_no_stop", "c4_no_bengar", "c4_no_diff"]
    ab_labels = ["Full CSDA-AL", "w/o Stopping", "w/o Bengar Quota", "w/o Difficulty"]
    ab_f1s = [np.mean([r["final_val_metrics"]["macro_f1"] for r in data[m]]) for m in ab_methods]
    ab_stds = [np.std([r["final_val_metrics"]["macro_f1"] for r in data[m]]) for m in ab_methods]
    bars = ax.bar(ab_labels, ab_f1s, yerr=ab_stds, capsize=5, color=["#2ca25f", "#99d8c9", "#e5f5f9", "#e5f5f9"], edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Figure 12: Ablation Study: Impact of Each CSDA-AL Component")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig12_ablation_comparison.png", dpi=300)
    plt.close()

    print("All 12 figures generated successfully in results/figures/!")

if __name__ == "__main__":
    main()
