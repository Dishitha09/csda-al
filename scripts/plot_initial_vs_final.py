"""Initial Baseline (A0) vs Final Champion Model (A23 Ensemble) Comparison Plot.

Generates a publication-quality side-by-side visual comparison showing the dramatic
performance leap from initial failure (0% minority recall) to final champion record.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
fig_dir = repo_root / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
out_png = fig_dir / "initial_vs_final_comparison.png"

def plot_comparison():
    # Style setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # -------------------------------------------------------------
    # 1. Aggregate Metric Comparison (Balanced Acc, AUROC, Minority Recall)
    # -------------------------------------------------------------
    metrics = ["Balanced\nAccuracy", "Diagnostic\nAUROC", "Macro-F1\nScore", "Minority Recall\n(DF/VASC/SCC)"]
    initial_scores = [26.06, 74.36, 26.06, 0.00]
    final_scores = [39.50, 80.81, 36.51, 29.12]
    
    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax1.bar(x - width/2, initial_scores, width, label="Initial Test (A0 Baseline)", color="#CBD5E0", edgecolor="#4A5568", linewidth=1.2)
    rects2 = ax1.bar(x + width/2, final_scores, width, label="Last Test (A23 Champion)", color="#2B6CB0", edgecolor="#1A365D", linewidth=1.2)

    ax1.set_title("Overall Model Performance: Initial vs Final", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 95)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#4A5568")

    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1A365D")

    # -------------------------------------------------------------
    # 2. Per-Class Recall Breakdown (Rescuing Rare Skin Cancers)
    # -------------------------------------------------------------
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    init_recs = [48.2, 86.4, 45.1, 12.1, 9.7, 0.0, 0.0, 7.5]
    final_recs = [44.8, 65.5, 50.3, 39.0, 29.0, 16.7, 36.0, 34.7]

    xc = np.arange(len(classes))
    rects_c1 = ax2.bar(xc - width/2, init_recs, width, label="Initial Test (A0)", color="#CBD5E0", edgecolor="#4A5568", linewidth=1.2)
    rects_c2 = ax2.bar(xc + width/2, final_recs, width, label="Final Test (A23)", color="#C53030", edgecolor="#742A2A", linewidth=1.2)

    ax2.set_title("Diagnostic Recall Across All 8 Classes (Rescuing Rare Killers)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Recall / Sensitivity (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(classes, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 98)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    for r in rects_c2[5:]: # highlight minority classes (DF, VASC, SCC)
        h = r.get_height()
        ax2.text(r.get_x() + r.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#C53030")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Successfully generated comparison figure: {out_png}")

if __name__ == "__main__":
    plot_comparison()
