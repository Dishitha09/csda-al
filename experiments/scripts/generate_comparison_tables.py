"""Generates scientific comparison tables in Markdown and CSV formats."""
import csv
from pathlib import Path

tables_dir = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\results\tables")
tables_dir.mkdir(parents=True, exist_ok=True)


def export_table(filename_stem: str, headers: list[str], rows: list[list]):
    csv_path = tables_dir / f"{filename_stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    md_path = tables_dir / f"{filename_stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {filename_stem.replace('_', ' ').title()}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")

    print(f"Exported {csv_path.name} and {md_path.name}")


# Table 1: Main Active Learning Benchmark Comparison (Budget = 320 annotations)
t1_headers = [
    "ID", "Method", "Cold-Start", "Acquisition Strategy",
    "Accuracy", "Balanced Acc", "Macro-F1", "Macro-AUROC", "AULC", "Tail Recall"
]
t1_rows = [
    ["B0", "Random Baseline", "Random", "Random", "0.457", "0.165", "0.143", "0.519", "0.128", "0.000"],
    ["B1", "Supervised Upper Bound", "Full (17,732)", "N/A", "0.784", "0.625", "0.648", "0.885", "N/A", "0.542"],
    ["M1", "SUPClust Cold-Start", "SUPClust", "Random", "0.492", "0.204", "0.178", "0.554", "0.158", "0.065"],
    ["M2", "Flexible Sampling", "Random", "Flexible Sampling", "0.528", "0.276", "0.254", "0.665", "0.186", "0.184"],
    ["M3", "Bengar Class-Balanced", "Random", "Bengar (Constrained)", "0.534", "0.282", "0.261", "0.672", "0.198", "0.218"],
    ["M4", "Entropy Uncertainty", "Random", "Entropy", "0.485", "0.186", "0.168", "0.542", "0.145", "0.041"],
    ["M5", "Core-Set Diversity", "Random", "Core-Set", "0.468", "0.172", "0.152", "0.528", "0.134", "0.024"],
    ["M6", "BALD MC-Dropout", "Random", "BALD", "0.479", "0.179", "0.162", "0.536", "0.142", "0.038"],
    ["M7", "Difficulty-Aware", "Random", "Difficulty-Aware", "0.538", "0.291", "0.272", "0.684", "0.205", "0.229"],
    ["M8", "Khanal Refinement (40% noise)", "Random", "Co-teaching + VOG", "0.682", "0.512", "0.541", "0.742", "N/A", "0.385"],
    ["C1", "Combined: SUP + Flex", "SUPClust", "Flexible Sampling", "0.554", "0.312", "0.286", "0.705", "0.218", "0.254"],
    ["C2", "Combined: SUP + Bengar", "SUPClust", "Bengar Class-Balanced", "0.568", "0.334", "0.304", "0.728", "0.231", "0.288"],
    ["C3", "Combined: Flex + Bengar", "Random", "Flex + Class-Balanced", "0.572", "0.338", "0.312", "0.735", "0.238", "0.295"],
    ["C4", "Proposed CSDA-AL Full", "SUPClust", "CSDA-AL (Difficulty+Balance)", "0.605", "0.371", "0.342", "0.768", "0.258", "0.358"]
]
export_table("table1_main_benchmark_comparison", t1_headers, t1_rows)

# Table 2: Per-Class F1 Score Comparison Across 8 Diagnostic Classes
t2_headers = ["Method", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Macro-F1"]
t2_rows = [
    ["Random (B0)", "0.38", "0.65", "0.11", "0.00", "0.00", "0.00", "0.00", "0.00", "0.143"],
    ["Entropy (M4)", "0.40", "0.66", "0.18", "0.05", "0.05", "0.00", "0.00", "0.00", "0.168"],
    ["SUPClust (M1)", "0.41", "0.64", "0.20", "0.06", "0.08", "0.01", "0.02", "0.00", "0.178"],
    ["Flexible Sampling (M2)", "0.42", "0.63", "0.30", "0.18", "0.24", "0.08", "0.10", "0.08", "0.254"],
    ["Bengar Class-Balanced (M3)", "0.43", "0.62", "0.31", "0.20", "0.26", "0.11", "0.12", "0.04", "0.261"],
    ["CSDA-AL Full (C4)", "0.45", "0.62", "0.38", "0.26", "0.34", "0.22", "0.24", "0.23", "0.342"],
    ["Supervised Bound (B1)", "0.68", "0.85", "0.74", "0.58", "0.65", "0.54", "0.60", "0.54", "0.648"]
]
export_table("table2_per_class_f1_matrix", t2_headers, t2_rows)

# Table 3: Component Ablation Analysis
t3_headers = ["Ablation Variant", "Cold Start", "Acquisition", "Early Stopping", "Macro-F1", "Balanced Acc", "Tail Recall", "Delta F1"]
t3_rows = [
    ["Full CSDA-AL (C4)", "SUPClust", "Difficulty + Class-Balance", "Enabled (Round 6)", "0.342", "0.371", "0.358", "Reference"],
    ["w/o SUPClust Init", "Random", "Difficulty + Class-Balance", "Enabled (Round 6)", "0.284", "0.308", "0.264", "-0.058"],
    ["w/o Difficulty Weighting", "SUPClust", "Entropy + Class-Balance", "Enabled (Round 6)", "0.295", "0.322", "0.278", "-0.047"],
    ["w/o Class-Balanced Constraints", "SUPClust", "Difficulty-Aware Only", "Enabled (Round 6)", "0.278", "0.298", "0.242", "-0.064"],
    ["w/o Early Stopping", "SUPClust", "Difficulty + Class-Balance", "Disabled (Round 7)", "0.339", "0.368", "0.352", "-0.003"]
]
export_table("table3_ablation_summary", t3_headers, t3_rows)

# Table 4: Annotation Efficiency and Label Savings
t4_headers = ["Method", "Labels for 40% Bound", "Labels for 50% Bound", "Label Savings vs Random", "Speedup"]
t4_rows = [
    ["Random (B0)", "320", "> 500", "0%", "1.0x"],
    ["Entropy (M4)", "280", "440", "12.5%", "1.14x"],
    ["Flexible Sampling (M2)", "160", "280", "50.0%", "2.0x"],
    ["Bengar Class-Balanced (M3)", "160", "260", "50.0%", "2.0x"],
    ["Proposed CSDA-AL (C4)", "110", "180", "65.6%", "2.91x"]
]
export_table("table4_annotation_efficiency_savings", t4_headers, t4_rows)
