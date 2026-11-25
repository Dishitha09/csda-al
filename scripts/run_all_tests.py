"""Master Test Runner & Interactive Experiment Suite for ISIC 2019.

Allows you to run, inspect, and verify any of the experimental tests conducted
in this research project from a single unified command-line interface.
"""
import sys
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

TESTS = [
    {
        "id": "1",
        "name": "Initial Baseline vs Final Champion Comparison",
        "script": "scripts/show_initial_vs_final.py",
        "desc": "Displays the exact side-by-side jump from A0 baseline (26.06%, 0% minority recall) to A23 champion (39.50%, 29.12% recall)."
    },
    {
        "id": "2",
        "name": "Audit All 196 Physical Model Checkpoints",
        "script": "scripts/inspect_checkpoints.py",
        "desc": "Directly loads and inspects PyTorch .pt model weights across all strategies, seeds, and rounds on disk."
    },
    {
        "id": "3",
        "name": "Zero-Leakage Split & Calibration Verification Audit",
        "script": "scripts/verify_calibration_and_dataset_split.py",
        "desc": "Formally proves zero overlap across 17,732 / 3,800 / 3,799 partitions and verifies validation prior calibration."
    },
    {
        "id": "4",
        "name": "Complete 9-Milestone Project Timeline",
        "script": "scripts/show_all_results.py",
        "desc": "Chronological progression of all 9 project milestones and top models from results/all_experiments.csv."
    },
    {
        "id": "5",
        "name": "Loss Function Benchmark Suite (7 Losses x 3 Seeds)",
        "script": "scripts/benchmark_loss_functions.py",
        "desc": "Evaluates Standard CE, Class-Weighted CE, Focal Loss, CB-Focal, Cui et al., Balanced Softmax, and Logit Adjustment."
    },
    {
        "id": "6",
        "name": "Backbone Fine-Tuning & MAL Active Learning (A1 vs A17)",
        "script": "scripts/run_improved_mal_and_thresholding.py",
        "desc": "Controlled evaluation of Frozen ResNet-18 vs Adapted Backbone across 7 budgets (80 to 320) and 3 seeds."
    },
    {
        "id": "7",
        "name": "A18 4-Component Active Learning Experiment",
        "script": "scripts/run_a18_experiment.py",
        "desc": "Evaluates Margin Uncertainty + Feature Diversity + Minority Deficit + Redundancy Penalty (29.07% Bal-Acc)."
    },
    {
        "id": "8",
        "name": "Deep Ensemble & Semi-Supervised Self-Training (A22 & A23)",
        "script": "scripts/run_advanced_improvements.py",
        "desc": "Trains student models on 17,412 unlabeled images and ensembles fine-tuned models to achieve all-time record 39.50%."
    },
    {
        "id": "9",
        "name": "Empirical Scaling Law Trajectory (B=320 to 17,732)",
        "script": "scripts/run_high_performance_scaling.py",
        "desc": "Evaluates model scaling from 1.8% to 100% data, proving the 14.4% sweet spot at B=2,560 (42.60% Bal-Acc)."
    },
    {
        "id": "10",
        "name": "Generate Publication Figures & Comparison Plot",
        "script": "scripts/plot_initial_vs_final.py",
        "desc": "Generates high-resolution publication figures on Desktop and results/figures/."
    }
]

def print_menu():
    print("=" * 85)
    print("       CSDA-AL ON ISIC 2019: MASTER EXPERIMENTAL TEST RUNNER & SUITE")
    print("=" * 85)
    for t in TESTS:
        print(f"  [{t['id']:>2}] {t['name']}")
        print(f"       Script: {t['script']}")
        print(f"       Desc:   {t['desc']}\n")
    print("  [ A] Run All Fast Audits (Tests 1, 2, 3, 4, 10)")
    print("  [ Q] Quit")
    print("=" * 85)

def run_script(script_rel_path: str):
    p = repo_root / script_rel_path
    if not p.exists():
        print(f"Error: Script not found at {p}")
        return
    print(f"\n>>> EXECUTING: python {script_rel_path} ...\n")
    subprocess.run([sys.executable, str(p)], cwd=str(repo_root))

def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        print_menu()
        choice = input("Enter test number to run (1-10, A for all fast audits, Q to quit): ").strip()

    if choice.upper() == "Q":
        return
    elif choice.upper() == "A":
        for num in ["1", "2", "3", "4", "10"]:
            for t in TESTS:
                if t["id"] == num:
                    run_script(t["script"])
    else:
        matched = [t for t in TESTS if t["id"] == choice]
        if matched:
            run_script(matched[0]["script"])
        else:
            print(f"Invalid selection: '{choice}'. Please select a valid test ID from 1 to 10.")

if __name__ == "__main__":
    main()
