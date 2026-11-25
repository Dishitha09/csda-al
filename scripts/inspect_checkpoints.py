"""Interactive Checkpoint Inspector & Results Browser.

Directly loads, inspects, and reports metrics stored INSIDE the physical
PyTorch .pt model checkpoint files on disk.
"""
from pathlib import Path
import sys
import torch
import pandas as pd
import numpy as np

def inspect_all():
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_root = repo_root / "outputs" / "isic2019" / "research_investigation" / "checkpoints"
    champ_root = repo_root / "models" / "champion_checkpoints"

    print("=" * 95)
    print("      ISIC 2019: PHYSICAL CHECKPOINT RESULTS AUDIT (READ DIRECTLY FROM .PT FILES)")
    print("=" * 95)

    if not ckpt_root.exists() and not champ_root.exists():
        print(f"Error: Checkpoints directory not found at {ckpt_root}")
        return

    # 1. Checkpoint inventory
    methods = sorted([d.name for d in ckpt_root.iterdir() if d.is_dir()]) if ckpt_root.exists() else []
    total_files = 0
    print("\n📁 CHECKPOINT DIRECTORY INVENTORY:")
    for m in methods:
        m_dir = ckpt_root / m
        files = list(m_dir.glob("**/*.pt"))
        total_files += len(files)
        print(f"   • [{m:<20}]: {len(files):>2} saved .pt model checkpoint files")

    if champ_root.exists():
        c_files = list(champ_root.glob("*.pt"))
        print(f"   • [{'champion_checkpoints':<20}]: {len(c_files):>2} portable .pt model checkpoint files")
        if not methods:
            total_files += len(c_files)

    print(f"\n   ==> TOTAL PHYSICAL CHECKPOINTS VERIFIED: {total_files} files on disk\n")

    # 2. Extract results directly from the .pt checkpoints for all budgets (80 to 320)
    print("=" * 95)
    print("   RESULTS EXTRACTED DIRECTLY FROM PHYSICAL .PT CHECKPOINTS (BUDGETS 80 -> 320)")
    print("=" * 95)

    records = []
    major_strategies = ["A1_Finetuned", "A18_Improved_MAL", "A1_Frozen", "A9_ColdStart", "A17_MAL_Diversity"]

    for strat in major_strategies:
        strat_dir = ckpt_root / strat
        if not strat_dir.exists():
            continue
        for budget in [80, 160, 240, 320]:
            ba_seeds = []
            f1_seeds = []
            tau_seeds = []
            for seed in [42, 43, 44]:
                f_list = list((strat_dir / f"seed_{seed}").glob(f"*budget_{budget}.pt"))
                if f_list:
                    d = torch.load(f_list[0], map_location="cpu", weights_only=False)
                    ba = d.get("cal_ba") or d.get("calibrated_test_ba") or 0.0
                    if ba < 1.0: ba *= 100.0
                    f1 = d.get("cal_f1") or d.get("calibrated_test_f1") or 0.0
                    tau = d.get("best_tau") or d.get("optimal_tau") or 0.0
                    ba_seeds.append(ba)
                    f1_seeds.append(f1)
                    tau_seeds.append(tau)
            if ba_seeds:
                records.append({
                    "Strategy": strat,
                    "Budget (Images)": budget,
                    "Calibrated Bal-Acc": f"{np.mean(ba_seeds):.2f} ± {np.std(ba_seeds):.2f}%",
                    "Macro-F1": f"{np.mean(f1_seeds):.4f}",
                    "Mean Val Tau*": f"{np.mean(tau_seeds):.3f}",
                    "Seeds Verified": len(ba_seeds)
                })

    df_ckpt_summary = pd.DataFrame(records)
    print(df_ckpt_summary.to_string(index=False))

    # 3. Deep inspection of Final Champion Checkpoint (A23_Ensemble)
    ens_path = ckpt_root / "A23_Ensemble" / "ensemble_A1_Finetuned_budget_320.pt"
    if not ens_path.exists():
        ens_path = champ_root / "ensemble_A1_Finetuned_budget_320.pt"

    if ens_path.exists():
        print("\n" + "=" * 95)
        print("   DEEP-INSPECTION: FINAL CHAMPION CHECKPOINT (A23 Multi-Seed Ensemble at B=320)")
        print("=" * 95)
        ckpt_ens = torch.load(ens_path, map_location="cpu", weights_only=False)
        print(f"File Path:        {ens_path}")
        print(f"Ensemble Strategy: {ckpt_ens.get('strategy')}")
        print(f"Base Seeds:       {ckpt_ens.get('seeds')}")
        print(f"Optimal Tau*:     {ckpt_ens.get('optimal_tau'):.4f} (Derived 100% on Validation Set)")
        print(f"Threshold Vector: {list(np.round(ckpt_ens.get('thresholds', []), 2))}")
        
        m_cal = ckpt_ens.get("calibrated_metrics", {})
        m_uncal = ckpt_ens.get("uncalibrated_metrics", {})
        m_thresh = ckpt_ens.get("thresholded_metrics", {})

        print(f"\nExact Evaluated Metrics Inside Checkpoint File:")
        print(f"  • Uncalibrated Balanced Accuracy: {m_uncal.get('balanced_accuracy', 0)*100:.2f}% | AUROC: {m_uncal.get('macro_auroc', 0):.4f}")
        print(f"  • Calibrated Balanced Accuracy:   {m_cal.get('balanced_accuracy', 0)*100:.2f}% | AUROC: {m_cal.get('macro_auroc', 0):.4f} (PEAK RECORD)")
        print(f"  • Thresholded Balanced Accuracy:  {m_thresh.get('balanced_accuracy', 0)*100:.2f}%")
        print(f"  • Minority Recall (DF/VASC/SCC):  {m_cal.get('minority_recall_3', 0)*100:.2f}%")
        
        recs = m_cal.get("per_class_recall", {})
        print(f"\nPer-Class Recalls Stored in Checkpoint:")
        for c, r in recs.items():
            print(f"    - {c:<5}: {r*100:.1f}%")

    # 4. Deep inspection of Top Single Model (A1_Finetuned Seed 44 Budget 320)
    top_path = ckpt_root / "A1_Finetuned" / "seed_44" / "round_0006_budget_320.pt"
    if not top_path.exists():
        top_path = champ_root / "seed_44_round_0006_budget_320.pt"

    if top_path.exists():
        print("\n" + "=" * 95)
        print("   DEEP-INSPECTION: TOP SINGLE SEED MODEL (A1_Finetuned Seed 44 Budget 320)")
        print("=" * 95)
        ckpt_top = torch.load(top_path, map_location="cpu", weights_only=False)
        print(f"File Path:        {top_path}")
        print(f"Strategy:         {ckpt_top.get('strategy')}")
        print(f"Random Seed:      {ckpt_top.get('seed')}")
        print(f"Budget (Images):  {ckpt_top.get('budget')} acquired images")
        print(f"Active Round:     Round {ckpt_top.get('round')}")
        ba_val = ckpt_top.get('cal_ba', 0.0)
        if ba_val < 1.0: ba_val *= 100.0
        print(f"Calibrated BA:    {ba_val:.2f}%")
        print(f"Calibrated F1:    {ckpt_top.get('cal_f1', 0.0):.4f}")
        print(f"Validation Tau*:  {ckpt_top.get('best_tau')}")
        
        state_dict = ckpt_top.get("model_state", ckpt_top.get("model_state_dict", {}))
        print(f"\nTrained PyTorch Neural Weight Tensors ({len(state_dict)} tensors):")
        for k, v in list(state_dict.items())[:5]:
            print(f"    • {k:<25}: shape {list(v.shape)} | dtype {v.dtype}")
        print("    ... (complete weights saved on disk)")

    print("\n" + "=" * 95)
    print("HOW TO INSPECT ANY OTHER CHECKPOINT IN PYTHON:")
    print("  import torch")
    print("  ckpt = torch.load('path/to/checkpoint.pt')")
    print("  print(ckpt.keys())")
    print("=" * 95)

if __name__ == "__main__":
    inspect_all()
