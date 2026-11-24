"""Interactive Checkpoint Inspector.

Allows you, your faculty, or any evaluator to instantly load, inspect,
and verify any saved PyTorch model checkpoint.
"""
from pathlib import Path
import torch

def inspect_all():
    repo_root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
    ckpt_root = repo_root / "outputs" / "isic2019" / "research_investigation" / "checkpoints"

    print("=================================================================")
    print("        ISIC 2019 SAVED CHECKPOINTS AUDIT & DIRECTORY")
    print("=================================================================\n")

    methods = sorted([d.name for d in ckpt_root.iterdir() if d.is_dir()])
    total_files = 0

    for m in methods:
        m_dir = ckpt_root / m
        files = list(m_dir.glob("**/*.pt"))
        total_files += len(files)
        print(f"📁 [{m}]: {len(files)} saved .pt checkpoints")

    print(f"\nTOTAL SAVED RESEARCH CHECKPOINTS: {total_files} files\n")

    # Inspect the top-performing model checkpoint (A1_Finetuned, Seed 44, Round 6, Budget 320)
    top_ckpt_path = ckpt_root / "A1_Finetuned" / "seed_44" / "round_0006_budget_320.pt"
    if top_ckpt_path.exists():
        print("=================================================================")
        print("   DEEP-INSPECTION: Top Model (A1_Finetuned Seed 44 Budget 320)")
        print("=================================================================")
        ckpt = torch.load(top_ckpt_path, map_location="cpu", weights_only=False)
        print(f"File Path:       {top_ckpt_path}")
        print(f"Strategy:        {ckpt.get('strategy', 'A1_Finetuned')}")
        print(f"Random Seed:     {ckpt.get('seed')}")
        print(f"Budget (Images): {ckpt.get('budget')} acquired images")
        print(f"Active Round:    Round {ckpt.get('round')}")
        print(f"Calibrated BA:   {ckpt.get('cal_ba', ckpt.get('calibrated_test_ba')):.2f}%")
        print(f"Calibrated F1:   {ckpt.get('cal_f1', ckpt.get('calibrated_test_f1')):.4f}")
        print(f"Optimal Tau:     {ckpt.get('best_tau')}")
        
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", {}))
        print(f"Neural Weights:  {len(state_dict)} trained parameter tensors")
        for k, v in list(state_dict.items())[:4]:
            print(f"   • {k}: shape {list(v.shape)}")
        print("   ...")

    # Also inspect A18 Improved MAL checkpoint
    a18_ckpt_path = ckpt_root / "A18_Improved_MAL" / "seed_42" / "round_0006_budget_320.pt"
    if a18_ckpt_path.exists():
        print("\n=================================================================")
        print("   DEEP-INSPECTION: Proposed A18 MAL (Seed 42 Budget 320)")
        print("=================================================================")
        ckpt = torch.load(a18_ckpt_path, map_location="cpu", weights_only=False)
        print(f"Strategy:        {ckpt.get('strategy')}")
        print(f"Random Seed:     {ckpt.get('seed')}")
        print(f"Acquired Images: {len(ckpt.get('labeled_indices', []))} sample IDs")
        print(f"Calibrated BA:   {ckpt.get('cal_ba'):.2f}%")
        print(f"Calibrated F1:   {ckpt.get('cal_f1'):.4f}")

if __name__ == "__main__":
    inspect_all()
