import os
from pathlib import Path
import yaml

base_dir = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
configs_dir = base_dir / "experiments" / "configs"
configs_dir.mkdir(parents=True, exist_ok=True)

methods = [
    {
        "id": "b0_random",
        "name": "Random Acquisition (B0)",
        "initial_strategy": "random",
        "strategy": "random",
        "stopping": False,
    },
    {
        "id": "m1_supclust",
        "name": "SUPClust Cold-Start (M1)",
        "initial_strategy": "supclust",
        "strategy": "random",
        "stopping": False,
    },
    {
        "id": "m2_flexsampling",
        "name": "Flexible Sampling (M2)",
        "initial_strategy": "random",
        "strategy": "flexible_sampling",
        "stopping": False,
    },
    {
        "id": "m3_bengar",
        "name": "Bengar Class-Balanced AL (M3)",
        "initial_strategy": "random",
        "strategy": "bengar_class_balanced",
        "stopping": False,
    },
    {
        "id": "m4_entropy",
        "name": "Entropy Uncertainty Sampling (M4)",
        "initial_strategy": "random",
        "strategy": "entropy",
        "stopping": False,
    },
    {
        "id": "m5_coreset",
        "name": "Core-Set Diversity Sampling (M5)",
        "initial_strategy": "random",
        "strategy": "coreset",
        "stopping": False,
    },
    {
        "id": "m6_bald",
        "name": "BALD MC-Dropout Uncertainty (M6)",
        "initial_strategy": "random",
        "strategy": "bald",
        "stopping": False,
    },
    {
        "id": "m7_difficulty_aware",
        "name": "Difficulty-Aware Sampling (M7)",
        "initial_strategy": "random",
        "strategy": "difficulty_aware",
        "stopping": False,
    },
    {
        "id": "c1_supclust_flexsampling",
        "name": "Combined C1: SUPClust + Flexible Sampling",
        "initial_strategy": "supclust",
        "strategy": "flexible_sampling",
        "stopping": False,
    },
    {
        "id": "c2_supclust_bengar",
        "name": "Combined C2: SUPClust + Bengar Class-Balanced",
        "initial_strategy": "supclust",
        "strategy": "bengar_class_balanced",
        "stopping": False,
    },
    {
        "id": "c3_flexsampling_bengar",
        "name": "Combined C3: FlexSampling + Bengar Class-Balanced",
        "initial_strategy": "random",
        "strategy": "bengar_class_balanced",
        "stopping": False,
    },
    {
        "id": "c4_csda_al_full",
        "name": "Combined C4: CSDA-AL Full Pipeline",
        "initial_strategy": "supclust",
        "strategy": "difficulty_aware",
        "stopping": True,
    },
    {
        "id": "c4_no_stopping_ablation",
        "name": "Ablation: CSDA-AL Full without Early Stopping",
        "initial_strategy": "supclust",
        "strategy": "difficulty_aware",
        "stopping": False,
    },
]

for m in methods:
    cfg = {
        "dataset": {
            "prepared_dir": "../../../data/isic2019/prepared",
            "image_dir": "../../../data/isic2019/images",
            "metadata_csv": None,
            "ground_truth_csv": None,
            "image_size": 224,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "ssl": {
            "checkpoint": "../../outputs/isic2019/simclr.pt",
            "epochs": 100,
            "batch_size": 64,
            "lr": 0.001,
            "temperature": 0.5,
            "projection_dim": 128,
        },
        "classifier": {
            "epochs": 5,
            "batch_size": 16,
            "lr": 0.001,
            "freeze_stage": "head",
            "pretrained": False,
            "dropout": 0.1,
            "weighted_ce": False,
        },
        "acquisition": {
            "initial_strategy": m["initial_strategy"],
            "strategy": m["strategy"],
            "initial_budget": 80,
            "batch_budget": 40,
            "alpha": 0.5,
            "diversity": False,
        },
        "stopping": {
            "enabled": m["stopping"],
            "min_rounds": 3,
            "patience": 2,
            "min_delta": 0.005,
            "min_per_class": 2,
        },
        "experiment": {
            "seed": 42,
            "max_annotations": 320,
            "max_rounds": 7,
            "output_dir": f"../../results/runs/{m['id']}_seed42",
            "device": "cpu",
            "num_workers": 0,
        },
    }
    out_file = configs_dir / f"{m['id']}.yaml"
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Created config: {out_file.name}")
