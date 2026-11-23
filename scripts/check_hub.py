import os
from pathlib import Path
import torch
from torchvision.models import ResNet18_Weights

hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
print("Hub checkpoints dir:", hub_dir)
if hub_dir.exists():
    print("Files in hub:", [f.name for f in hub_dir.iterdir()])
else:
    print("Hub checkpoints dir does not exist.")

filename = Path(ResNet18_Weights.DEFAULT.url).name
print("Expected ResNet18 filename:", filename)
target = hub_dir / filename
print("Target file exists?", target.is_file())
