"""ResNet-18 representations with explicit, offline-safe initialization."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Backbone(nn.Module):
    output_dim = 512

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.network = resnet18(weights=None)
        if pretrained:
            # Never fetch weights as a side effect of constructing a model.
            filename = Path(ResNet18_Weights.DEFAULT.url).name
            cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not cached.is_file():
                raise FileNotFoundError(
                    f"ImageNet weights are not cached at {cached}. Supply an approved local "
                    "checkpoint or use pretrained=False; no weights were downloaded."
                )
            self.network.load_state_dict(torch.load(cached, map_location="cpu", weights_only=True))
        self.network.fc = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [N,3,H,W]")
        return self.network(images)


def build_backbone(pretrained: bool = False) -> ResNet18Backbone:
    return ResNet18Backbone(pretrained=pretrained)
