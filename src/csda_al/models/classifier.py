"""Eight diagnostic logits from an SSL-compatible ResNet-18 backbone."""
from pathlib import Path

import torch
from torch import nn

from .backbone import ResNet18Backbone


class Classifier(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = False,
                 backbone: nn.Module | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if num_classes != 8:
            raise ValueError("CSDA-AL uses exactly eight known ISIC 2019 classes")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.backbone = backbone if backbone is not None else ResNet18Backbone(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(512, num_classes)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encode(images)))

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Return probabilities; caller controls eval/dropout mode explicitly."""
        return torch.softmax(self(images), dim=1)

    def initialize_from_ssl(self, source: nn.Module | str | Path) -> None:
        if isinstance(source, (str, Path)):
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
            backbone_state = {k.removeprefix("backbone."): v for k, v in state.items()
                              if k.startswith("backbone.")}
            if not backbone_state:
                raise ValueError("Checkpoint does not contain an SSL backbone")
        else:
            backbone_state = source.backbone.state_dict()
        self.backbone.load_state_dict(backbone_state, strict=True)


ResNetClassifier = Classifier
