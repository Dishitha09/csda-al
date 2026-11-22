"""Original PyTorch SimCLR: two views, shared encoder, symmetric NT-Xent."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ResNet18Backbone


class SimCLR(nn.Module):
    def __init__(self, backbone: nn.Module | None = None, projection_dim: int = 128,
                 hidden_dim: int = 512) -> None:
        super().__init__()
        if projection_dim < 1 or hidden_dim < 1:
            raise ValueError("Projection dimensions must be positive")
        self.backbone = backbone if backbone is not None else ResNet18Backbone()
        self.projection_head = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.encode(images))


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Mean over 2N anchors (half Google's sum of directional means)."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if z1.ndim != 2 or z1.shape != z2.shape or len(z1) < 2 or z1.shape[1] < 1:
        raise ValueError("Paired projections must have matching [N,D] shapes, N>=2")
    if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
        raise ValueError("Projections must be finite")
    z = F.normalize(torch.cat((z1, z2), dim=0).float(), dim=1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(len(z), device=z.device, dtype=torch.bool), -torch.inf)
    positive = (torch.arange(len(z), device=z.device) + len(z1)) % len(z)
    return F.cross_entropy(logits, positive)
