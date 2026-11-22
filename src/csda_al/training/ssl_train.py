"""Label-free SSL training; dry runs can cap the number of optimizer steps."""
from collections.abc import Iterable
import math

import numpy as np
import torch
from torch import nn

from ..models.simclr import nt_xent_loss


def train_ssl(model: nn.Module, loader: Iterable, *, device: str = "cpu", epochs: int = 1,
              max_steps: int | None = None, temperature: float = 0.5,
              lr: float = 0.001, optimizer: torch.optim.Optimizer | None = None) -> dict:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be positive")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    dataset = getattr(loader, "dataset", None)
    split = getattr(dataset, "split", "train")
    if str(getattr(split, "value", split)).lower() != "train":
        raise ValueError("SSL only accepts TRAIN images")
    model.to(device)
    model.train()
    optimizer = optimizer if optimizer is not None else torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        for batch in loader:
            if len(batch) != 2 or not isinstance(batch[0], (tuple, list)) or len(batch[0]) != 2:
                raise ValueError("SSL loader must yield ((view1,view2),image_ids), never targets")
            (view1, view2), ids = batch
            if len(view1) != len(ids) or len(view2) != len(ids):
                raise ValueError("Paired SSL views and IDs must align")
            optimizer.zero_grad(set_to_none=True)
            loss = nt_xent_loss(model(view1.to(device)), model(view2.to(device)), temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SSL loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if max_steps is not None and len(losses) >= max_steps:
                break
        if max_steps is not None and len(losses) >= max_steps:
            break
    if not losses:
        raise ValueError("Cannot pretrain an empty loader")
    return {"loss": float(np.mean(losses)), "losses": losses, "steps": len(losses), "target_labels_used": 0}


class SSLTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 0.001,
                 temperature: float = 0.5) -> None:
        self.model, self.device, self.lr, self.temperature = model, device, lr, temperature
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def fit(self, loader: Iterable, epochs: int = 1, max_steps: int | None = None) -> dict:
        return train_ssl(self.model, loader, device=self.device, lr=self.lr, temperature=self.temperature,
                         epochs=epochs, max_steps=max_steps, optimizer=self.optimizer)
