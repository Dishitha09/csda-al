"""Supervised optimization restricted to acquired TRAIN examples."""
from collections.abc import Iterable
import math

import numpy as np
import torch
from torch import nn

from ..features.extract import infer_batches
from .finetune import apply_training_modes, configure_trainable_parameters, optimizer_parameter_groups


class SupervisedTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 0.001,
                 freeze_stage: str = "head", weighted_ce: bool = False,
                 backbone_lr: float | None = None, weight_decay: float = 0.0) -> None:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr * 0.1
        if not math.isfinite(self.backbone_lr) or self.backbone_lr <= 0:
            raise ValueError("backbone_lr must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.weight_decay = weight_decay
        self.weighted_ce = weighted_ce
        self.total_steps = 0
        self.set_freeze_stage(freeze_stage)

    def set_freeze_stage(self, stage: str) -> None:
        """Rebuild groups while retaining momentum for previously trained parameters."""
        old_state = getattr(self, "optimizer", None)
        configure_trainable_parameters(self.model, stage)
        self.freeze_stage = stage
        self.optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(self.model, self.lr, self.backbone_lr), weight_decay=self.weight_decay
        )
        if old_state is not None:
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter in old_state.state:
                        self.optimizer.state[parameter] = old_state.state[parameter]

    def fit(self, loader: Iterable, epochs: int = 1, max_steps: int | None = None) -> dict:
        dataset = getattr(loader, "dataset", None)
        split = getattr(dataset, "split", None)
        if str(getattr(split, "value", split)).lower() != "train":
            raise ValueError("Supervised fit accepts TRAIN data only")
        if getattr(dataset, "acquired_only", False) is not True:
            raise ValueError("Supervised fit requires acquired labels only")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        weights = None
        if self.weighted_ce:
            counts = torch.zeros(8, dtype=torch.float32)
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError("Labeled batches must be (images,labels,ids)")
                labels = torch.as_tensor(batch[1], dtype=torch.long)
                if labels.numel() and (labels.min() < 0 or labels.max() > 7):
                    raise ValueError("Labels must be in [0,7]")
                counts += torch.bincount(labels, minlength=8)
            weights = torch.ones(8)
            observed = counts > 0
            weights[observed] = counts[observed].sum() / (observed.sum() * counts[observed])
            weights = weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        losses: list[float] = []
        steps = 0
        for _ in range(epochs):
            apply_training_modes(self.model, self.freeze_stage)
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError("Supervised fit requires (images,acquired_labels,image_ids)")
                images, labels, ids = batch
                if len(images) != len(ids):
                    raise ValueError("Training images and IDs are misaligned")
                labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
                if labels.ndim != 1 or len(labels) != len(images) or torch.any((labels < 0) | (labels > 7)):
                    raise ValueError("Training labels must be aligned class indices 0-7")
                self.optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(images.to(self.device)), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite supervised loss")
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
                steps += 1
                self.total_steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            if max_steps is not None and steps >= max_steps:
                break
        if not losses:
            raise ValueError("Cannot fit on an empty labeled loader")
        return {"loss": float(np.mean(losses)), "losses": losses, "steps": steps,
                "total_steps": self.total_steps, "freeze_stage": self.freeze_stage,
                "weighted_ce": self.weighted_ce}

    def predict(self, loader: Iterable) -> tuple[tuple[str, ...], np.ndarray]:
        return infer_batches(self.model, loader, device=self.device)

    def extract(self, loader: Iterable) -> tuple[tuple[str, ...], np.ndarray]:
        return infer_batches(self.model, loader, device=self.device, embeddings=True)
