"""Declared head-only and staged ResNet unfreezing policies."""
from typing import Any
from torch import nn


def configure_trainable_parameters(model: nn.Module, stage: str = "head") -> None:
    if stage not in ("head", "layer4", "all"):
        raise ValueError("freeze stage must be head, layer4, or all")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    if stage == "all":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(True)
    elif stage == "layer4":
        for parameter in last_backbone_stage(model).parameters():
            parameter.requires_grad_(True)


def last_backbone_stage(model: nn.Module) -> nn.Module:
    backbone = model.backbone
    if hasattr(backbone, "network"):
        backbone = backbone.network
    if not hasattr(backbone, "layer4"):
        raise ValueError("layer4 unfreezing requires a ResNet backbone")
    return backbone.layer4


def apply_training_modes(model: nn.Module, stage: str) -> None:
    model.train()
    if stage != "all":
        model.backbone.eval()
        if stage == "layer4":
            last_backbone_stage(model).train()


def optimizer_parameter_groups(model: nn.Module, head_lr: float,
                               backbone_lr: float) -> list[dict[str, Any]]:
    head = [p for p in model.head.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = [{"params": head, "lr": head_lr, "name": "head"}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    grouped = [id(p) for group in groups for p in group["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != trainable:
        raise ValueError("Optimizer groups must cover trainable parameters exactly once")
    return groups
