"""Deterministic inference returns persistent image IDs with every row."""
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .cache import FeatureStore


def infer_batches(model: nn.Module, loader: Iterable, *, device: str | torch.device = "cpu",
                  embeddings: bool = False) -> tuple[tuple[str, ...], np.ndarray]:
    modes = [(module, module.training) for module in model.modules()]
    ids: list[str] = []
    outputs: list[np.ndarray] = []
    model.to(device)
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                if len(batch) not in (2, 3):
                    raise ValueError("Inference batch must be (images,ids) or (images,labels,ids)")
                images, batch_ids = batch[0], batch[-1]
                if isinstance(batch_ids, str):
                    raise ValueError("Expected a batch of IDs")
                batch_ids = tuple(batch_ids)
                if len(images) != len(batch_ids):
                    raise ValueError("Images and IDs must align")
                values = model.encode(images.to(device)) if embeddings else torch.softmax(model(images.to(device)), dim=1)
                if values.ndim != 2 or not torch.isfinite(values).all():
                    raise ValueError("Invalid model inference output")
                ids.extend(batch_ids)
                outputs.append(values.detach().cpu().numpy())
    finally:
        for module, was_training in modes:
            module.train(was_training)
    if not outputs:
        raise ValueError("Inference loader is empty")
    if len(set(ids)) != len(ids):
        raise ValueError("Inference loader returned duplicate IDs")
    return tuple(ids), np.concatenate(outputs)


def extract_features(model: nn.Module, loader: Iterable, *, device: str = "cpu",
                     normalization: str = "none", encoder_id: str = "resnet18",
                     encoder_version: str = "untrained", preprocessing_id: str = "deterministic",
                     split_fingerprint: str = "unspecified") -> FeatureStore:
    ids, values = infer_batches(model, loader, device=device, embeddings=True)
    if normalization == "l2":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(values.dtype).eps):
            raise ValueError("Cannot L2-normalize a zero feature vector")
        values = values / norms
    return FeatureStore(ids, values, encoder_id, encoder_version, preprocessing_id,
                        normalization, split_fingerprint)


extract_embeddings = extract_features
