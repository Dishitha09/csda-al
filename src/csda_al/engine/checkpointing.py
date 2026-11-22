"""Atomic state-dict checkpoints with architecture metadata and safe loading."""
import os
from pathlib import Path
import random
from typing import Any, cast
import uuid

import torch
import numpy as np
from torch import nn

SCHEMA_VERSION = 1
CLASS_ORDER = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")


def save_checkpoint(path: str | Path, model: nn.Module, *, metadata: dict | None = None,
                    optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    data = {"schema_version": SCHEMA_VERSION, "model_type": type(model).__name__,
            "class_order": CLASS_ORDER, "model_state": model.state_dict(),
            "metadata": dict(metadata or {}), "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}
    if torch.cuda.is_available():
        data["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        data["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        data["scheduler_state"] = scheduler.state_dict()
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return data["metadata"]


def load_checkpoint(path: str | Path, model: nn.Module, *,
                    optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None,
                    map_location: str = "cpu", expected_metadata: dict | None = None,
                    restore_rng: bool = False) -> dict:
    data = torch.load(path, map_location=map_location, weights_only=True)
    if data.get("schema_version") != SCHEMA_VERSION or tuple(data.get("class_order", ())) != CLASS_ORDER:
        raise ValueError("Incompatible checkpoint schema or class order")
    if data.get("model_type") != type(model).__name__:
        raise ValueError("Checkpoint model architecture differs")
    metadata = data.get("metadata", {})
    if expected_metadata is not None and any(metadata.get(k) != v for k, v in expected_metadata.items()):
        raise ValueError("Checkpoint metadata mismatch")
    model.load_state_dict(data["model_state"], strict=True)
    if optimizer is not None:
        if "optimizer_state" not in data:
            raise ValueError("Checkpoint lacks requested optimizer state")
        optimizer.load_state_dict(data["optimizer_state"])
    if scheduler is not None:
        if "scheduler_state" not in data:
            raise ValueError("Checkpoint lacks requested scheduler state")
        scheduler.load_state_dict(data["scheduler_state"])
    if restore_rng:
        torch.set_rng_state(data["torch_rng_state"].cpu())
        random.setstate(data["python_rng_state"])
        numpy_state = data["numpy_rng_state"]
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        if "cuda_rng_states" in data and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(data["cuda_rng_states"])
    return metadata
