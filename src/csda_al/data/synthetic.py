"""Tiny generated RGB images for wiring checks, never ISIC research results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

from .metadata import ImageRecord


def create_synthetic_dataset(root: str | Path, seed: int = 42, *, train_size: int = 32,
                             validation_size: int = 8, test_size: int = 8,
                             image_size: int = 32) -> tuple[list[ImageRecord], dict[str, int]]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records, truth = [], {}
    for split, count in (("train", train_size), ("validation", validation_size), ("test", test_size)):
        for index in range(count):
            image_id = f"synthetic_{split}_{index:04d}"
            # Uneven TRAIN supports; validation/test cycle classes explicitly.
            label = int(rng.choice(8, p=np.array([16, 8, 4, 2, 1, 1, 1, 1]) / 34)) if split == "train" else index % 8
            pixels = rng.integers(0, 100, size=(image_size, image_size, 3), dtype=np.uint8)
            pixels[image_size//4:3*image_size//4, image_size//4:3*image_size//4, label % 3] += 100
            path = root / f"{image_id}.png"
            if not path.exists():
                Image.fromarray(pixels).save(path)
            records.append(ImageRecord(image_id, path, lesion_id=image_id, split=split))
            truth[image_id] = label
    return records, truth
