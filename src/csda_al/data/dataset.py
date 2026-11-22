"""Distinct image-only and acquired-label datasets."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .metadata import ImageRecord


class ImageDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], transform: Callable | None = None):
        self.records = tuple(records)
        ids = [r.image_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset IDs must be unique")
        self.transform = transform
        splits = {r.split for r in self.records}
        if len(splits) > 1:
            raise ValueError("A dataset must contain a single split")
        self.split = next(iter(splits), "train")

    def __len__(self) -> int:
        return len(self.records)

    def image(self, index: int):
        with Image.open(self.records[index].image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image) if self.transform else image

    def __getitem__(self, index: int):
        return self.image(index), self.records[index].image_id


class LabeledImageDataset(ImageDataset):
    def __init__(self, records: Sequence[ImageRecord], labels: Mapping[str, int],
                 transform: Callable | None = None, *, acquired_only: bool = False):
        super().__init__(records, transform)
        if {r.image_id for r in self.records} != set(labels):
            raise ValueError("Labels must match exactly the dataset IDs")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 8 for v in labels.values()):
            raise ValueError("Labels must be integer indices 0..7")
        if self.split == "train" and not acquired_only:
            raise ValueError("TRAIN labeled datasets must be explicitly acquired_only")
        self.acquired_only = acquired_only
        self._acquired_labels = dict(labels)

    def __getitem__(self, index: int):
        image, image_id = super().__getitem__(index)
        return image, self._acquired_labels[image_id], image_id
