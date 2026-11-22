"""ISIC 2019 named-column ingestion; truth is never attached to ImageRecord."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

CLASSES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
SPLITS = ("train", "validation", "test")
MISSING_GROUP_TOKENS = frozenset({"", "nan", "none", "null", "na", "n/a", "<na>", "nat", "unknown", "unk"})


def normalize_group_id(value: str | None) -> str | None:
    """Normalize declared missing identifiers without inventing patient linkage."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Patient/lesion identifiers must be strings or null")
    cleaned = value.strip()
    return None if cleaned.casefold() in MISSING_GROUP_TOKENS else cleaned


def validate_image_id(value: str) -> str:
    """Require a portable neutral identifier, never a directory or diagnosis path."""
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
            or value in {".", ".."}):
        raise ValueError("image_id must be a nonempty safe identifier without whitespace or path components")
    return value


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    patient_id: str | None = None
    lesion_id: str | None = None
    split: str = "train"

    def __post_init__(self) -> None:
        validate_image_id(self.image_id)
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split: {self.split}")
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "patient_id", normalize_group_id(self.patient_id))
        object.__setattr__(self, "lesion_id", normalize_group_id(self.lesion_id))


def load_metadata(image_dir: str | Path, metadata_csv: str | Path | None = None) -> list[ImageRecord]:
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"ISIC image directory does not exist: {root}. Supply dataset.image_dir.")
    # Enumerate actual files once. Probing both .png and .PNG on Windows
    # otherwise counts one physical file twice on a case-insensitive volume.
    files_by_stem: dict[str, list[Path]] = {}
    for image_path in sorted(root.iterdir()):
        if image_path.is_file() and image_path.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
            files_by_stem.setdefault(image_path.stem, []).append(image_path)
    if metadata_csv is None:
        rows = [{"image": image_id} for image_id in sorted(files_by_stem)]
    else:
        path = Path(metadata_csv)
        if not path.is_file():
            raise ValueError(f"Missing ISIC metadata CSV: {path}")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if (not reader.fieldnames or "image" not in reader.fieldnames
                    or len(reader.fieldnames) != len(set(reader.fieldnames))):
                raise ValueError("Metadata CSV requires an image column")
            rows = list(reader)
    records = []
    seen = set()
    for row in rows:
        image_id = row["image"].strip()
        validate_image_id(image_id)
        if not image_id or image_id in seen or Path(image_id).name != image_id:
            raise ValueError(f"Empty, duplicate or unsafe image ID: {image_id}")
        seen.add(image_id)
        matches = files_by_stem.get(image_id, [])
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one image file for {image_id}, found {len(matches)}")
        records.append(ImageRecord(image_id, matches[0], row.get("patient_id") or None,
                                   row.get("lesion_id") or None))
    if not records:
        raise ValueError("No images found in the supplied dataset")
    return records


def load_truth(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing ISIC ground-truth CSV: {path}")
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or not {"image", *CLASSES}.issubset(reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError("Ground-truth CSV needs image and all eight named class columns")
        for row in reader:
            values = [float(row[c]) for c in CLASSES]
            if any(x not in (0.0, 1.0) for x in values) or sum(values) != 1:
                raise ValueError(f"Expected exactly one eight-class one-hot target: {row['image']}")
            if "UNK" in row and float(row["UNK"] or 0) != 0:
                raise ValueError("Unknown-class targets are outside the eight-class closed-set protocol")
            key = row["image"].strip()
            validate_image_id(key)
            if not key or key in result:
                raise ValueError("Empty or duplicate truth image ID")
            result[key] = values.index(1.0)
    return result
