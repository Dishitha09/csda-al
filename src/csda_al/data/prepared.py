"""Prepared label-free manifests and role-separated private annotation stores.

Only the preparation boundary receives the full truth mapping. Runtime TRAIN
truth is loaded inside LabelOracle.from_prepared_train; validation has a distinct
loader. No runtime helper here opens the sealed final-test label file.
"""
from __future__ import annotations

import csv
import hashlib
from numbers import Integral
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

from .metadata import CLASSES, SPLITS, ImageRecord, validate_image_id
from .split import SPLIT_SCHEMA_VERSION, validate_split
from ..utils.io import atomic_json, fingerprint, read_json

BUNDLE_SCHEMA_VERSION = 1
PUBLIC_FIELDS = ("image_id", "relative_path", "patient_id", "lesion_id", "duplicate_group_id", "byte_sha256", "split")
PRIVATE_FIELDS = ("image_id", "class_index", "split")
PRIVATE_FILENAMES = {"train": "train_oracle.csv", "validation": "validation_labels.csv", "test": "test_labels.csv"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _with_fingerprint(value: dict) -> dict:
    return {**value, "fingerprint": fingerprint(value)}


def _verify_fingerprint(value: dict, name: str) -> None:
    if (not isinstance(value, dict) or value.get("fingerprint")
            != fingerprint({key: item for key, item in value.items() if key != "fingerprint"})):
        raise ValueError(f"{name} fingerprint mismatch")


def _neutral_image_path(image_root: Path, relative: str, image_id: str) -> Path:
    path = PurePosixPath(relative)
    if (not isinstance(relative, str) or "\\" in relative or path.is_absolute()
            or len(path.parts) != 2 or path.parts[0] != "images"
            or any(part in {".", ".."} or ":" in part for part in path.parts)
            or path.as_posix() != relative or path.stem != image_id
            or path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}):
        raise ValueError("Prepared image paths must be neutral images/<image_id>.<extension> paths")
    destination = (image_root / path).resolve()
    if not destination.is_relative_to(image_root) or not destination.is_file():
        raise ValueError("Prepared image path is missing or escapes the approved dataset root")
    return destination


def prepare_bundle(
    records: Sequence[ImageRecord], truth: Mapping[str, int], manifest: dict,
    bundle_dir: str | Path, image_root: str | Path,
) -> dict:
    """Publish a frozen split and separate label stores without selecting by truth.

    image_root is the dataset root containing ``images/``; bundle_dir must be
    its direct child, normally ``prepared/``. The split is validated before
    truth is partitioned. This function never repairs or rebalances assignments
    based on labels. Existing bundles cannot be overwritten implicitly.
    """
    root = Path(image_root).resolve()
    destination = Path(bundle_dir).resolve()
    if not root.is_dir() or destination.parent != root:
        raise ValueError("Prepared bundle must be a direct child of the approved dataset root")
    if destination.exists():
        raise ValueError("Prepared bundle already exists; use a new reviewed output directory")
    # Source inventory has already hashed every checked image. Use those hashes
    # to avoid a redundant full-dataset pass, retaining full provenance in schema3.
    validated = validate_split(records, manifest, precomputed_sha256=manifest.get("byte_sha256"))
    if any(a.split != b.split for a, b in zip(records, validated, strict=True)):
        raise ValueError("Prepared records must retain the already frozen split assignments")
    ids = {record.image_id for record in records}
    if set(truth) != ids or any(
        isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < len(CLASSES)
        for value in truth.values()
    ):
        raise ValueError("Preparation truth must cover exactly the frozen records with valid eight-class indices")
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    labels_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for record in sorted(records, key=lambda item: item.image_id):
        try:
            relative = record.image_path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Prepared images must remain inside the approved dataset root") from exc
        _neutral_image_path(root, relative, record.image_id)
        rows_by_split[record.split].append({
            "image_id": record.image_id, "relative_path": relative,
            "patient_id": record.patient_id or "", "lesion_id": record.lesion_id or "",
            "duplicate_group_id": manifest["additional_groups"].get(record.image_id, ""),
            "byte_sha256": manifest["byte_sha256"][record.image_id], "split": record.split,
        })
        labels_by_split[record.split].append({"image_id": record.image_id,
                                              "class_index": int(truth[record.image_id]), "split": record.split})
    temporary = Path(tempfile.mkdtemp(prefix=".prepared-build-", dir=root)).resolve()
    if temporary.parent != root:
        raise RuntimeError("Unexpected bundle staging directory")
    try:
        public = {}
        private = {}
        for split in SPLITS:
            public_path = temporary / "public" / f"{split}.csv"
            _write_csv(public_path, PUBLIC_FIELDS, rows_by_split[split])
            public[split] = {"path": f"public/{split}.csv", "sha256": _digest(public_path),
                             "rows": len(rows_by_split[split])}
            private_path = temporary / "private" / PRIVATE_FILENAMES[split]
            _write_csv(private_path, PRIVATE_FIELDS, labels_by_split[split])
            private[split] = {"path": f"private/{PRIVATE_FILENAMES[split]}", "sha256": _digest(private_path),
                              "rows": len(labels_by_split[split]), "role": split}
        atomic_json(temporary / "public" / "split_manifest.json", manifest)
        public_manifest = _with_fingerprint({
            "schema_version": BUNDLE_SCHEMA_VERSION, "bundle_type": "csda_al_prepared_isic2019",
            "image_root": "../", "class_order": list(CLASSES),
            "split_fingerprint": manifest["fingerprint"], "dataset_fingerprint": manifest["dataset_fingerprint"],
            "public_manifests": public,
            "split_manifest": {"path": "public/split_manifest.json",
                               "sha256": _digest(temporary / "public" / "split_manifest.json")},
            "validation_annotation_cost": len(rows_by_split["validation"]),
            "label_independent_split": True,
            "sealing": "separate_role_files_and_runtime_loaders_not_cryptographic_or_OS_access_control",
        })
        atomic_json(temporary / "bundle.json", public_manifest)
        atomic_json(temporary / "private" / "manifest.json", _with_fingerprint({
            "schema_version": BUNDLE_SCHEMA_VERSION, "class_order": list(CLASSES),
            "public_bundle_fingerprint": public_manifest["fingerprint"],
            "split_fingerprint": manifest["fingerprint"], "label_files": private,
        }))
        # Windows virus/index scanners can briefly hold freshly written files.
        # Retry only this transient permission failure; never replace an existing
        # reviewed bundle or hide other publication errors.
        for attempt in range(6):
            try:
                if destination.exists():
                    raise ValueError("Prepared bundle appeared during publication; refusing replacement")
                os.rename(temporary, destination)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * (2 ** attempt))
        return public_manifest
    finally:
        if temporary.exists():
            if temporary.parent != root or not temporary.name.startswith(".prepared-build-"):
                raise RuntimeError("Refusing cleanup outside the checked dataset staging directory")
            shutil.rmtree(temporary)


def load_bundle_manifest(bundle_dir: str | Path) -> dict:
    """Read public bundle metadata only; no private target file is opened."""
    bundle = Path(bundle_dir).resolve()
    value = read_json(bundle / "bundle.json")
    _verify_fingerprint(value, "Prepared bundle")
    if (value.get("schema_version") != BUNDLE_SCHEMA_VERSION
            or value.get("bundle_type") != "csda_al_prepared_isic2019"
            or value.get("class_order") != list(CLASSES)
            or value.get("image_root") != "../" or value.get("label_independent_split") is not True):
        raise ValueError("Incompatible prepared bundle schema, label order or dataset root")
    return value


def load_prepared_records(
    bundle_dir: str | Path, split: str, *, verify_images: bool = False,
) -> list[ImageRecord]:
    """Load one label-free public manifest, without opening other splits or truth.

    Public CSV checksum and path confinement are always checked. Optional image
    byte verification is a deliberate full-input I/O pass, useful before training.
    """
    if split not in SPLITS:
        raise ValueError("Prepared split must be train, validation or test")
    bundle = Path(bundle_dir).resolve()
    info = load_bundle_manifest(bundle)
    descriptor = info["public_manifests"][split]
    expected_path = f"public/{split}.csv"
    if descriptor.get("path") != expected_path:
        raise ValueError("Prepared public manifest has an unexpected path")
    path = bundle / expected_path
    if _digest(path) != descriptor.get("sha256"):
        raise ValueError("Prepared public manifest checksum mismatch")
    split_descriptor = info.get("split_manifest", {})
    if split_descriptor.get("path") != "public/split_manifest.json":
        raise ValueError("Prepared split manifest has an unexpected path")
    split_path = bundle / "public" / "split_manifest.json"
    if _digest(split_path) != split_descriptor.get("sha256"):
        raise ValueError("Prepared frozen split checksum mismatch")
    frozen = read_json(split_path)
    _verify_fingerprint(frozen, "Prepared frozen split")
    if (frozen.get("schema_version") != SPLIT_SCHEMA_VERSION or frozen.get("content_checked") is not True
            or frozen.get("fingerprint") != info["split_fingerprint"]
            or frozen.get("dataset_fingerprint") != info["dataset_fingerprint"]):
        raise ValueError("Prepared public and frozen split identities differ")
    records = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PUBLIC_FIELDS:
            raise ValueError("Prepared public manifest fields must be exactly the label-free schema")
        for row in reader:
            image_id = validate_image_id(row["image_id"])
            if row["split"] != split or image_id in seen:
                raise ValueError("Prepared public manifest contains duplicate IDs or incorrect split markers")
            if (frozen["assignments"].get(image_id) != split
                    or frozen["byte_sha256"].get(image_id) != row["byte_sha256"]
                    or frozen["additional_groups"].get(image_id, "") != row["duplicate_group_id"]):
                raise ValueError("Prepared public rows differ from frozen split/content/duplicate identities")
            seen.add(image_id)
            image_path = _neutral_image_path(bundle.parent, row["relative_path"], image_id)
            if verify_images and _digest(image_path) != row["byte_sha256"]:
                raise ValueError("Prepared image byte checksum mismatch")
            records.append(ImageRecord(image_id, image_path, row["patient_id"], row["lesion_id"], split))
    if len(records) != descriptor.get("rows"):
        raise ValueError("Prepared public manifest row count mismatch")
    if seen != {image_id for image_id, assignment in frozen["assignments"].items() if assignment == split}:
        raise ValueError("Prepared public IDs do not equal the frozen split IDs")
    return records


def _load_private_truth(bundle_dir: str | Path, split: str) -> dict[str, int]:
    """Internal TRAIN/validation provider; final-test targets are never loaded."""
    if split not in {"train", "validation"}:
        raise ValueError("Runtime label providers cannot open sealed final-test truth")
    bundle = Path(bundle_dir).resolve()
    public = load_bundle_manifest(bundle)
    records = load_prepared_records(bundle, split)
    manifest = read_json(bundle / "private" / "manifest.json")
    _verify_fingerprint(manifest, "Private label manifest")
    if (manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
            or manifest.get("class_order") != list(CLASSES)
            or manifest.get("public_bundle_fingerprint") != public["fingerprint"]
            or manifest.get("split_fingerprint") != public["split_fingerprint"]):
        raise ValueError("Private labels do not match the prepared public bundle")
    descriptor = manifest["label_files"][split]
    expected_path = f"private/{PRIVATE_FILENAMES[split]}"
    if descriptor.get("path") != expected_path or descriptor.get("role") != split:
        raise ValueError("Private label store role/path mismatch")
    path = bundle / expected_path
    if _digest(path) != descriptor.get("sha256"):
        raise ValueError("Private label store checksum mismatch")
    truth: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PRIVATE_FIELDS:
            raise ValueError("Private labels require the exact image_id,class_index,split schema")
        for row in reader:
            image_id = validate_image_id(row["image_id"])
            try:
                label = int(row["class_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Private class indices must be integer strings") from exc
            if row["split"] != split or image_id in truth or not 0 <= label < len(CLASSES):
                raise ValueError("Private labels contain invalid targets, duplicates or another split")
            truth[image_id] = label
    if set(truth) != {record.image_id for record in records} or len(truth) != descriptor.get("rows"):
        raise ValueError("Private labels must cover exactly their public split IDs")
    return truth


def load_validation_truth(bundle_dir: str | Path) -> dict[str, int]:
    """Reveal the separately costed validation labels, never TRAIN/test truth."""
    return _load_private_truth(bundle_dir, "validation")
