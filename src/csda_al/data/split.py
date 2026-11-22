"""Label-independent connected-group splits with immutable content provenance."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .metadata import ImageRecord, SPLITS, validate_image_id
from ..utils.io import atomic_json, read_json, fingerprint

SPLIT_SCHEMA_VERSION = 3


def _validated_inputs(
    records: Sequence[ImageRecord],
    additional_groups: Mapping[str, str] | None = None,
    precomputed_sha256: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    ids = [validate_image_id(record.image_id) for record in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Split records require unique nonempty IDs")
    groups = dict(additional_groups or {})
    if not set(groups).issubset(ids) or any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in groups.values()
    ):
        raise ValueError("Additional duplicate groups must map known IDs to nonempty group identifiers")
    if precomputed_sha256 is None:
        content = {record.image_id: hashlib.sha256(record.image_path.read_bytes()).hexdigest()
                   for record in records}
    else:
        content = dict(precomputed_sha256)
        if set(content) != set(ids) or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in content.values()
        ):
            raise ValueError("Precomputed SHA256 must cover every image ID with a lowercase 64-character hash")
    return groups, content


def _identity(records: Sequence[ImageRecord], groups: Mapping[str, str], content: Mapping[str, str]) -> str:
    return fingerprint([
        {"image_id": record.image_id, "patient_id": record.patient_id,
         "lesion_id": record.lesion_id, "sha256": content[record.image_id],
         "additional_group": groups.get(record.image_id)}
        for record in sorted(records, key=lambda item: item.image_id)
    ])


def _keys(record: ImageRecord, groups: Mapping[str, str], content: Mapping[str, str]):
    return (("patient", record.patient_id), ("lesion", record.lesion_id),
            ("content", content[record.image_id]), ("additional", groups.get(record.image_id)))


def grouped_split(
    records: Sequence[ImageRecord], seed: int = 42,
    fractions: tuple[float, float, float] = (.7, .15, .15), check_content: bool = True,
    *, additional_groups: Mapping[str, str] | None = None,
    precomputed_sha256: Mapping[str, str] | None = None,
) -> tuple[list[ImageRecord], dict]:
    """Split connected patient/lesion/byte/externally supplied similarity groups, without labels.

    Additional groups come from label-free pixel identity and conservative
    grouping of unconfirmed similarity candidates; they do not prove duplication.
    Precomputed hashes must come from the checked source inventory. apply_split
    recomputes bytes by default to detect later source changes. Disabling content
    grouping is unsupported because it could create a leaking split.
    """
    if check_content is not True:
        raise ValueError("Content checking is mandatory for research splits")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    fractions_array = np.asarray(fractions, dtype=np.float64)
    if (fractions_array.shape != (3,) or not np.isfinite(fractions_array).all()
            or (fractions_array < 0).any() or not np.isclose(fractions_array.sum(), 1)):
        raise ValueError("Three finite nonnegative split fractions must sum to one")
    groups, content = _validated_inputs(records, additional_groups, precomputed_sha256)
    ids = [record.image_id for record in records]
    parent = {image_id: image_id for image_id in ids}

    def find(image_id: str) -> str:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    seen: dict[tuple[str, str], str] = {}
    for record in sorted(records, key=lambda item: item.image_id):
        for namespace, key in _keys(record, groups, content):
            if key is None:
                continue
            relation = (namespace, key)
            if relation in seen:
                parent[find(record.image_id)] = find(seen[relation])
            seen[relation] = record.image_id
    connected: dict[str, list[str]] = {}
    for image_id in sorted(ids):
        connected.setdefault(find(image_id), []).append(image_id)
    order = sorted(connected)
    np.random.default_rng(seed).shuffle(order)
    targets = fractions_array * len(ids)
    counts = np.zeros(3, dtype=int)
    assignments = {}
    for group_id in order:
        deficit = targets - counts
        deficit[fractions_array == 0] = -np.inf
        index = int(np.argmax(deficit))
        for image_id in connected[group_id]:
            assignments[image_id] = SPLITS[index]
        counts[index] += len(connected[group_id])
    result = [replace(record, split=assignments[record.image_id]) for record in records]
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION, "seed": int(seed), "label_independent": True,
        "dataset_fingerprint": _identity(records, groups, content), "assignments": assignments,
        "fractions": fractions_array.tolist(), "counts": dict(zip(SPLITS, counts.tolist())),
        "content_checked": True, "byte_sha256": content, "additional_groups": groups,
        "additional_groups_fingerprint": fingerprint(groups),
        "additional_group_policy": "label_free_pixel_identity_and_conservatively_grouped_unconfirmed_similarity_candidates",
        "connected_group_count": len(connected),
        "patient_ids_available": sum(record.patient_id is not None for record in records),
        "lesion_ids_available": sum(record.lesion_id is not None for record in records),
    }
    manifest["fingerprint"] = fingerprint(manifest)
    return result, manifest


def save_split(path: str | Path, manifest: dict) -> None:
    atomic_json(path, manifest)


def validate_split(
    records: Sequence[ImageRecord], manifest: dict,
    *, additional_groups: Mapping[str, str] | None = None,
    precomputed_sha256: Mapping[str, str] | None = None,
) -> list[ImageRecord]:
    """Check manifest integrity and all linked groups without accessing targets."""
    stored_hash = manifest.get("fingerprint")
    if stored_hash != fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}):
        raise ValueError("Split manifest fingerprint mismatch")
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION or manifest.get("content_checked") is not True:
        raise ValueError("Unsupported split manifest schema; Stage 1 content-checked schema 3 is required")
    stored_groups = manifest.get("additional_groups", {})
    selected_groups = stored_groups if additional_groups is None else dict(additional_groups)
    if (selected_groups != stored_groups
            or manifest.get("additional_groups_fingerprint") != fingerprint(selected_groups)):
        raise ValueError("Split manifest additional duplicate groups changed")
    groups, content = _validated_inputs(records, selected_groups, precomputed_sha256)
    if (manifest.get("dataset_fingerprint") != _identity(records, groups, content)
            or content != manifest.get("byte_sha256")):
        raise ValueError("Split manifest dataset fingerprint mismatch: image contents or group metadata changed")
    mapping = manifest.get("assignments", {})
    if set(mapping) != {record.image_id for record in records} or any(value not in SPLITS for value in mapping.values()):
        raise ValueError("Split manifest must cover exactly the dataset image IDs")
    counts = {split: sum(value == split for value in mapping.values()) for split in SPLITS}
    if manifest.get("counts") != counts:
        raise ValueError("Split manifest counts do not match assignments")
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        for namespace, key in _keys(record, groups, content):
            if key is None:
                continue
            relation = (namespace, key)
            if relation in seen and seen[relation] != mapping[record.image_id]:
                raise ValueError("Split manifest leaks a patient, lesion or duplicate image across splits")
            seen[relation] = mapping[record.image_id]
    return [replace(record, split=mapping[record.image_id]) for record in records]


def apply_split(
    records: Sequence[ImageRecord], path: str | Path,
    *, additional_groups: Mapping[str, str] | None = None,
    precomputed_sha256: Mapping[str, str] | None = None,
) -> tuple[list[ImageRecord], dict]:
    manifest = read_json(path)
    return validate_split(records, manifest, additional_groups=additional_groups,
                          precomputed_sha256=precomputed_sha256), manifest
