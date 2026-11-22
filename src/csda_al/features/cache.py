"""ID-aligned immutable embeddings and atomic, checksum-verified NPZ caches."""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


@dataclass(frozen=True)
class FeatureStore:
    image_ids: tuple[str, ...]
    features: np.ndarray
    encoder_id: str = "unspecified"
    encoder_version: str = "untrained"
    preprocessing_id: str = "unspecified"
    normalization: str = "none"
    split_fingerprint: str = "unspecified"
    schema_version: int = 1
    _index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        ids = tuple(self.image_ids)
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Feature image IDs must be nonempty, unique strings")
        arr = np.asarray(self.features)
        if arr.ndim != 2 or arr.shape[0] != len(ids) or arr.shape[1] == 0 or arr.dtype.kind != "f":
            raise ValueError("features must be a floating [N,D] matrix aligned with image_ids")
        if not np.isfinite(arr).all():
            raise ValueError("features must be finite")
        if self.normalization not in ("none", "l2"):
            raise ValueError("normalization must be 'none' or 'l2'")
        if self.normalization == "l2" and not np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5):
            raise ValueError("l2 metadata requires nonzero unit-norm features")
        if self.schema_version != 1:
            raise ValueError("Unsupported feature schema")
        for name in ("encoder_id", "encoder_version", "preprocessing_id", "split_fingerprint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        arr = np.array(arr, copy=True, order="C")
        arr.setflags(write=False)
        object.__setattr__(self, "image_ids", ids)
        object.__setattr__(self, "features", arr)
        object.__setattr__(self, "_index", MappingProxyType({i: j for j, i in enumerate(ids)}))

    def rows(self, image_ids: Sequence[str]) -> np.ndarray:
        ids = tuple(image_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Requested feature IDs must be unique")
        try:
            positions = [self._index[i] for i in ids]
        except KeyError as exc:
            raise KeyError(f"Missing feature ID: {exc.args[0]}") from exc
        result = self.features[positions].copy()
        result.setflags(write=False)
        return result

    @property
    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(json.dumps(self.image_ids, ensure_ascii=False).encode("utf-8"))
        hasher.update(str(self.features.dtype).encode())
        hasher.update(str(self.features.shape).encode())
        hasher.update(self.features.tobytes())
        return hasher.hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {name: getattr(self, name) for name in (
            "encoder_id", "encoder_version", "preprocessing_id", "normalization", "split_fingerprint", "schema_version")}
        manifest.update({"sha256": self.digest, "shape": list(self.features.shape), "dtype": str(self.features.dtype)})
        temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temp.open("wb") as stream:
                np.savez_compressed(stream, image_ids=np.asarray(self.image_ids), features=self.features,
                                    manifest=np.asarray(json.dumps(manifest, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, expected_encoder_version: str | None = None,
             expected_preprocessing_id: str | None = None,
             expected_split_fingerprint: str | None = None) -> "FeatureStore":
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"image_ids", "features", "manifest"}:
                raise ValueError("Invalid feature-cache fields")
            manifest = json.loads(str(payload["manifest"].item()))
            arr = payload["features"]
            ids = tuple(payload["image_ids"].tolist())
        checksum = manifest.pop("sha256")
        shape, dtype = manifest.pop("shape"), manifest.pop("dtype")
        store = cls(ids, arr, **manifest)
        if list(store.features.shape) != shape or str(store.features.dtype) != dtype or store.digest != checksum:
            raise ValueError("Feature-cache checksum/shape mismatch")
        for expected, actual, name in ((expected_encoder_version, store.encoder_version, "encoder"),
                                      (expected_preprocessing_id, store.preprocessing_id, "preprocessing"),
                                      (expected_split_fingerprint, store.split_fingerprint, "split")):
            if expected is not None and expected != actual:
                raise ValueError(f"Stale feature cache: {name} identity differs")
        return store


def save_feature_store(store: FeatureStore, path: str | Path) -> None:
    store.save(path)


def load_feature_store(path: str | Path, **kwargs: Any) -> FeatureStore:
    return FeatureStore.load(path, **kwargs)
