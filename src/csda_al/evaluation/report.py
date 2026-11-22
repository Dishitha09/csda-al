"""Versioned evaluation results, final-test identity locks, and atomic exports."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import io
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

from csda_al.data.metadata import CLASSES
from csda_al.utils.io import atomic_json
from .metrics import classification_metrics


@dataclass(frozen=True)
class EvaluationLock:
    """Declared final identities; an audit control, not a security boundary."""

    checkpoint_id: str
    config_fingerprint: str
    split_fingerprint: str
    locked: bool = True

    def __post_init__(self) -> None:
        for field in (self.checkpoint_id, self.config_fingerprint, self.split_fingerprint):
            if not isinstance(field, str) or not field.strip():
                raise ValueError("test evaluation locks require nonempty checkpoint/config/split identities")
        if not isinstance(self.locked, bool):
            raise ValueError("locked must be boolean")

    def validate(self, checkpoint_id: str, config_fingerprint: str, split_fingerprint: str) -> None:
        """Reject unlocked runs and any changed final-test identity."""
        if not self.locked:
            raise ValueError("test evaluation requires a locked final evaluation protocol")
        actual = (checkpoint_id, config_fingerprint, split_fingerprint)
        expected = (self.checkpoint_id, self.config_fingerprint, self.split_fingerprint)
        if actual != expected:
            raise ValueError("test evaluation identities differ from the final evaluation lock")


@dataclass(frozen=True)
class EvaluationResult:
    """JSON-safe observations at a disclosed training and validation label cost."""

    split: str
    metrics: dict[str, Any]
    label_budget: int
    validation_annotation_cost: int
    checkpoint_id: str = ""
    config_fingerprint: str = ""
    split_fingerprint: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test"}:
            raise ValueError("result split must be validation or test")
        for value in (self.label_budget, self.validation_annotation_cost):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("annotation costs must be nonnegative integers")
        if self.schema_version != 1:
            raise ValueError("unsupported evaluation result schema version")
        if self.metrics.get("label_order") != list(CLASSES):
            raise ValueError("evaluation results require the fixed eight-class label order")
        if len(self.metrics.get("support", [])) != len(CLASSES):
            raise ValueError("evaluation results require eight support entries")
        if np.asarray(self.metrics.get("confusion_matrix", [])).shape != (len(CLASSES), len(CLASSES)):
            raise ValueError("evaluation results require an eight-by-eight confusion matrix")
        json.dumps(self.metrics, allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["total_annotation_cost"] = self.label_budget + self.validation_annotation_cost
        return result

    @property
    def per_class_support(self) -> tuple[int, ...]:
        return tuple(self.metrics["support"])

    @property
    def confusion_matrix(self) -> list[list[int]]:
        return [list(row) for row in self.metrics["confusion_matrix"]]


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    probabilities_or_pred: Sequence[int] | np.ndarray,
    split: str = "validation",
    lock: EvaluationLock | None = None,
    checkpoint_id: str = "",
    config_fingerprint: str = "",
    split_fingerprint: str = "",
    label_budget: int = 0,
    validation_annotation_cost: int = 0,
) -> EvaluationResult:
    """Evaluate validation or explicitly locked test predictions, with 8 classes.

    Targets belong to this evaluator alone and must never enter QueryContext.
    The lock validates declared identities, not the file contents themselves;
    callers compute fingerprints from committed config, split, and checkpoint.
    """
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be 'validation' or 'test'")
    if split == "test":
        if not isinstance(lock, EvaluationLock):
            raise ValueError("test evaluation requires an EvaluationLock")
        lock.validate(checkpoint_id, config_fingerprint, split_fingerprint)
    for name, value in (("label_budget", label_budget), ("validation_annotation_cost", validation_annotation_cost)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    values = np.asarray(probabilities_or_pred)
    if values.ndim == 2:
        if values.shape[1] != len(CLASSES):
            raise ValueError("probability matrix must have exactly eight columns in fixed class order")
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("probabilities must be finite numeric values")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("probabilities must lie within [0, 1]")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("each probability row must sum to one")
        predicted = values.argmax(axis=1)
    elif values.ndim == 1:
        predicted = values
    else:
        raise ValueError("predictions must be class indices or an N-by-eight probability matrix")
    y_prob = values if values.ndim == 2 else None
    metrics = classification_metrics(y_true, predicted, len(CLASSES), y_prob=y_prob)
    return EvaluationResult(
        split=split, metrics=metrics, label_budget=int(label_budget),
        validation_annotation_cost=int(validation_annotation_cost), checkpoint_id=checkpoint_id,
        config_fingerprint=config_fingerprint, split_fingerprint=split_fingerprint,
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_result(
    result: EvaluationResult,
    output_dir: str | Path,
    prefix: str = "evaluation",
    *,
    synthetic: bool = False,
) -> dict[str, Path]:
    """Atomically write individual JSON and CSV files, explicitly tagged by provenance.

    The three-file export is not an all-or-nothing filesystem transaction. The
    JSON is the canonical complete result; CSV files are derived convenience
    views. Callers should use a unique run directory to avoid mixed versions.
    """
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be a nonempty filename component")
    folder = Path(output_dir)
    paths = {
        "json": folder / f"{prefix}.json",
        "per_class_csv": folder / f"{prefix}_per_class.csv",
        "confusion_csv": folder / f"{prefix}_confusion.csv",
    }
    payload = result.to_dict()
    payload["provenance"] = "synthetic_demonstration_no_real_training" if synthetic else "caller_supplied_evaluation_predictions"
    atomic_json(paths["json"], payload)
    table = io.StringIO(newline="")
    rows = result.metrics["per_class"]
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(paths["per_class_csv"], table.getvalue())
    matrix = io.StringIO(newline="")
    csv_writer = csv.writer(matrix)
    csv_writer.writerow(["true_class\\predicted_class", *result.metrics["label_order"]])
    for name, row in zip(result.metrics["label_order"], result.metrics["confusion_matrix"]):
        csv_writer.writerow([name, *row])
    _atomic_text(paths["confusion_csv"], matrix.getvalue())
    return paths
