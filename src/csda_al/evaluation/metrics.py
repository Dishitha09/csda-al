"""Pure classification metrics with explicit zero-support behavior.

The diagnostic order is fixed by :data:`csda_al.data.metadata.CLASSES`.
No frequency function accepts unlabeled-pool truth; callers supply only acquired
training labels. Arrays are copied or treated read-only and never mutated.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Sequence

import numpy as np

from csda_al.data.metadata import CLASSES


def _class_count(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    return int(num_classes)


def _labels(values: Sequence[int] | np.ndarray, num_classes: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional class-index array")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer class indices")
    if np.any(array < 0) or np.any(array >= num_classes):
        raise ValueError(f"{name} contains indices outside [0, {num_classes})")
    return array.astype(np.int64, copy=False)


def _names(num_classes: int) -> list[str]:
    return list(CLASSES) if num_classes == len(CLASSES) else [f"class_{i}" for i in range(num_classes)]


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int = 8,
) -> dict[str, Any]:
    """Return JSON-safe metrics; rows=true and columns=predicted in confusion.

    Macro metrics average *all configured classes* with zero division set to
    zero. Balanced accuracy averages recall over classes with true support.
    Empty evaluation sets have no accuracy/balanced accuracy (``None``), while
    per-class and aggregate precision/recall/F1 use explicit zero division.
    """
    count = _class_count(num_classes)
    truth = _labels(y_true, count, "y_true")
    predicted = _labels(y_pred, count, "y_pred")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    confusion = np.bincount(
        truth * count + predicted, minlength=count * count,
    ).reshape(count, count)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    correct = np.diag(confusion)
    precision = np.divide(correct, predicted_count, out=np.zeros(count), where=predicted_count > 0)
    recall = np.divide(correct, support, out=np.zeros(count), where=support > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(count), where=(precision + recall) > 0,
    )
    total = int(support.sum())
    micro = float(correct.sum() / total) if total else 0.0
    names = _names(count)
    result: dict[str, Any] = {
        "num_classes": count,
        "label_order": names,
        "n_examples": total,
        "accuracy": micro if total else None,
        "balanced_accuracy": float(recall[support > 0].mean()) if total else None,
        "balanced_accuracy_policy": "mean_recall_over_classes_with_true_support",
        "macro_policy": "all_configured_classes_zero_division_zero",
        "zero_division": 0.0,
        "confusion_matrix": confusion.tolist(),
        "support": support.tolist(),
        "zero_support_classes": [names[i] for i in range(count) if support[i] == 0],
        "zero_support_class_indices": np.flatnonzero(support == 0).tolist(),
        "per_class": [
            {
                "class_index": i, "class_name": names[i],
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
                "predicted_count": int(predicted_count[i]), "supported": bool(support[i]),
            }
            for i in range(count)
        ],
    }
    weights = support / total if total else np.zeros(count)
    for metric, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        result[f"macro_{metric}"] = float(values.mean())
        result[f"micro_{metric}"] = micro
        result[f"weighted_{metric}"] = float(np.dot(values, weights))
    return result


def frequency_groups(
    acquired_training_labels: Sequence[int] | np.ndarray,
    num_classes: int = 8,
    *,
    head_min: int = 20,
    middle_min: int = 5,
) -> dict[str, Any]:
    """Group classes by *acquired* training counts and declared thresholds.

    These defaults are reporting conventions, not published ISIC boundaries.
    Unobserved classes belong to tail and remain explicitly identifiable.
    """
    count = _class_count(num_classes)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (head_min, middle_min)):
        raise ValueError("frequency thresholds must be integers")
    if not 0 < middle_min < head_min:
        raise ValueError("thresholds must satisfy 0 < middle_min < head_min")
    labels = _labels(acquired_training_labels, count, "acquired_training_labels")
    counts = np.bincount(labels, minlength=count)
    return {
        "source": "acquired_training_labels_only",
        "counts": counts.tolist(),
        "head_min": int(head_min), "middle_min": int(middle_min),
        "head": np.flatnonzero(counts >= head_min).tolist(),
        "middle": np.flatnonzero((counts >= middle_min) & (counts < head_min)).tolist(),
        "tail": np.flatnonzero(counts < middle_min).tolist(),
        "unobserved": np.flatnonzero(counts == 0).tolist(),
        "label_order": _names(count),
    }


def head_middle_tail_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    acquired_training_labels: Sequence[int] | np.ndarray,
    num_classes: int = 8,
    *,
    head_min: int = 20,
    middle_min: int = 5,
) -> dict[str, Any]:
    """Report group metrics without inspecting any hidden training labels."""
    metrics = classification_metrics(y_true, y_pred, num_classes)
    grouping = frequency_groups(
        acquired_training_labels, num_classes, head_min=head_min, middle_min=middle_min,
    )
    summaries = {}
    for group in ("head", "middle", "tail"):
        rows = [metrics["per_class"][i] for i in grouping[group]]
        supported = [row for row in rows if row["supported"]]
        summaries[group] = {
            "class_indices": grouping[group],
            "support": sum(row["support"] for row in rows),
            "supported_classes": len(supported),
            "macro_f1": float(np.mean([row["f1"] for row in rows])) if rows else None,
            "balanced_accuracy": float(np.mean([row["recall"] for row in supported])) if supported else None,
        }
    return {"grouping": grouping, "groups": summaries}
