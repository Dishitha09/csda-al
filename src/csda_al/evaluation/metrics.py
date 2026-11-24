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
    y_prob: Sequence[Sequence[float]] | np.ndarray | None = None,
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
    if y_prob is not None:
        auroc_res = compute_macro_auroc(truth, y_prob, count)
        result["macro_auroc"] = auroc_res["macro_auroc"]
        result["per_class_auroc"] = auroc_res["per_class_auroc"]
    else:
        result["macro_auroc"] = None
        result["per_class_auroc"] = [None] * count
    return result


def compute_macro_auroc(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[Sequence[float]] | np.ndarray,
    num_classes: int = 8,
) -> dict[str, Any]:
    """Compute one-vs-rest Macro-AUROC across supported classes.

    For classes with zero true positives or zero true negatives, AUROC is
    undefined (None). Macro-AUROC averages across classes where AUROC is defined.
    """
    from sklearn.metrics import roc_auc_score

    count = _class_count(num_classes)
    truth = _labels(y_true, count, "y_true")
    if truth.size == 0:
        return {"macro_auroc": None, "per_class_auroc": [None] * count, "evaluated_classes": 0}
    probs = np.asarray(y_prob, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != count or len(probs) != len(truth):
        raise ValueError(f"y_prob must have shape (N, {count}) matching length of y_true")

    per_class: list[float | None] = []
    valid: list[float] = []

    for c in range(count):
        binary_truth = (truth == c).astype(np.int64)
        positives = int(binary_truth.sum())
        negatives = len(binary_truth) - positives
        if positives > 0 and negatives > 0:
            try:
                val = float(roc_auc_score(binary_truth, probs[:, c]))
                per_class.append(val)
                valid.append(val)
            except Exception:
                per_class.append(None)
        else:
            per_class.append(None)

    return {
        "macro_auroc": float(np.mean(valid)) if valid else None,
        "per_class_auroc": per_class,
        "evaluated_classes": len(valid),
    }


def evaluate_complete_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[Sequence[float]] | np.ndarray | None = None,
    num_classes: int = 8,
) -> dict[str, Any]:
    """Comprehensive evaluation returning aggregate, per-class, minority recall, and confusion matrix.
    
    Guarantees:
    - Balanced Accuracy strictly equals the arithmetic mean of all 8 class recalls.
    - Macro-F1 strictly equals the arithmetic mean of all 8 class F1 scores.
    - Minority Recall (DF/VASC/SCC) is strictly (DF_rec + VASC_rec + SCC_rec) / 3.
    """
    raw = classification_metrics(y_true, y_pred, num_classes=num_classes, y_prob=y_prob)
    names = _names(num_classes)
    
    recalls = {row["class_name"]: row["recall"] for row in raw["per_class"]}
    precisions = {row["class_name"]: row["precision"] for row in raw["per_class"]}
    f1s = {row["class_name"]: row["f1"] for row in raw["per_class"]}
    
    # Mathematical identity verification
    expected_ba = float(np.mean([row["recall"] for row in raw["per_class"]]))
    expected_macro_f1 = float(np.mean([row["f1"] for row in raw["per_class"]]))
    
    if not np.isclose(raw["balanced_accuracy"], expected_ba, atol=1e-5):
        raise ValueError(f"Balanced Accuracy ({raw['balanced_accuracy']}) != arithmetic mean recall ({expected_ba})")
    if not np.isclose(raw["macro_f1"], expected_macro_f1, atol=1e-5):
        raise ValueError(f"Macro-F1 ({raw['macro_f1']}) != arithmetic mean F1 ({expected_macro_f1})")
        
    # Standardized Minority Recall (DF/VASC/SCC)
    df_rec = recalls.get("DF", 0.0)
    vasc_rec = recalls.get("VASC", 0.0)
    scc_rec = recalls.get("SCC", 0.0)
    minority_rec_3 = float((df_rec + vasc_rec + scc_rec) / 3.0)
    
    ak_rec = recalls.get("AK", 0.0)
    bkl_rec = recalls.get("BKL", 0.0)
    minority_rec_5 = float((df_rec + vasc_rec + scc_rec + ak_rec + bkl_rec) / 5.0)

    return {
        "balanced_accuracy": raw["balanced_accuracy"],
        "macro_f1": raw["macro_f1"],
        "macro_auroc": raw["macro_auroc"],
        "accuracy": raw["accuracy"],
        "minority_recall_3": minority_rec_3,
        "minority_recall_5": minority_rec_5,
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
        "confusion_matrix": raw["confusion_matrix"],
        "raw_metrics": raw,
    }


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
