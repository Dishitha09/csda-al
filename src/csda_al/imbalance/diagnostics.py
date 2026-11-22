"""Retrospective diagnostics, run only after annotations are committed."""
import numpy as np
from .class_difficulty import _labels
from csda_al.data.metadata import CLASSES


def post_annotation_composition(acquired_labels) -> dict:
    y = _labels(acquired_labels, "acquired labels")
    counts = np.bincount(y, minlength=8)
    return {"phase": "post_annotation_only", "class_order": list(CLASSES),
            "counts": counts.tolist(), "total": len(y)}


def prediction_composition(probabilities) -> dict:
    from csda_al.acquisition.base import probability_array
    p = probability_array(probabilities)
    return {"predicted_counts": np.bincount(p.argmax(axis=1), minlength=8).tolist(),
            "expected_membership": p.sum(axis=0).tolist(), "class_order": list(CLASSES)}
