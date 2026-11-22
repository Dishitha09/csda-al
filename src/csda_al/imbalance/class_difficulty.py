"""Validation-only class difficulty, following Flexible Sampling's principle.

Smoothing and predicted-membership acquisition are explicitly CSDA-AL choices.
This module never accepts the sealed training oracle or test labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from csda_al.data.metadata import CLASSES


def _labels(values: Any, name: str, classes: int = 8) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim != 1 or (x.size and x.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if x.size and ((x < 0).any() or (x >= classes).any()):
        raise ValueError(f"{name} contains an invalid class index")
    return x.astype(np.int64, copy=True)


@dataclass(frozen=True)
class DifficultyEstimate:
    values: np.ndarray
    validation_support: np.ndarray
    raw_recall: np.ndarray
    prior_mask: np.ndarray
    model_checkpoint_id: str = ""
    validation_split_hash: str = ""
    validation_annotation_cost: int = 0
    smoothing_config: dict[str, Any] | None = None
    class_order: tuple[str, ...] = CLASSES

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        support = np.array(self.validation_support, copy=True)
        recall = np.array(self.raw_recall, dtype=np.float64, copy=True)
        prior = np.array(self.prior_mask, dtype=bool, copy=True)
        if any(a.shape != (8,) for a in (values, support, recall, prior)):
            raise ValueError("difficulty arrays require exactly eight aligned classes")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("difficulty values must be finite in [0,1]")
        if support.dtype.kind not in "iu" or (support < 0).any():
            raise ValueError("validation support must be a nonnegative integer vector")
        observed = support > 0
        if not np.isfinite(recall[observed]).all() or (recall[observed] < 0).any() or (recall[observed] > 1).any():
            raise ValueError("observed raw recalls must be finite in [0,1]")
        if not np.all(prior[~observed]):
            raise ValueError("unsupported classes must be flagged as prior-derived")
        recall[~observed] = np.nan
        if tuple(self.class_order) != tuple(CLASSES):
            raise ValueError("difficulty class order must match ISIC class order")
        cost = self.validation_annotation_cost
        if isinstance(cost, (bool, np.bool_)) or not isinstance(cost, (int, np.integer)) or cost < int(support.sum()):
            raise ValueError("validation cost must account for every available validation label")
        for name, value in [("values", values), ("validation_support", support),
                            ("raw_recall", recall), ("prior_mask", prior)]:
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "smoothing_config", dict(self.smoothing_config or {}))
        object.__setattr__(self, "validation_annotation_cost", int(cost))
        object.__setattr__(self, "class_order", tuple(CLASSES))

    @property
    def support(self) -> np.ndarray:
        return self.validation_support

    def to_dict(self) -> dict[str, Any]:
        return {"values": self.values.tolist(), "validation_support": self.validation_support.tolist(),
                "raw_recall": [float(x) if np.isfinite(x) else None for x in self.raw_recall],
                "prior_mask": self.prior_mask.tolist(), "class_order": list(self.class_order),
                "model_checkpoint_id": self.model_checkpoint_id,
                "validation_split_hash": self.validation_split_hash,
                "validation_annotation_cost": self.validation_annotation_cost,
                "smoothing_config": dict(self.smoothing_config or {})}


def compute_class_difficulty(y_true, y_pred, *, num_classes: int = 8,
                             smoothing: float = 1.0, recall_prior: float = 0.5,
                             missing_policy: str = "prior", validation_annotation_cost: int | None = None,
                             model_checkpoint_id: str = "", validation_split_hash: str = "",
                             split: str = "validation") -> DifficultyEstimate | None:
    if split not in {"validation", "val"}:
        raise ValueError("class difficulty may only use the paid validation split")
    if num_classes != 8:
        raise ValueError("CSDA-AL fixes eight diagnostic classes")
    if not np.isfinite(smoothing) or smoothing < 0 or not np.isfinite(recall_prior) or not 0 <= recall_prior <= 1:
        raise ValueError("smoothing must be >=0 and recall_prior must lie in [0,1]")
    if missing_policy not in {"prior", "raise"}:
        raise ValueError("missing_policy must be 'prior' or 'raise'")
    truth = _labels(y_true, "validation truth")
    predicted = _labels(y_pred, "validation predictions")
    if truth.shape != predicted.shape:
        raise ValueError("validation truth/predictions must align")
    if not len(truth):
        return None
    support = np.bincount(truth, minlength=8)
    tp = np.bincount(truth[truth == predicted], minlength=8)
    missing = support == 0
    if missing_policy == "raise" and missing.any():
        raise ValueError("validation classes have zero support")
    raw = np.divide(tp, support, out=np.full(8, np.nan), where=support > 0)
    recall = np.full(8, recall_prior)
    recall[~missing] = (tp[~missing]+smoothing*recall_prior)/(support[~missing]+smoothing)
    cost = len(truth) if validation_annotation_cost is None else validation_annotation_cost
    return DifficultyEstimate(1-recall, support, raw, missing, model_checkpoint_id,
                              validation_split_hash, cost,
                              {"smoothing": smoothing, "recall_prior": recall_prior,
                               "missing_policy": missing_policy})
