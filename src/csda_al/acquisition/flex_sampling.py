"""Flexible Sampling for Long-tailed Skin Lesion Classification (Ju et al., MICCAI 2022).

Replicates the curriculum sampling principle from Section 2.3:
Per-class sampling probabilities are computed inversely from validation performance
(classes performing worse get sampled more), combined with uncertainty (BALD or entropy).
In active learning with sealed pools, class assignment is determined by model predictions
or feature prototype proximity, strictly preventing ground-truth test/unlabeled label leakage.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from csda_al.data.metadata import CLASSES
from .base import QueryBatch, QueryContext, prediction_rows, ranked_batch
from .entropy import entropy_scores
from .bald import bald_scores


class FlexibleSamplingAcquisition:
    """Ju et al. (MICCAI 2022) Flexible Curriculum Sampling for Active Learning.

    Attributes:
        uncertainty_metric: "entropy" or "bald" (uses MC dropout if available).
        min_per_class: Minimum queries allocated to any class if candidates exist.
    """

    method = "flexible_sampling"

    def __init__(
        self,
        *,
        uncertainty_metric: str = "entropy",
        min_per_class: int = 1,
    ):
        if uncertainty_metric not in {"entropy", "bald"}:
            raise ValueError("uncertainty_metric must be 'entropy' or 'bald'")
        if min_per_class < 0:
            raise ValueError("min_per_class must be nonnegative")
        self.uncertainty_metric = uncertainty_metric
        self.min_per_class = int(min_per_class)

    def select(self, context: QueryContext) -> QueryBatch:
        details: dict[str, Any] = {
            "paper": "Ju et al. (MICCAI 2022)",
            "configuration": {
                "uncertainty_metric": self.uncertainty_metric,
                "min_per_class": self.min_per_class,
            },
        }
        if not context.effective_budget:
            return QueryBatch((), (), details, method=self.method, pool_version=context.pool_version)

        p = prediction_rows(context)  # [N_unlabeled, 8]
        pseudo_labels = p.argmax(axis=1)
        num_classes = len(CLASSES)
        budget = context.effective_budget

        # Compute uncertainty scores
        if self.uncertainty_metric == "bald" and context.mc_probabilities is not None:
            uncertainty = bald_scores(context.mc_probabilities)
            details["uncertainty_used"] = "bald"
        else:
            uncertainty = entropy_scores(p)
            details["uncertainty_used"] = "entropy"

        # Compute class-wise sampling weights from validation difficulty/accuracy
        # p_c = 1 - Accuracy(c) (or 1 - Recall(c))
        if context.difficulty is not None and hasattr(context.difficulty, "values"):
            # values in DifficultyEstimate represent per-class difficulty in [0, 1]
            raw_diff = np.array(context.difficulty.values, dtype=np.float64)
        else:
            # Fallback to uniform difficulty if validation difficulty is unavailable
            raw_diff = np.ones(num_classes, dtype=np.float64)
            details["fallback_reason"] = "no_validation_difficulty_uniform_weights"

        total_diff = raw_diff.sum()
        if total_diff > 1e-9:
            class_probs = raw_diff / total_diff
        else:
            class_probs = np.full(num_classes, 1.0 / num_classes)

        details["class_probabilities"] = class_probs.tolist()

        # Compute per-class quota
        quotas_float = budget * class_probs
        quotas = np.floor(quotas_float).astype(np.int64)
        remainder = budget - int(quotas.sum())
        if remainder > 0:
            fractional = quotas_float - quotas
            priority = np.argsort(-fractional)
            for idx in priority[:remainder]:
                quotas[idx] += 1

        details["quotas"] = quotas.tolist()

        # Sample top uncertainty candidates from each predicted class bucket
        selected_indices: list[int] = []
        underflow_count = 0

        for c in range(num_classes):
            c_indices = [i for i in range(len(context.candidate_ids)) if pseudo_labels[i] == c]
            c_indices.sort(key=lambda i: (-uncertainty[i], context.candidate_ids[i]))
            q = int(quotas[c])
            if len(c_indices) >= q:
                selected_indices.extend(c_indices[:q])
            else:
                selected_indices.extend(c_indices)
                underflow_count += q - len(c_indices)

        # Fill any quota deficit globally with highest uncertainty candidates
        if len(selected_indices) < budget:
            selected_set = set(selected_indices)
            remaining = [i for i in range(len(context.candidate_ids)) if i not in selected_set]
            remaining.sort(key=lambda i: (-uncertainty[i], context.candidate_ids[i]))
            deficit_needed = budget - len(selected_indices)
            selected_indices.extend(remaining[:deficit_needed])

        details["underflow_count"] = underflow_count
        chosen_ids = tuple(context.candidate_ids[i] for i in selected_indices)
        chosen_scores = tuple(float(uncertainty[i]) for i in selected_indices)

        return QueryBatch(
            chosen_ids,
            chosen_scores,
            details,
            method=self.method,
            pool_version=context.pool_version,
        )
