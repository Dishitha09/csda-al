"""Class-Balanced Active Learning for Image Classification (Bengar et al., WACV 2022).

Balances acquired class distributions under severe class imbalance by allocating
acquisition quotas or penalizing uncertainty with predicted pseudo-labels.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from csda_al.data.metadata import CLASSES
from .base import QueryBatch, QueryContext, prediction_rows, ranked_batch
from .entropy import entropy_scores


class BengarClassBalancedAcquisition:
    """Bengar et al. (WACV 2022) Class-Balanced Active Learning.

    Supports:
    - "constrained" (default): Computes per-class acquisition quotas proportional
      to class deficits from a uniform target, and fills each quota from top-entropy
      candidates within the predicted class.
    - "penalized": Scores each candidate by H(x) * (1 / (N_c + 1))^beta where N_c
      is the acquired count of the predicted class.
    """

    method = "bengar_class_balanced"

    def __init__(
        self,
        *,
        mode: str = "constrained",
        beta: float = 1.0,
        fallback_to_entropy: bool = True,
    ):
        if mode not in {"constrained", "penalized"}:
            raise ValueError("mode must be 'constrained' or 'penalized'")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be nonnegative")
        self.mode = mode
        self.beta = float(beta)
        self.fallback_to_entropy = fallback_to_entropy

    def select(self, context: QueryContext) -> QueryBatch:
        details: dict[str, Any] = {
            "paper": "Bengar et al. (WACV 2022)",
            "configuration": {
                "mode": self.mode,
                "beta": self.beta,
                "fallback_to_entropy": self.fallback_to_entropy,
            },
        }
        if not context.effective_budget:
            return QueryBatch((), (), details, method=self.method, pool_version=context.pool_version)

        p = prediction_rows(context)  # [N_unlabeled, 8]
        entropies = entropy_scores(p)  # [N_unlabeled]
        pseudo_labels = p.argmax(axis=1)  # [N_unlabeled]
        num_classes = len(CLASSES)

        # Determine acquired class counts
        if context.labeled_counts is not None:
            counts = np.array(context.labeled_counts, dtype=np.int64)
        else:
            counts = np.zeros(num_classes, dtype=np.int64)

        details["prior_labeled_counts"] = counts.tolist()
        budget = context.effective_budget

        if self.mode == "penalized":
            # Penalized uncertainty: S(x_i) = H(x_i) * (1 / (counts[hat_y_i] + 1))^beta
            penalties = 1.0 / np.power(counts[pseudo_labels] + 1.0, self.beta)
            scores = entropies * penalties
            details["mode"] = "penalized"
            return ranked_batch(context, scores, self.method, details)

        # Constrained quota mode (primary Bengar formulation):
        # Target uniform count per class: T = (|L| + B) / C
        total_samples = counts.sum() + budget
        target_per_class = total_samples / num_classes
        deficits = np.maximum(0.0, target_per_class - counts)
        sum_deficits = deficits.sum()

        if sum_deficits > 1e-9:
            quotas_float = budget * (deficits / sum_deficits)
        else:
            quotas_float = np.full(num_classes, budget / num_classes)

        # Integer quota rounding preserving sum == budget
        quotas = np.floor(quotas_float).astype(np.int64)
        remainder = budget - int(quotas.sum())
        if remainder > 0:
            fractional = quotas_float - quotas
            priority = np.argsort(-fractional)
            for idx in priority[:remainder]:
                quotas[idx] += 1

        details["quotas"] = quotas.tolist()
        details["deficits"] = deficits.tolist()

        # Partition candidates by predicted class and fill per-class quotas
        selected_indices: list[int] = []
        underflow_count = 0

        for c in range(num_classes):
            c_indices = [i for i in range(len(context.candidate_ids)) if pseudo_labels[i] == c]
            # Sort by entropy descending with stable tie-break
            c_indices.sort(key=lambda i: (-entropies[i], context.candidate_ids[i]))
            q = int(quotas[c])
            if len(c_indices) >= q:
                selected_indices.extend(c_indices[:q])
            else:
                selected_indices.extend(c_indices)
                underflow_count += q - len(c_indices)

        # If some classes had fewer candidates than their quota (underflow),
        # fill remaining slots with highest entropy candidates across remaining unselected pool
        if len(selected_indices) < budget:
            selected_set = set(selected_indices)
            remaining = [i for i in range(len(context.candidate_ids)) if i not in selected_set]
            remaining.sort(key=lambda i: (-entropies[i], context.candidate_ids[i]))
            deficit_needed = budget - len(selected_indices)
            selected_indices.extend(remaining[:deficit_needed])

        details["underflow_count"] = underflow_count
        chosen_ids = tuple(context.candidate_ids[i] for i in selected_indices)
        chosen_scores = tuple(float(entropies[i]) for i in selected_indices)

        return QueryBatch(
            chosen_ids,
            chosen_scores,
            details,
            method=self.method,
            pool_version=context.pool_version,
        )
