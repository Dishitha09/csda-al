"""Combined CSDA-AL Acquisition Strategy (C4).

Unifies:
1. Predictive uncertainty (Shannon Entropy or BALD).
2. Expected candidate difficulty derived from validation recall trajectories.
3. Bengar et al. (WACV 2022) constrained class-balancing quotas.
4. Robust underflow redistribution ensuring exact budget fulfillment without label leakage.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from csda_al.data.metadata import CLASSES
from csda_al.imbalance.probability_weighting import expected_difficulty, normalize_difficulty
from .base import (
    AcquisitionStrategy,
    QueryBatch,
    QueryContext,
    integer,
    prediction_rows,
)
from .entropy import entropy_scores
from .bald import bald_scores


class CombinedCSDAAcquisition:
    """CSDA-AL Integrated Acquisition Strategy (C4).

    Fuses dynamic difficulty-weighted uncertainty with Bengar class-balancing quota constraints:
    - Instance score: S(x_i) = alpha * H_norm(x_i) + (1 - alpha) * Diff_norm(x_i).
    - Class quotas: Allocated proportionally to class deficits d_c = max(0, T - N_c) from uniform target T.
    - Selection: Top S(x_i) candidates from each predicted pseudo-class bucket.
    - Underflow: Remaining deficit slots are filled greedily by S(x_i) across the entire unselected pool.
    """

    method = "csda_al_combined"

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        normalization: str = "minmax",
        uncertainty_metric: str = "entropy",
        diversity_weight: float = 0.0,
        shortlist_factor: int = 4,
        missing_validation: str = "entropy",
    ):
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if normalization not in {"minmax", "identity"}:
            raise ValueError("normalization must be minmax or identity")
        if uncertainty_metric not in {"entropy", "bald"}:
            raise ValueError("uncertainty_metric must be entropy or bald")
        if not np.isfinite(diversity_weight) or not 0.0 <= diversity_weight <= 1.0:
            raise ValueError("diversity_weight must lie in [0, 1]")
        integer(shortlist_factor, "shortlist_factor", 1)
        if missing_validation not in {"entropy", "raise"}:
            raise ValueError("missing_validation must be entropy or raise")

        self.alpha = float(alpha)
        self.normalization = normalization
        self.uncertainty_metric = uncertainty_metric
        self.diversity_weight = float(diversity_weight)
        self.shortlist_factor = int(shortlist_factor)
        self.missing_validation = missing_validation

    def select(self, context: QueryContext) -> QueryBatch:
        config = {
            "alpha": self.alpha,
            "normalization": self.normalization,
            "uncertainty_metric": self.uncertainty_metric,
            "diversity_weight": self.diversity_weight,
            "shortlist_factor": self.shortlist_factor,
            "missing_validation": self.missing_validation,
        }
        details: dict[str, Any] = {
            "strategy": "CSDA-AL Unified (Difficulty-Aware + Bengar Class-Balanced)",
            "configuration": config,
        }

        budget = context.effective_budget
        if not budget:
            return QueryBatch((), (), details, method=self.method, pool_version=context.pool_version)

        p = prediction_rows(context)  # [N_unlabeled, 8]
        num_candidates = len(context.candidate_ids)
        num_classes = len(CLASSES)
        pseudo_labels = p.argmax(axis=1)

        # 1. Compute uncertainty
        if self.uncertainty_metric == "bald" and context.mc_probabilities is not None:
            raw_uncertainty = bald_scores(context.mc_probabilities)
            details["uncertainty_used"] = "bald"
        else:
            raw_uncertainty = entropy_scores(p)
            details["uncertainty_used"] = "entropy"

        # Normalize uncertainty to [0, 1]
        u_span = float(raw_uncertainty.max() - raw_uncertainty.min())
        norm_uncertainty = (
            (raw_uncertainty - raw_uncertainty.min()) / u_span
            if u_span > 1e-12
            else np.zeros_like(raw_uncertainty)
        )

        # 2. Compute difficulty-aware instance score
        if context.difficulty is not None:
            raw_difficulty = expected_difficulty(p, context.difficulty)
            norm_difficulty, norm_meta = normalize_difficulty(raw_difficulty, method=self.normalization)
            details.update(norm_meta)
            instance_scores = self.alpha * norm_uncertainty + (1.0 - self.alpha) * norm_difficulty
        else:
            if self.missing_validation == "raise":
                raise ValueError("CSDA-AL combined acquisition requires validation difficulty")
            details["fallback_reason"] = "no_validation_difficulty_use_pure_uncertainty"
            instance_scores = norm_uncertainty

        # 3. Compute Bengar-style class-balancing quotas
        if context.labeled_counts is not None:
            counts = np.array(context.labeled_counts, dtype=np.int64)
        else:
            counts = np.zeros(num_classes, dtype=np.int64)

        details["prior_labeled_counts"] = counts.tolist()

        total_samples = counts.sum() + budget
        target_per_class = total_samples / num_classes
        deficits = np.maximum(0.0, target_per_class - counts)
        sum_deficits = deficits.sum()

        if sum_deficits > 1e-9:
            quotas_float = budget * (deficits / sum_deficits)
        else:
            quotas_float = np.full(num_classes, budget / num_classes)

        # Exact integer quota allocation
        quotas = np.floor(quotas_float).astype(np.int64)
        remainder = budget - int(quotas.sum())
        if remainder > 0:
            fractional = quotas_float - quotas
            priority = np.argsort(-fractional)
            for idx in priority[:remainder]:
                quotas[idx] += 1

        details["quotas"] = quotas.tolist()
        details["deficits"] = deficits.tolist()

        # 4. Partition candidates by predicted class and select top instance_scores
        selected_indices: list[int] = []
        underflow_count = 0

        for c in range(num_classes):
            c_indices = [i for i in range(num_candidates) if pseudo_labels[i] == c]
            # Sort by combined instance score descending with stable ID tie-break
            c_indices.sort(key=lambda i: (-instance_scores[i], context.candidate_ids[i]))
            q = int(quotas[c])
            if len(c_indices) >= q:
                selected_indices.extend(c_indices[:q])
            else:
                selected_indices.extend(c_indices)
                underflow_count += q - len(c_indices)

        # 5. Underflow redistribution: fill any quota deficit using overall top scores
        if len(selected_indices) < budget:
            selected_set = set(selected_indices)
            remaining = [i for i in range(num_candidates) if i not in selected_set]
            remaining.sort(key=lambda i: (-instance_scores[i], context.candidate_ids[i]))
            deficit_needed = budget - len(selected_indices)
            selected_indices.extend(remaining[:deficit_needed])

        details["underflow_count"] = underflow_count
        chosen_ids = tuple(context.candidate_ids[i] for i in selected_indices)
        chosen_scores = tuple(float(instance_scores[i]) for i in selected_indices)

        return QueryBatch(
            chosen_ids,
            chosen_scores,
            details,
            method=self.method,
            pool_version=context.pool_version,
        )
