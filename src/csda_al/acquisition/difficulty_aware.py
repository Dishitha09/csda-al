"""Experimental CSDA-AL adaptation of Flexible Sampling's curriculum principle.

Unlabeled class membership is predicted, never supplied by a truth-label bucket.
The fused score is our proposed integration, not a paper reproduction claim.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from csda_al.imbalance.probability_weighting import expected_difficulty, normalize_difficulty
from .base import QueryBatch, QueryContext, integer, prediction_rows, ranked_batch
from .entropy import entropy_scores


class DifficultyAwareAcquisition:
    method = "difficulty_aware"
    def __init__(self, *, alpha: float = 0.5, normalization: str = "minmax",
                 diversity_weight: float = 0.0, shortlist_factor: int = 4,
                 missing_validation: str = "entropy"):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0,1]")
        if normalization not in {"minmax", "identity"}:
            raise ValueError("normalization must be minmax or identity")
        if not np.isfinite(diversity_weight) or not 0 <= diversity_weight <= 1:
            raise ValueError("diversity_weight must lie in [0,1]")
        integer(shortlist_factor, "shortlist_factor", 1)
        if missing_validation not in {"entropy", "raise"}:
            raise ValueError("missing_validation must be entropy or raise")
        self.alpha = float(alpha)
        self.normalization = normalization
        self.diversity_weight = float(diversity_weight)
        self.shortlist_factor = int(shortlist_factor)
        self.missing_validation = missing_validation

    def select(self, context: QueryContext) -> QueryBatch:
        config = {"alpha": self.alpha, "normalization": self.normalization,
                  "diversity_weight": self.diversity_weight,
                  "shortlist_factor": self.shortlist_factor,
                  "missing_validation": self.missing_validation}
        details: dict[str, Any] = {"experimental_adaptation": True, "configuration": config}
        if not context.effective_budget:
            return QueryBatch((), (), details, method=self.method, pool_version=context.pool_version)
        p = prediction_rows(context)
        uncertainty = entropy_scores(p)
        if context.difficulty is None:
            if self.missing_validation == "raise":
                raise ValueError("difficulty acquisition requires paid validation difficulty")
            details["fallback_reason"] = "no_validation_difficulty_use_entropy"
            return ranked_batch(context, uncertainty, self.method, details)
        membership = expected_difficulty(p, context.difficulty)
        weighted, normalization_details = normalize_difficulty(membership, method=self.normalization)
        scores = self.alpha * uncertainty + (1-self.alpha) * weighted
        details.update(normalization_details)
        details["validation_support"] = context.difficulty.validation_support.tolist()
        details["prior_classes"] = np.flatnonzero(context.difficulty.prior_mask).tolist()
        details["validation_annotation_cost"] = context.difficulty.validation_annotation_cost
        if not self.diversity_weight or context.effective_budget == 1:
            return ranked_batch(context, scores, self.method, details)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], context.candidate_ids[i]))
        shortlist = order[:min(len(order), self.shortlist_factor*context.effective_budget)]
        ids = tuple(context.candidate_ids[i] for i in shortlist)
        x = np.asarray(context.feature_store.rows(ids), dtype=np.float64)
        minimum = np.full(len(ids), np.inf)
        if context.labeled_ids:
            for center in context.feature_store.rows(context.labeled_ids):
                minimum = np.minimum(minimum, np.linalg.norm(x-center, axis=1))
        selected: list[int] = []
        while len(selected) < context.effective_budget:
            available = [i for i in range(len(ids)) if i not in selected]
            if np.isinf(minimum).all():
                chosen = min(available, key=lambda i: (-scores[shortlist[i]], ids[i]))
            else:
                dist = minimum[available]
                span = float(dist.max()-dist.min())
                norm = (dist-dist.min())/span if span > 1e-12 else np.zeros_like(dist)
                combined = ((1-self.diversity_weight)*scores[np.array(shortlist)[available]]
                            + self.diversity_weight*norm)
                chosen = min(range(len(available)), key=lambda i: (-combined[i], ids[available[i]]))
                chosen = available[chosen]
            selected.append(chosen)
            minimum = np.minimum(minimum, np.linalg.norm(x-x[chosen], axis=1))
        details["diversity_shortlist_size"] = len(shortlist)
        return QueryBatch(tuple(ids[i] for i in selected),
                          tuple(float(scores[shortlist[i]]) for i in selected), details,
                          method=self.method, pool_version=context.pool_version)


class DifficultyOnlyAcquisition(DifficultyAwareAcquisition):
    method = "difficulty_only"
    def __init__(self, **kwargs):
        super().__init__(alpha=0.0, **kwargs)


EntropyDifficultyAcquisition = DifficultyAwareAcquisition
