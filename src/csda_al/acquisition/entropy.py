"""Normalized predictive entropy."""
import numpy as np
from .base import QueryBatch, QueryContext, prediction_rows, probability_array, ranked_batch


def entropy_scores(probabilities: np.ndarray) -> np.ndarray:
    p = probability_array(probabilities)
    return -(p * np.log(np.clip(p, np.finfo(float).tiny, 1))).sum(axis=1) / np.log(p.shape[1])


class EntropyAcquisition:
    def select(self, context: QueryContext) -> QueryBatch:
        if not context.effective_budget:
            return QueryBatch((), method="entropy", pool_version=context.pool_version)
        return ranked_batch(context, entropy_scores(prediction_rows(context)), "entropy")
