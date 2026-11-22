"""Greedy farthest-first coverage with O(ND) working memory."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from .base import QueryBatch, QueryContext


def farthest_first(store, candidate_ids: Sequence[str], budget: int,
                   seed_ids: Sequence[str] = ()) -> tuple[tuple[str, ...], tuple[float, ...]]:
    ids = tuple(sorted(candidate_ids))
    if not budget or not ids:
        return (), ()
    x = np.asarray(store.rows(ids), dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    minimum = np.full(len(ids), np.inf)
    for center in store.rows(tuple(seed_ids)) if seed_ids else ():
        minimum = np.minimum(minimum, np.linalg.norm(x-center, axis=1))
    selected: list[int] = []
    scores: list[float] = []
    if not seed_ids:
        # A declared representativeness seed for an otherwise undefined empty cover.
        first = int(np.argmin(np.linalg.norm(x-x.mean(axis=0), axis=1)))
        selected.append(first)
        scores.append(0.0)
        minimum = np.linalg.norm(x-x[first], axis=1)
        minimum[first] = -np.inf
    while len(selected) < min(budget, len(ids)):
        i = int(np.argmax(minimum))
        scores.append(float(minimum[i]))
        selected.append(i)
        minimum = np.minimum(minimum, np.linalg.norm(x-x[i], axis=1))
        minimum[selected] = -np.inf
    return tuple(ids[i] for i in selected), tuple(scores)


class CoreSetAcquisition:
    def select(self, context: QueryContext) -> QueryBatch:
        ids, scores = farthest_first(context.feature_store, context.candidate_ids,
                                     context.effective_budget, context.labeled_ids)
        return QueryBatch(ids, scores, {"empty_seed_policy": "nearest_global_centroid",
                                       "distance": "euclidean"}, method="coreset",
                          pool_version=context.pool_version)
