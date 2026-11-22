"""Seeded uniform querying without target access."""
import numpy as np
from .base import QueryBatch, QueryContext


class RandomInitialAcquisition:
    def select(self, context: QueryContext) -> QueryBatch:
        ids = sorted(context.candidate_ids)
        order = np.random.default_rng(context.seed).permutation(len(ids))[:context.effective_budget]
        return QueryBatch(tuple(ids[i] for i in order), method="random",
                          diagnostics={"seed": context.seed}, pool_version=context.pool_version)


RandomAcquisition = RandomInitialAcquisition
