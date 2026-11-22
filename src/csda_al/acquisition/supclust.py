"""Original SUPClust implementation of Ono et al. (2024), Equations 1-3.

Within-cluster kNN, ceil rounding, numerical floors and degenerate-pool fallback
are disclosed engineering conventions rather than fully specified paper settings.
"""
from __future__ import annotations

import warnings
from typing import Any
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .base import QueryBatch, QueryContext, integer
from .coreset import farthest_first


def typicality_scores(features: np.ndarray, k_neighbors: int = 10,
                      epsilon: float = 1e-12) -> np.ndarray:
    integer(k_neighbors, "k_neighbors", 1)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("typicality requires a nonempty finite feature matrix")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if len(x) == 1:
        return np.ones(1)  # Sentinel, not an estimated singleton density.
    k = min(k_neighbors, len(x)-1)
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(x)
    distances, indices = nn.kneighbors(x)
    means = np.array([distances[i][indices[i] != i][:k].mean() for i in range(len(x))])
    return 1 / np.maximum(means, epsilon)


def center_weights(centers: np.ndarray, cluster: int, temperature: float = 1.0) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim != 2 or len(c) < 2 or not c.shape[1] or not np.isfinite(c).all():
        raise ValueError("SUP requires at least two finite centers")
    integer(cluster, "cluster")
    if cluster >= len(c) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid cluster or finite positive temperature")
    logits = -np.linalg.norm(c-c[cluster], axis=1) / temperature
    logits[cluster] = -np.inf
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights[cluster] = 0
    return weights / weights.sum()


def sup_scores(features: np.ndarray, centers: np.ndarray, cluster: int,
               temperature: float = 1.0, epsilon: float = 1e-12) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim != 2 or not c.shape[1] or x.ndim != 2 or x.shape[1] != c.shape[1] or not np.isfinite(x).all():
        raise ValueError("SUP candidate features must align with centers and be finite")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    weights = center_weights(c, cluster, temperature)
    # Chunked pairwise distances avoid a candidates x centers x dimensions tensor.
    result = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), 256):
        distances = cdist(x[start:start+256], c, metric="euclidean")
        result[start:start+256] = 1 / np.maximum(distances @ weights, epsilon)
    return result


class SUPClustAcquisition:
    def __init__(self, *, temperature: float = 1.0, typical_fraction: float = 0.1,
                 k_neighbors: int = 10, n_init: int = 10, epsilon: float = 1e-12,
                 variant: str = "supclust", strict: bool = False):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(typical_fraction) or not 0 < typical_fraction <= 1:
            raise ValueError("typical_fraction must lie in (0,1]")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        integer(k_neighbors, "k_neighbors", 1)
        integer(n_init, "n_init", 1)
        if variant not in {"supclust", "without_typicality", "without_SUP", "typiclust", "kmeans"}:
            raise ValueError("unknown clustering acquisition variant")
        self.temperature = float(temperature)
        self.typical_fraction = float(typical_fraction)
        self.k_neighbors = int(k_neighbors)
        self.n_init = int(n_init)
        self.epsilon = float(epsilon)
        self.variant = variant
        self.strict = bool(strict)

    def select(self, context: QueryContext) -> QueryBatch:
        b = context.effective_budget
        configuration = {"temperature": self.temperature, "typical_fraction": self.typical_fraction,
                         "k_neighbors": self.k_neighbors, "n_init": self.n_init,
                         "epsilon": self.epsilon, "variant": self.variant, "strict": self.strict}
        details: dict[str, Any] = {"configuration": configuration, "requested_batch": context.batch_budget,
                   "effective_batch": b, "feature_encoder": str(context.feature_store.encoder_id),
                   "typicality_neighbors": "within_cluster_excluding_self",
                   "rounding": "max(1,ceil(fraction*cluster_size))", "tie_policy": "image_id",
                   "fallback_ids": [], "fallback_reasons": [], "candidates": []}
        if not b:
            return QueryBatch((), (), details, method=self.variant, pool_version=context.pool_version)
        ids = tuple(sorted(context.train_pool_ids))
        x = np.asarray(context.feature_store.rows(ids), dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != len(ids) or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError("clustering requires an aligned finite TRAIN feature matrix")
        requested = len(context.labeled_ids) + b
        if requested > len(ids):
            raise ValueError("requested clusters exceed TRAIN pool size")
        distinct = len(np.unique(x, axis=0))
        fit_k = min(requested, distinct)
        # The paper rule remains requested K=L+b; collapsing identical vectors is
        # a recorded exception to avoid empty centers and misleading SUP values.
        with warnings.catch_warnings(record=True) as caught:
            km = KMeans(n_clusters=fit_k, random_state=context.seed, n_init=self.n_init,
                        algorithm="lloyd").fit(x)
        active_labels = np.unique(km.labels_)
        reindex = {int(old): new for new, old in enumerate(active_labels)}
        assignments = np.array([reindex[int(v)] for v in km.labels_])
        centers = km.cluster_centers_[active_labels]
        labeled_set = set(context.labeled_ids)
        occupied = {int(assignments[i]) for i, image_id in enumerate(ids) if image_id in labeled_set}
        eligible = [i for i in range(len(centers)) if i not in occupied]
        eligible.sort(key=lambda cluster: (-int(np.sum(assignments == cluster)),
                                          min(ids[j] for j in np.flatnonzero(assignments == cluster))))
        details.update({"requested_clusters": requested, "effective_clusters": len(centers),
                        "distinct_vectors": distinct, "occupied_clusters": len(occupied),
                        "eligible_clusters": len(eligible),
                        "kmeans_warnings": [str(w.message) for w in caught]})
        requires_sup = self.variant in {"supclust", "without_typicality"}
        no_boundary = requires_sup and len(centers) < 2
        if self.strict and (no_boundary or len(eligible) < b):
            raise ValueError("SUPClust cannot fill paper-style batch: degenerate centers/eligible clusters")
        selected: list[str] = []
        scores: list[float | None] = []
        rng = np.random.default_rng(context.seed)
        for cluster in ([] if no_boundary else eligible[:b]):
            members = np.flatnonzero(assignments == cluster)
            f = x[members]
            typ = typicality_scores(f, self.k_neighbors, self.epsilon)
            if self.variant == "kmeans":
                local_scores = -np.linalg.norm(f-centers[cluster], axis=1)
                local = int(np.argmax(local_scores))
                score = float(local_scores[local])
            elif self.variant == "typiclust":
                local = int(np.argmax(typ))
                score = float(typ[local])
            else:
                count = max(1, int(np.ceil(self.typical_fraction * len(members))))
                candidates = np.argsort(-typ, kind="stable")[:count]
                if self.variant == "without_typicality":
                    candidates = np.arange(len(members))
                if self.variant == "without_SUP":
                    local = int(rng.choice(candidates))
                    score = None
                else:
                    boundary = sup_scores(f[candidates], centers, cluster,
                                          self.temperature, self.epsilon)
                    # Equal boundary scores resolve by canonical member ID, not typicality rank.
                    best = min(range(len(candidates)),
                               key=lambda i: (-boundary[i], ids[members[candidates[i]]]))
                    local = int(candidates[best])
                    score = float(boundary[best])
            chosen_id = ids[members[local]]
            selected.append(chosen_id)
            scores.append(score)
            details["candidates"].append({"image_id": chosen_id, "cluster": int(cluster),
                                           "cluster_size": len(members),
                                           "typicality": float(typ[local]),
                                           "singleton": len(members) == 1})
        if len(selected) < b:
            remaining = tuple(i for i in context.candidate_ids if i not in set(selected))
            fallback, _ = farthest_first(context.feature_store, remaining, b-len(selected),
                                          tuple(context.labeled_ids)+tuple(selected))
            selected.extend(fallback)
            scores.extend([None]*len(fallback))
            details["fallback_ids"] = list(fallback)
            details["fallback_reasons"] = ["SUP_undefined_single_center" if no_boundary
                                            else "insufficient_eligible_clusters"] * len(fallback)
        return QueryBatch(tuple(selected), tuple(scores), details, method=self.variant,
                          pool_version=context.pool_version)


class KMeansRepresentativeAcquisition(SUPClustAcquisition):
    def __init__(self, **kwargs):
        super().__init__(variant="kmeans", **kwargs)
