"""ID-based selector contracts. No oracle or unrevealed targets enter these APIs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from csda_al.data.metadata import CLASSES


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def unique_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


def probability_array(values: Any, ndim: int = 2) -> np.ndarray:
    p = np.array(values, dtype=np.float64, copy=True)
    if p.ndim != ndim or p.shape[-1] != len(CLASSES):
        raise ValueError(f"probabilities must have {ndim} dimensions and eight classes")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and lie in [0,1]")
    if not np.allclose(p.sum(axis=-1), 1, rtol=0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    p.setflags(write=False)
    return p


@dataclass(frozen=True)
class PredictionBatch:
    image_ids: tuple[str, ...]
    probabilities: np.ndarray
    model_checkpoint_id: str = ""
    pool_version: int = 0
    class_order: tuple[str, ...] = CLASSES

    def __post_init__(self) -> None:
        ids = unique_ids(self.image_ids, "prediction IDs")
        p = probability_array(self.probabilities)
        if len(ids) != len(p):
            raise ValueError("prediction IDs and probabilities do not align")
        if tuple(self.class_order) != tuple(CLASSES):
            raise ValueError("prediction class order differs from ISIC class order")
        integer(self.pool_version, "pool_version")
        object.__setattr__(self, "image_ids", ids)
        object.__setattr__(self, "probabilities", p)
        object.__setattr__(self, "class_order", tuple(CLASSES))

    def rows(self, image_ids: Sequence[str]) -> np.ndarray:
        index = {x: i for i, x in enumerate(self.image_ids)}
        try:
            return self.probabilities[[index[x] for x in image_ids]]
        except KeyError as exc:
            raise ValueError(f"missing prediction ID: {exc.args[0]}") from exc


@dataclass(frozen=True)
class QueryContext:
    feature_store: Any
    labeled_ids: tuple[str, ...] = ()
    unlabeled_ids: tuple[str, ...] | None = None
    batch_budget: int = 8
    seed: int = 42
    pool_version: int = 0
    predictions: PredictionBatch | None = None
    difficulty: Any = None
    mc_probabilities: np.ndarray | None = None
    labeled_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        universe = unique_ids(self.feature_store.image_ids, "feature IDs")
        labeled = unique_ids(self.labeled_ids, "labeled IDs")
        unlabeled = unique_ids(
            self.unlabeled_ids if self.unlabeled_ids is not None
            else tuple(x for x in universe if x not in set(labeled)), "unlabeled IDs"
        )
        if set(labeled) & set(unlabeled) or set(labeled) | set(unlabeled) != set(universe):
            raise ValueError("labeled/unlabeled IDs must partition the feature TRAIN universe")
        integer(self.batch_budget, "batch_budget")
        integer(self.seed, "seed")
        integer(self.pool_version, "pool_version")
        object.__setattr__(self, "labeled_ids", labeled)
        object.__setattr__(self, "unlabeled_ids", unlabeled)
        if self.predictions is not None:
            if set(self.predictions.image_ids) != set(unlabeled):
                raise ValueError("prediction IDs must equal current unlabeled IDs")
            if self.predictions.pool_version != self.pool_version:
                raise ValueError("stale prediction pool version")
        if self.difficulty is not None and self.predictions is not None:
            difficulty_model = self.difficulty.model_checkpoint_id
            if difficulty_model != self.predictions.model_checkpoint_id:
                raise ValueError("difficulty and predictions use different checkpoints")
        if self.mc_probabilities is not None:
            mc = probability_array(self.mc_probabilities, ndim=3)
            if mc.shape[0] < 2 or mc.shape[1] != len(unlabeled):
                raise ValueError("MC predictions must have shape [T>=2, unlabeled_count, 8]")
            object.__setattr__(self, "mc_probabilities", mc)
        if self.labeled_counts is not None:
            counts = tuple(int(x) for x in self.labeled_counts)
            if len(counts) != len(CLASSES) or any(x < 0 for x in counts):
                raise ValueError("labeled_counts must have eight nonnegative class counts")
            object.__setattr__(self, "labeled_counts", counts)

    @property
    def effective_budget(self) -> int:
        return min(self.batch_budget, len(self.candidate_ids))

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Validated candidate IDs, with constructor optionality resolved."""
        if self.unlabeled_ids is None:
            raise RuntimeError("QueryContext must be initialized before selecting candidates")
        return self.unlabeled_ids

    @property
    def train_pool_ids(self) -> tuple[str, ...]:
        return tuple(self.feature_store.image_ids)


@dataclass(frozen=True)
class QueryBatch:
    image_ids: tuple[str, ...]
    scores: tuple[float | None, ...] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    method: str = "unknown"
    version: str = "1.0"
    config_fingerprint: str = ""
    pool_version: int = 0

    def __post_init__(self) -> None:
        ids = unique_ids(self.image_ids, "query IDs")
        object.__setattr__(self, "image_ids", ids)
        if self.scores is not None:
            scores = tuple(None if x is None else float(x) for x in self.scores)
            if len(scores) != len(ids) or any(x is not None and not np.isfinite(x) for x in scores):
                raise ValueError("query scores must be aligned and finite or None")
            object.__setattr__(self, "scores", scores)
        integer(self.pool_version, "pool_version")
        import copy
        import hashlib
        import json
        details = copy.deepcopy(self.diagnostics)
        json.dumps(details, allow_nan=False)
        object.__setattr__(self, "diagnostics", details)
        if not self.config_fingerprint:
            payload = {"method": self.method, "version": self.version,
                       "configuration": details.get("configuration", {})}
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            object.__setattr__(self, "config_fingerprint", fingerprint)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class AcquisitionStrategy(Protocol):
    def select(self, context: QueryContext) -> QueryBatch: ...


def prediction_rows(context: QueryContext) -> np.ndarray:
    if context.predictions is None:
        raise ValueError("this acquisition strategy requires model probabilities")
    return context.predictions.rows(context.candidate_ids)


def ranked_batch(context: QueryContext, scores: np.ndarray, method: str,
                 diagnostics: dict[str, Any] | None = None) -> QueryBatch:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(context.candidate_ids),) or not np.isfinite(scores).all():
        raise ValueError("acquisition scores must be a finite aligned vector")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], context.candidate_ids[i]))
    chosen = order[:context.effective_budget]
    return QueryBatch(tuple(context.candidate_ids[i] for i in chosen),
                      tuple(float(scores[i]) for i in chosen), diagnostics or {}, method=method,
                      pool_version=context.pool_version)
