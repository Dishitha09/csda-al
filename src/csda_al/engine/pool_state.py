"""Immutable ID-based pool transitions."""
from dataclasses import dataclass, asdict
from typing import Sequence


@dataclass(frozen=True)
class PoolState:
    train_candidate_ids: tuple[str, ...]
    labeled_ids: tuple[str, ...] = ()
    selected_history: tuple[tuple[str, ...], ...] = ()
    query_round: int = 0
    state_version: int = 0
    split_fingerprint: str = ""

    def __post_init__(self) -> None:
        for name in ("query_round", "state_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("train_candidate_ids", "labeled_ids"):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if len(values) != len(set(values)) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError(f"{name} must contain unique nonempty IDs")
        object.__setattr__(self, "selected_history", tuple(tuple(r) for r in self.selected_history))
        if any(not batch for batch in self.selected_history):
            raise ValueError("Selected history cannot contain empty query batches")
        history = tuple(i for batch in self.selected_history for i in batch)
        if not set(self.labeled_ids).issubset(self.train_candidate_ids):
            raise ValueError("Labeled pool contains non-TRAIN IDs")
        if history != self.labeled_ids or len(history) != len(set(history)):
            raise ValueError("Selected history must equal unique acquired labels in order")
        if self.query_round != len(self.selected_history) or self.state_version < self.query_round:
            raise ValueError("Round/version does not agree with acquisition history")

    @property
    def unlabeled_ids(self) -> tuple[str, ...]:
        labeled = set(self.labeled_ids)
        return tuple(i for i in self.train_candidate_ids if i not in labeled)

    @property
    def total_unique_annotations(self) -> int:
        return len(self.labeled_ids)

    def acquire(self, selected: Sequence[str]) -> "PoolState":
        ids = tuple(selected)
        if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(self.unlabeled_ids):
            raise ValueError("Query must contain unique previously unlabeled TRAIN IDs")
        return PoolState(self.train_candidate_ids, self.labeled_ids + ids,
                         self.selected_history + (ids,), self.query_round + 1,
                         self.state_version + 1, self.split_fingerprint)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "PoolState":
        return cls(**value)
