"""TRAIN-only truth with durable, replayable requests and unique charges."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..utils.io import atomic_json, read_json, fingerprint


@dataclass(frozen=True)
class AnnotationReceipt:
    request_id: str
    image_ids: tuple[str, ...]
    acquired_labels: tuple[int, ...]
    new_unique_cost: int
    cumulative_training_cost: int
    receipt_hash: str


class LabelOracle:
    @classmethod
    def from_prepared_train(cls, bundle_dir: str | Path, *, max_annotations: int | None = None,
                            ledger_path: str | Path | None = None) -> "LabelOracle":
        """Load only checksum-bound TRAIN truth inside the sealed oracle boundary."""
        from .prepared import _load_private_truth
        truth = _load_private_truth(bundle_dir, "train")
        return cls(truth, train_ids=tuple(truth), max_annotations=max_annotations, ledger_path=ledger_path)

    def __init__(self, train_truth: Mapping[str, int], *, train_ids: Sequence[str] | None = None,
                 max_annotations: int | None = None, ledger_path: str | Path | None = None):
        self.__truth = dict(train_truth)
        if train_ids is not None:
            supplied_ids = tuple(train_ids)
            if len(supplied_ids) != len(set(supplied_ids)) or set(supplied_ids) != set(train_truth):
                raise ValueError("Oracle mapping must cover exactly unique TRAIN IDs")
        if not self.__truth or any(not isinstance(k, str) or not k for k in self.__truth):
            raise ValueError("Oracle requires nonempty TRAIN IDs")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 8
               for v in self.__truth.values()):
            raise ValueError("Oracle targets must be integer class indices 0..7")
        self.__maximum = len(train_truth) if max_annotations is None else max_annotations
        if isinstance(self.__maximum, bool) or not isinstance(self.__maximum, int) or self.__maximum < 0:
            raise ValueError("max_annotations must be a nonnegative integer")
        self.__path = Path(ledger_path) if ledger_path else None
        self.__receipts = read_json(self.__path) if self.__path and self.__path.exists() else {}
        if not isinstance(self.__receipts, dict):
            raise ValueError("Oracle ledger must be a mapping of request receipts")
        self.__charged: set[str] = set()
        for request_id, receipt in self.__receipts.items():
            expected_fields = {"request_id", "image_ids", "acquired_labels", "new_unique_cost",
                               "cumulative_training_cost", "receipt_hash"}
            if not isinstance(receipt, dict) or set(receipt) != expected_fields:
                raise ValueError("Oracle ledger has an invalid receipt schema")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("Oracle ledger has an invalid request ID")
            data = {k: v for k, v in receipt.items() if k != "receipt_hash"}
            if receipt["receipt_hash"] != fingerprint(data) or request_id != receipt["request_id"]:
                raise ValueError("Oracle ledger is corrupt")
            ids = receipt["image_ids"]
            labels = receipt["acquired_labels"]
            if (not isinstance(ids, list) or not ids or not isinstance(labels, list)
                    or len(ids) != len(labels) or any(not isinstance(i, str) or not i for i in ids)):
                raise ValueError("Oracle ledger has invalid image IDs or labels")
            for cost_name in ("new_unique_cost", "cumulative_training_cost"):
                cost = receipt[cost_name]
                if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
                    raise ValueError("Oracle ledger annotation costs must be nonnegative integers")
            if receipt["new_unique_cost"] != len(ids):
                raise ValueError("Oracle ledger unique annotation cost does not match its image IDs")
            if self.__charged.intersection(ids) or len(ids) != len(set(ids)):
                raise ValueError("Duplicate oracle charges in ledger")
            if any(i not in self.__truth or self.__truth[i] != y
                   for i, y in zip(ids, receipt["acquired_labels"], strict=True)):
                raise ValueError("Oracle ledger does not match the supplied TRAIN truth")
            self.__charged.update(ids)
        cumulative = 0
        for receipt in sorted(self.__receipts.values(), key=lambda item: item["cumulative_training_cost"]):
            cumulative += receipt["new_unique_cost"]
            if receipt["cumulative_training_cost"] != cumulative:
                raise ValueError("Oracle ledger cumulative annotation costs are inconsistent")
        if len(self.__charged) > self.__maximum:
            raise ValueError("Committed oracle costs exceed configured budget")

    @property
    def unique_annotations(self) -> int:
        return len(self.__charged)

    @property
    def ledger_path(self) -> Path | None:
        """Durable request-ledger location; exposes no truth or label lookup."""
        return self.__path.resolve() if self.__path is not None else None

    @property
    def annotated_ids(self) -> tuple[str, ...]:
        """Previously charged IDs only; labels remain available via receipts."""
        return tuple(sorted(self.__charged))

    def annotate(self, selected_ids: Sequence[str], *, request_id: str | None = None) -> AnnotationReceipt:
        ids = tuple(selected_ids)
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
            raise ValueError("request_id must be a nonempty string")
        request_id = request_id if request_id is not None else fingerprint(ids)
        # The engine holds a run-level OS writer lock. Also reject stale oracle
        # instances rather than overwriting receipts committed since construction.
        if self.__path:
            persisted = read_json(self.__path) if self.__path.exists() else {}
            if fingerprint(persisted) != fingerprint(self.__receipts):
                raise ValueError("Oracle ledger changed; reload the oracle before annotating")
        if request_id in self.__receipts:
            data = self.__receipts[request_id]
            if tuple(data["image_ids"]) != ids:
                raise ValueError("Request ID reused with different selected IDs")
            return AnnotationReceipt(**{**data, "image_ids": ids,
                                        "acquired_labels": tuple(data["acquired_labels"]), "new_unique_cost": 0})
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Annotation request must contain unique nonempty IDs")
        if any(i not in self.__truth for i in ids):
            raise ValueError("Annotation may query TRAIN IDs only")
        if self.__charged.intersection(ids):
            raise ValueError("Previously acquired ID in a new request")
        if len(self.__charged) + len(ids) > self.__maximum:
            raise ValueError("Annotation budget exceeded")
        data = {"request_id": request_id, "image_ids": list(ids),
                "acquired_labels": [self.__truth[i] for i in ids], "new_unique_cost": len(ids),
                "cumulative_training_cost": len(self.__charged) + len(ids)}
        data["receipt_hash"] = fingerprint(data)
        new_receipts = {**self.__receipts, request_id: data}
        if self.__path:
            atomic_json(self.__path, new_receipts)
        self.__receipts = new_receipts
        self.__charged.update(ids)
        return AnnotationReceipt(**{**data, "image_ids": ids, "acquired_labels": tuple(data["acquired_labels"])})
