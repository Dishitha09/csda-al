"""Fixed-budget AL with label-safe contexts and recoverable oracle transactions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..acquisition import PredictionBatch, QueryBatch, QueryContext, make_strategy, mc_dropout_probabilities
from ..config import Config
from ..data.dataset import ImageDataset, LabeledImageDataset
from ..data.metadata import ImageRecord, CLASSES
from ..data.oracle import LabelOracle
from ..data.transforms import build_image_transform
from ..evaluation import evaluate_predictions
from ..evaluation.report import export_result
from ..features.cache import FeatureStore
from ..imbalance.class_difficulty import compute_class_difficulty
from ..imbalance.diagnostics import post_annotation_composition
from ..training.supervised_train import SupervisedTrainer
from ..utils.io import RunLock, atomic_json, read_json, fingerprint
from ..utils.logging import JsonlLogger
from ..utils.seed import seed_everything, seed_worker
from .budget import AnnotationBudget
from .checkpointing import load_checkpoint, save_checkpoint
from .pool_state import PoolState
from .stopping import ClassAwareStoppingCriterion


class ActiveLearningEngine:
    """Only TRAIN records/truth-oracle and paid validation inputs are accepted.

    No test records or targets are part of the controller API. The authoritative
    state and durable oracle request ledger permit retry across crash windows.
    """

    def __init__(self, config: Config, train_records: Sequence[ImageRecord],
                 validation_records: Sequence[ImageRecord], validation_labels: dict[str, int],
                 feature_store: FeatureStore, oracle: LabelOracle, trainer: SupervisedTrainer,
                 *, synthetic: bool = False):
        self.config = config
        self.records = {r.image_id: r for r in train_records}
        self.validation_records = tuple(validation_records)
        self.validation_labels = dict(validation_labels)
        if len(self.records) != len(train_records) or any(r.split != "train" for r in train_records):
            raise ValueError("Engine candidates must be unique TRAIN records")
        if set(feature_store.image_ids) != set(self.records):
            raise ValueError("Feature store must cover exactly TRAIN candidates")
        if any(r.split != "validation" for r in validation_records):
            raise ValueError("Difficulty evaluation accepts validation records only")
        val_ids = {r.image_id for r in validation_records}
        if len(val_ids) != len(validation_records) or val_ids != set(validation_labels):
            raise ValueError("Validation labels must cover exactly validation records")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 8
               for v in validation_labels.values()):
            raise ValueError("Validation labels must be integer class indices 0..7")
        if val_ids.intersection(self.records):
            raise ValueError("TRAIN and validation IDs overlap")
        if oracle.ledger_path is None:
            raise ValueError("Persistent active learning requires a durable oracle ledger_path")
        self.store, self.oracle, self.trainer = feature_store, oracle, trainer
        self.synthetic = synthetic
        self.output = Path(config.experiment.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.budget = AnnotationBudget(config.experiment.max_annotations, len(val_ids),
                                       config.experiment.total_label_cap)
        self.split_hash = feature_store.split_fingerprint
        self.run_identity = fingerprint({
            "config": config.to_dict(), "features": feature_store.digest,
            "feature_provenance": {key: getattr(feature_store, key) for key in
                                   ("encoder_id", "encoder_version", "preprocessing_id",
                                    "normalization", "split_fingerprint", "schema_version")},
            "validation_labels_digest": fingerprint(self.validation_labels),
            "validation_ids": sorted(val_ids), "synthetic": synthetic,
        })
        self.logger = JsonlLogger(self.output / "events.jsonl")
        stopping_cfg = getattr(self.config, "stopping", None)
        stopping_enabled = getattr(stopping_cfg, "enabled", False) if stopping_cfg else False
        min_rounds = getattr(stopping_cfg, "min_rounds", 3) if stopping_cfg else 3
        patience = getattr(stopping_cfg, "patience", 2) if stopping_cfg else 2
        min_delta = getattr(stopping_cfg, "min_delta", 0.005) if stopping_cfg else 0.005
        min_per_class = getattr(stopping_cfg, "min_per_class", 2) if stopping_cfg else 2
        minority_delta = getattr(stopping_cfg, "minority_delta", 0.01) if stopping_cfg else 0.01
        self.stopping_criterion = ClassAwareStoppingCriterion(
            min_rounds=min_rounds,
            patience=patience,
            min_delta=min_delta,
            min_per_class=min_per_class,
            minority_delta=minority_delta,
            enabled=stopping_enabled,
        )

    def _loader(self, ids: Sequence[str] | None = None, *, labels: dict | None = None,
                validation: bool = False, training: bool = False, seed: int = 42) -> DataLoader:
        records = list(self.validation_records) if validation else [self.records[i] for i in ids or ()]
        transform = build_image_transform(self.config.dataset.image_size, training=training)
        if labels is None:
            dataset = ImageDataset(records, transform)
        else:
            dataset = LabeledImageDataset(records, labels, transform, acquired_only=not validation)
        return DataLoader(dataset, batch_size=self.config.classifier.batch_size, shuffle=training,
                          num_workers=self.config.experiment.num_workers,
                          generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)

    def _strategy(self, name: str):
        acq = self.config.acquisition
        if name in {"supclust", "typiclust", "kmeans", "kmeans_representative", "without_sup",
                    "without_typicality", "supclust_without_sup", "supclust_without_typicality"}:
            return make_strategy(name, temperature=acq.temperature, k_neighbors=acq.k_neighbors)
        if name in {"difficulty_aware", "entropy_difficulty"}:
            return make_strategy(name, alpha=acq.alpha, diversity_weight=0.25 if acq.diversity else 0.0)
        if name == "difficulty_only":
            return make_strategy(name, diversity_weight=0.25 if acq.diversity else 0.0)
        return make_strategy(name)

    def _validate_snapshot(self, snapshot: dict, max_train_steps: int | None) -> None:
        """Reject inconsistent resumes before a further annotation can be charged."""
        if snapshot.get("schema_version") != 1 or snapshot.get("run_identity") != self.run_identity:
            raise ValueError("Resume config, feature provenance, split or validation inputs changed")
        if snapshot.get("max_train_steps") != max_train_steps:
            raise ValueError("Resume must preserve the recorded training-step limit")
        state = PoolState.from_dict(snapshot["pool"])
        if set(state.train_candidate_ids) != set(self.records) or state.split_fingerprint != self.split_hash:
            raise ValueError("Resume pool does not match TRAIN candidates/split")
        labels = snapshot.get("acquired_labels", {})
        if set(labels) != set(state.labeled_ids):
            raise ValueError("Acquired label IDs and committed pool differ")
        if tuple(tuple(q["image_ids"]) for q in snapshot.get("history", [])) != state.selected_history:
            raise ValueError("Acquisition audit history and committed pool differ")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 8 for v in labels.values()):
            raise ValueError("Invalid acquired labels in committed snapshot")
        phase = snapshot.get("phase")
        if phase not in {"ready", "prepared", "training_pending", "evaluation_pending"}:
            raise ValueError("Unknown active-learning transaction phase")
        allowed_pools = [set(state.labeled_ids)]
        if phase == "prepared":
            pending = snapshot.get("pending") or {}
            query = QueryBatch(**pending.get("query", {}))
            if (pending.get("request_id") != f"round_{state.query_round:04d}"
                    or query.pool_version != state.state_version or not query.image_ids
                    or not set(query.image_ids).issubset(state.unlabeled_ids)):
                raise ValueError("Invalid prepared annotation transaction")
            proposed = (self.config.acquisition.initial_budget if not state.labeled_ids
                        else self.config.acquisition.batch_budget)
            if len(query.image_ids) > min(proposed, self.budget.remaining(len(state.labeled_ids))):
                raise ValueError("Prepared annotation transaction exceeds its budget")
            allowed_pools.append(set(state.labeled_ids).union(query.image_ids))
        if set(self.oracle.annotated_ids) not in allowed_pools:
            raise ValueError("Oracle ledger and committed pool counts differ")

    def _query(self, state: PoolState, snapshot: dict, count: int) -> QueryBatch:
        initial = not state.labeled_ids
        name = self.config.acquisition.initial_strategy if initial else self.config.acquisition.strategy
        predictions, difficulty, mc = None, None, None
        if not initial:
            loader = self._loader(state.unlabeled_ids)
            ids, probabilities = self.trainer.predict(loader)
            predictions = PredictionBatch(ids, probabilities, snapshot["checkpoint_id"], state.state_version)
            if self.validation_records:
                val_ids, val_probs = self.trainer.predict(self._loader(validation=True))
                difficulty = compute_class_difficulty(
                    [self.validation_labels[i] for i in val_ids], val_probs.argmax(axis=1),
                    smoothing=self.config.acquisition.smoothing,
                    recall_prior=self.config.acquisition.prior_recall,
                    validation_annotation_cost=self.budget.validation_annotation_cost,
                    model_checkpoint_id=snapshot["checkpoint_id"], validation_split_hash=self.split_hash)
            if difficulty is None and name in {"difficulty_aware", "entropy_difficulty", "difficulty_only"}:
                name = "entropy"
                self.logger.log("no_validation_fallback", strategy="entropy", round=state.query_round)
            if name == "bald":
                draws = []
                for batch in loader:
                    draws.append(mc_dropout_probabilities(self.trainer.model, batch[0].to(self.trainer.device),
                                 passes=self.config.acquisition.mc_passes,
                                 seed=self.config.experiment.seed + state.query_round))
                mc = np.concatenate(draws, axis=1)
        labeled_counts = None
        if snapshot.get("acquired_labels") and state.labeled_ids:
            raw_counts = np.bincount([snapshot["acquired_labels"][i] for i in state.labeled_ids],
                                     minlength=len(CLASSES))
            labeled_counts = tuple(int(x) for x in raw_counts[:len(CLASSES)])
        context = QueryContext(self.store, state.labeled_ids, state.unlabeled_ids, count,
                               self.config.experiment.seed + state.query_round, state.state_version,
                               predictions, difficulty, mc, labeled_counts)
        query = self._strategy(name).select(context)
        # This is deliberately a second validation outside the strategy.
        if (query.pool_version != state.state_version or len(query.image_ids) != count
                or len(set(query.image_ids)) != len(query.image_ids)
                or not set(query.image_ids).issubset(state.unlabeled_ids)):
            raise ValueError("Strategy returned invalid, stale, repeated or underfilled query")
        return query

    def run(self, *, resume: bool = False, max_train_steps: int | None = None) -> dict:
        """Execute to a fixed budget/round count; max_train_steps bounds smoke work per round."""
        if max_train_steps is not None and (isinstance(max_train_steps, bool)
                                           or not isinstance(max_train_steps, int) or max_train_steps < 1):
            raise ValueError("max_train_steps must be a positive integer")
        with RunLock(self.output / ".run.lock"):
            state_path = self.output / "state.json"
            if state_path.exists():
                if not resume:
                    raise ValueError("Run already exists; use --resume or a fresh output directory")
                snapshot = read_json(state_path)
                self._validate_snapshot(snapshot, max_train_steps)
                if snapshot.get("checkpoint"):
                    checkpoint_path = (self.output / snapshot["checkpoint"]).resolve()
                    if not checkpoint_path.is_relative_to(self.output.resolve()):
                        raise ValueError("Checkpoint path escapes the run directory")
                    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != snapshot["checkpoint_id"]:
                        raise ValueError("Committed checkpoint checksum differs")
                    load_checkpoint(checkpoint_path, self.trainer.model,
                                    optimizer=self.trainer.optimizer,
                                    expected_metadata={"run_identity": self.run_identity}, restore_rng=True)
                    self.trainer.total_steps = snapshot.get("trainer_steps", 0)
            else:
                if self.oracle.unique_annotations:
                    raise ValueError("Fresh engine requires zero acquired TRAIN labels")
                state = PoolState(tuple(sorted(self.records)), split_fingerprint=self.split_hash)
                snapshot = {"schema_version": 1, "run_identity": self.run_identity, "pool": state.to_dict(),
                            "phase": "ready", "acquired_labels": {}, "checkpoint": None,
                            "checkpoint_id": "", "pending": None, "synthetic": self.synthetic,
                            "validation_annotation_cost": self.budget.validation_annotation_cost,
                            "history": [], "trainer_steps": 0, "max_train_steps": max_train_steps}
                # Needed to replay the first optimization from identical weights after a crash.
                save_checkpoint(self.output / "initial.pt", self.trainer.model,
                                optimizer=self.trainer.optimizer, metadata={"run_identity": self.run_identity})
                snapshot["checkpoint"] = "initial.pt"
                snapshot["checkpoint_id"] = hashlib.sha256((self.output / "initial.pt").read_bytes()).hexdigest()
                atomic_json(state_path, snapshot)
                self.logger.log("run_started", synthetic=self.synthetic, config_fingerprint=self.config.fingerprint,
                                seed=self.config.experiment.seed, validation_cost=len(self.validation_records))
            while True:
                state = PoolState.from_dict(snapshot["pool"])
                if snapshot["phase"] == "ready":
                    remaining = self.budget.remaining(len(state.labeled_ids))
                    if (snapshot.get("stopped_early") or remaining == 0 or not state.unlabeled_ids
                            or state.query_round >= self.config.experiment.max_rounds):
                        break
                    proposed = (self.config.acquisition.initial_budget if not state.labeled_ids
                                else self.config.acquisition.batch_budget)
                    query = self._query(state, snapshot, min(proposed, remaining, len(state.unlabeled_ids)))
                    snapshot["pending"] = {"request_id": f"round_{state.query_round:04d}",
                                           "query": json.loads(json.dumps(query.to_dict(), allow_nan=False))}
                    snapshot["phase"] = "prepared"
                    atomic_json(state_path, snapshot)
                if snapshot["phase"] == "prepared":
                    pending = snapshot["pending"]
                    ids = tuple(pending["query"]["image_ids"])
                    receipt = self.oracle.annotate(ids, request_id=pending["request_id"])
                    acquired = dict(zip(receipt.image_ids, receipt.acquired_labels, strict=True))
                    state = state.acquire(ids)
                    if set(self.oracle.annotated_ids) != set(state.labeled_ids):
                        raise ValueError("Oracle ledger and acquired pool counts differ")
                    snapshot["pool"] = state.to_dict()
                    snapshot["acquired_labels"].update(acquired)
                    snapshot["phase"] = "training_pending"
                    snapshot["history"].append(pending["query"])
                    atomic_json(state_path, snapshot)
                    # Truth-derived composition is retrospective only, after the query is committed.
                    atomic_json(self.output / "diagnostics" / f"round_{state.query_round:04d}_composition.json",
                                {"synthetic": self.synthetic, **post_annotation_composition(receipt.acquired_labels)})
                    self.logger.log("annotations_committed", round=state.query_round, ids=list(ids),
                                    training_cost=len(state.labeled_ids), method=pending["query"]["method"])
                if snapshot["phase"] == "training_pending":
                    state = PoolState.from_dict(snapshot["pool"])
                    seed_everything(self.config.experiment.seed + state.query_round)
                    started = time.perf_counter()
                    training = self.trainer.fit(self._loader(state.labeled_ids,
                                  labels=snapshot["acquired_labels"], training=True,
                                  seed=self.config.experiment.seed + state.query_round),
                                  epochs=self.config.classifier.epochs, max_steps=max_train_steps)
                    elapsed = time.perf_counter() - started
                    checkpoint = f"checkpoints/round_{state.query_round:04d}.pt"
                    save_checkpoint(self.output / checkpoint, self.trainer.model, optimizer=self.trainer.optimizer,
                                    metadata={"run_identity": self.run_identity, "class_order": list(CLASSES),
                                              "config_fingerprint": self.config.fingerprint,
                                              "split_fingerprint": self.split_hash, "synthetic": self.synthetic,
                                              "labeled_ids_hash": fingerprint(state.labeled_ids),
                                              "encoder_version": self.store.encoder_version,
                                              "preprocessing_id": self.store.preprocessing_id,
                                              "steps": self.trainer.total_steps})
                    checkpoint_id = hashlib.sha256((self.output / checkpoint).read_bytes()).hexdigest()
                    snapshot.update(checkpoint=checkpoint, checkpoint_id=checkpoint_id,
                                    trainer_steps=self.trainer.total_steps, phase="evaluation_pending", pending=None)
                    atomic_json(state_path, snapshot)
                    self.logger.log("model_ready", round=state.query_round, checkpoint_id=checkpoint_id,
                                    training=training, compute_seconds=elapsed, synthetic=self.synthetic)
                if snapshot["phase"] == "evaluation_pending":
                    state = PoolState.from_dict(snapshot["pool"])
                    if self.validation_records:
                        val_ids, probabilities = self.trainer.predict(self._loader(validation=True))
                        result = evaluate_predictions([self.validation_labels[i] for i in val_ids], probabilities,
                                  checkpoint_id=snapshot["checkpoint_id"], config_fingerprint=self.config.fingerprint,
                                  split_fingerprint=self.split_hash, label_budget=len(state.labeled_ids),
                                  validation_annotation_cost=self.budget.validation_annotation_cost)
                        export_result(result, self.output / "validation", prefix=f"round_{state.query_round:04d}",
                                      synthetic=self.synthetic)
                        acquired_counts = tuple(np.bincount([snapshot["acquired_labels"][i] for i in state.labeled_ids],
                                                            minlength=len(CLASSES)))
                        decision = self.stopping_criterion.evaluate(state.query_round, acquired_counts, result.metrics)
                        atomic_json(self.output / "diagnostics" / f"round_{state.query_round:04d}_stopping.json",
                                    {"synthetic": self.synthetic, **decision.to_dict()})
                        if decision.should_stop:
                            snapshot["stopped_early"] = True
                            snapshot["stop_reason"] = decision.reason
                            self.logger.log("early_stopping_triggered", round=state.query_round, reason=decision.reason)
                    snapshot["phase"] = "ready"
                    atomic_json(state_path, snapshot)
            state = PoolState.from_dict(snapshot["pool"])
            term_reason = snapshot.get("stop_reason") if snapshot.get("stopped_early") else "fixed_budget_round_limit_or_exhausted_pool"
            summary = {"synthetic": self.synthetic, "no_real_training": self.synthetic,
                       "rounds": state.query_round, "training_annotations": len(state.labeled_ids),
                       "validation_annotations": self.budget.validation_annotation_cost,
                       "total_annotation_cost": len(state.labeled_ids) + self.budget.validation_annotation_cost,
                       "unlabeled_remaining": len(state.unlabeled_ids), "training_steps": self.trainer.total_steps,
                       "checkpoint_id": snapshot["checkpoint_id"], "history": snapshot["history"],
                       "termination": term_reason}
            atomic_json(self.output / "summary.json", summary)
            return summary
