"""Tests for Bengar class-balanced AL, Flexible Sampling, stopping criteria, and Macro-AUROC."""

import numpy as np
import pytest

from csda_al.acquisition import (
    BengarClassBalancedAcquisition,
    FlexibleSamplingAcquisition,
    CombinedCSDAAcquisition,
    PredictionBatch,
    QueryContext,
    make_strategy,
)
from csda_al.data.metadata import CLASSES
from csda_al.engine.stopping import ClassAwareStoppingCriterion, StoppingDecision
from csda_al.evaluation.metrics import classification_metrics, compute_macro_auroc


class MockFeatureStore:
    def __init__(self, ids):
        self.image_ids = tuple(ids)
        self.encoder_id = "mock"
        self.encoder_version = "1.0"
        self.preprocessing_id = "mock"
        self.normalization = "imagenet"
        self.split_fingerprint = "mock_split"
        self.schema_version = 1
        self.digest = "mock_digest"

    def rows(self, ids):
        return np.zeros((len(ids), 512), dtype=np.float32)


class MockDifficulty:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float64)
        self.model_checkpoint_id = "test_cp"
        self.validation_support = np.ones(8, dtype=np.int64) * 10
        self.validation_annotation_cost = 80


def test_bengar_strategy_registration():
    s1 = make_strategy("bengar")
    assert isinstance(s1, BengarClassBalancedAcquisition)
    s2 = make_strategy("bengar_class_balanced")
    assert isinstance(s2, BengarClassBalancedAcquisition)
    s3 = make_strategy("class_balanced")
    assert isinstance(s3, BengarClassBalancedAcquisition)


def test_flex_sampling_strategy_registration():
    s1 = make_strategy("flexible_sampling")
    assert isinstance(s1, FlexibleSamplingAcquisition)
    s2 = make_strategy("flex_sampling")
    assert isinstance(s2, FlexibleSamplingAcquisition)


def test_bengar_constrained_selection():
    n_candidates = 40
    ids = [f"img_{i:03d}" for i in range(n_candidates)]
    store = MockFeatureStore(ids)

    # 8 classes, unbalanced probabilities
    probs = np.full((n_candidates, 8), 0.05)
    # assign pseudo-classes: first 20 class 0, next 10 class 1, etc.
    for i in range(20):
        probs[i, 0] = 0.65
    for i in range(20, 30):
        probs[i, 1] = 0.65
    for i in range(30, 40):
        probs[i, 2] = 0.65

    pred = PredictionBatch(tuple(ids), probs, model_checkpoint_id="cp", pool_version=1)
    # Labeled counts: class 0 is heavily dominant (50 vs 2)
    labeled_counts = (50, 2, 2, 0, 0, 0, 0, 0)
    context = QueryContext(
        store, labeled_ids=(), unlabeled_ids=tuple(ids),
        batch_budget=8, pool_version=1, predictions=pred, labeled_counts=labeled_counts,
    )

    strategy = BengarClassBalancedAcquisition(mode="constrained")
    query = strategy.select(context)

    assert len(query.image_ids) == 8
    assert query.pool_version == 1
    assert query.method == "bengar_class_balanced"
    # Deficit should prioritize non-class-0 candidates
    assert query.diagnostics["configuration"]["mode"] == "constrained"
    assert "quotas" in query.diagnostics


def test_bengar_penalized_selection():
    n_candidates = 20
    ids = [f"img_{i:03d}" for i in range(n_candidates)]
    store = MockFeatureStore(ids)

    probs = np.full((n_candidates, 8), 0.1)
    probs[:, 0] = 0.3
    # Candidate 0 and 1 have higher uncertainty
    probs[0] = [0.125] * 8
    probs[1] = [0.125] * 8

    pred = PredictionBatch(tuple(ids), probs, model_checkpoint_id="cp", pool_version=1)
    labeled_counts = (100, 1, 1, 1, 1, 1, 1, 1)
    context = QueryContext(
        store, labeled_ids=(), unlabeled_ids=tuple(ids),
        batch_budget=4, pool_version=1, predictions=pred, labeled_counts=labeled_counts,
    )

    strategy = BengarClassBalancedAcquisition(mode="penalized", beta=1.0)
    query = strategy.select(context)
    assert len(query.image_ids) == 4
    assert query.diagnostics["mode"] == "penalized"


def test_flexible_sampling_selection():
    n_candidates = 24
    ids = [f"img_{i:03d}" for i in range(n_candidates)]
    store = MockFeatureStore(ids)

    probs = np.full((n_candidates, 8), 0.05)
    for i in range(n_candidates):
        probs[i, i % 8] = 0.65

    pred = PredictionBatch(tuple(ids), probs, model_checkpoint_id="test_cp", pool_version=1)
    diff = MockDifficulty([0.1, 0.9, 0.8, 0.2, 0.5, 0.4, 0.3, 0.7])
    context = QueryContext(
        store, labeled_ids=(), unlabeled_ids=tuple(ids),
        batch_budget=8, pool_version=1, predictions=pred, difficulty=diff,
    )

    strategy = FlexibleSamplingAcquisition(uncertainty_metric="entropy")
    query = strategy.select(context)

    assert len(query.image_ids) == 8
    assert query.method == "flexible_sampling"
    assert "class_probabilities" in query.diagnostics


def test_class_aware_stopping_criterion():
    stopping = ClassAwareStoppingCriterion(min_rounds=3, patience=2, min_delta=0.01, min_per_class=2)

    # Round 0: below min_rounds
    counts = [1, 1, 1, 1, 1, 1, 1, 1]
    val_metrics = {"macro_f1": 0.30, "per_class": [{"recall": 0.2}] * 8}
    dec0 = stopping.evaluate(0, counts, val_metrics)
    assert not dec0.should_stop
    assert "minimum_rounds_not_reached" in dec0.reason

    # Round 1: below min_rounds
    val_metrics = {"macro_f1": 0.35, "per_class": [{"recall": 0.3}] * 8}
    dec1 = stopping.evaluate(1, counts, val_metrics)
    assert not dec1.should_stop

    # Round 2: below min_rounds
    val_metrics = {"macro_f1": 0.40, "per_class": [{"recall": 0.35}] * 8}
    dec2 = stopping.evaluate(2, counts, val_metrics)
    assert not dec2.should_stop

    # Round 3: min_rounds reached, but min_per_class not met (class counts have 1 < 2)
    val_metrics = {"macro_f1": 0.405, "per_class": [{"recall": 0.35}] * 8}
    dec3 = stopping.evaluate(3, counts, val_metrics)
    assert not dec3.should_stop
    assert not dec3.min_class_coverage_met

    # Round 4: class counts now met (all >= 2), but macro_f1 improved (+0.05 > 0.01)
    counts_met = [5, 5, 4, 3, 3, 2, 2, 2]
    val_metrics = {"macro_f1": 0.46, "per_class": [{"recall": 0.4}] * 8}
    dec4 = stopping.evaluate(4, counts_met, val_metrics)
    assert not dec4.should_stop
    assert dec4.patience_counter == 0

    # Round 5: plateau (+0.002 < 0.01) -> patience 1
    val_metrics = {"macro_f1": 0.462, "per_class": [{"recall": 0.4}] * 8}
    dec5 = stopping.evaluate(5, counts_met, val_metrics)
    assert not dec5.should_stop
    assert dec5.patience_counter == 1

    # Round 6: plateau (+0.001 < 0.01) -> patience 2 >= 2 -> STOP!
    val_metrics = {"macro_f1": 0.463, "per_class": [{"recall": 0.4}] * 8}
    dec6 = stopping.evaluate(6, counts_met, val_metrics)
    assert dec6.should_stop
    assert "class_aware_plateau_patience_exhausted" in dec6.reason


def test_macro_auroc_computation():
    # 20 samples, 8 classes
    np.random.seed(42)
    y_true = np.random.randint(0, 8, size=40)
    # Ensure at least 2 distinct classes present for several classes
    y_true[:8] = np.arange(8)
    y_true[8:16] = np.arange(8)

    y_prob = np.random.uniform(0.01, 1.0, size=(40, 8))
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    result = compute_macro_auroc(y_true, y_prob, 8)
    assert result["macro_auroc"] is not None
    assert 0.0 <= result["macro_auroc"] <= 1.0
    assert len(result["per_class_auroc"]) == 8
    assert result["evaluated_classes"] == 8

    # Also check integrated classification_metrics
    metrics = classification_metrics(y_true, y_prob.argmax(axis=1), 8, y_prob=y_prob)
    assert "macro_auroc" in metrics
    assert metrics["macro_auroc"] is not None


def test_combined_csda_strategy_registration():
    s1 = make_strategy("csda_al")
    assert isinstance(s1, CombinedCSDAAcquisition)
    s2 = make_strategy("csda_al_combined")
    assert isinstance(s2, CombinedCSDAAcquisition)
    s3 = make_strategy("combined_csda")
    assert isinstance(s3, CombinedCSDAAcquisition)
    s4 = make_strategy("c4")
    assert isinstance(s4, CombinedCSDAAcquisition)


def test_combined_csda_selection():
    n_candidates = 32
    ids = [f"cand_{i:03d}" for i in range(n_candidates)]
    store = MockFeatureStore(ids)

    # Predictions with varying confidence and predicted classes
    probs = np.full((n_candidates, 8), 0.05)
    for i in range(n_candidates):
        probs[i, i % 8] = 0.65

    pred = PredictionBatch(tuple(ids), probs, model_checkpoint_id="test_cp", pool_version=1)
    diff = MockDifficulty([0.2, 0.9, 0.8, 0.1, 0.4, 0.5, 0.7, 0.6])
    labeled_counts = (40, 2, 2, 1, 1, 0, 0, 0)  # Heavy head class 0

    context = QueryContext(
        store, labeled_ids=(), unlabeled_ids=tuple(ids),
        batch_budget=8, pool_version=1, predictions=pred, difficulty=diff,
        labeled_counts=labeled_counts,
    )

    strategy = CombinedCSDAAcquisition(alpha=0.5)
    query = strategy.select(context)

    assert len(query.image_ids) == 8
    assert query.pool_version == 1
    assert query.method == "csda_al_combined"
    assert "quotas" in query.diagnostics
    assert "deficits" in query.diagnostics
    assert "strategy" in query.diagnostics
    # Head class 0 should have lowest deficit and therefore 0 or lowest quota
    assert query.diagnostics["quotas"][0] <= query.diagnostics["quotas"][5]

