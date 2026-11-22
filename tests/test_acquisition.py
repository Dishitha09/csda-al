"""Acquisition contracts, information gain and deterministic ID selection."""
from dataclasses import fields
import json
import numpy as np
import pytest
from csda_al.acquisition import (
    QueryContext, PredictionBatch, QueryBatch, make_strategy, entropy_scores,
    bald_scores, mc_dropout_probabilities,
)
from csda_al.features.cache import FeatureStore
from csda_al.imbalance import compute_class_difficulty


def context():
    ids = tuple(f"id_{i:02d}" for i in range(16))
    store = FeatureStore(ids, np.random.default_rng(4).normal(size=(16, 5)))
    p = np.full((16, 8), 1/8)
    p[1] = np.eye(8)[0]
    pred = PredictionBatch(ids[::-1], p[::-1], "checkpoint", 3)
    d = compute_class_difficulty(list(range(8)), [0]*8, model_checkpoint_id="checkpoint")
    mc = np.stack([p, np.roll(p, 1, axis=1)])
    return QueryContext(store, batch_budget=4, pool_version=3, predictions=pred,
                        difficulty=d, mc_probabilities=mc)


@pytest.mark.parametrize("name", ["random", "entropy", "coreset", "bald", "supclust",
                                  "typiclust", "kmeans", "difficulty_only", "difficulty_aware",
                                  "entropy_difficulty", "supclust_without_sup", "supclust_without_typicality"])
def test_all_strategies_return_valid_repeatable_batches(name):
    c = context()
    method = make_strategy(name)
    q = method.select(c)
    assert len(q.image_ids) == len(set(q.image_ids)) == 4
    assert set(q.image_ids) <= set(c.unlabeled_ids)
    assert q.image_ids == method.select(c).image_ids
    assert q.pool_version == 3
    json.dumps(q.to_dict(), allow_nan=False)


def test_entropy_and_bald_known_cases():
    uniform = np.full((1, 8), 1/8)
    certain = np.eye(8)[:1]
    assert entropy_scores(uniform)[0] == pytest.approx(1)
    assert entropy_scores(certain)[0] == 0
    assert np.allclose(bald_scores(np.stack([uniform, uniform])), 0)
    disagreement = np.stack([np.eye(8)[[0]], np.eye(8)[[1]]])
    assert bald_scores(disagreement)[0] == pytest.approx(np.log(2)/np.log(8))


def test_core_set_uses_geometry_and_id_ties():
    store = FeatureStore(("a", "b", "c"), np.array([[-2., 0.], [0., 0.], [2., 0.]]))
    q = make_strategy("coreset").select(QueryContext(store, batch_budget=2))
    assert q.image_ids == ("b", "a")
    q = make_strategy("coreset").select(QueryContext(store, labeled_ids=("b",), batch_budget=2))
    assert q.image_ids == ("a", "c")


def test_difficulty_only_and_entropy_can_rank_differently():
    store = FeatureStore(("a", "b"), np.array([[1., 0.], [0., 1.]]))
    p = np.stack([np.full(8, 1/8), np.eye(8)[1]])
    d = compute_class_difficulty([0, 1], [0, 0], smoothing=0)
    c = QueryContext(store, batch_budget=1, predictions=PredictionBatch(("a", "b"), p), difficulty=d)
    assert make_strategy("entropy").select(c).image_ids == ("a",)
    assert make_strategy("difficulty_only").select(c).image_ids == ("b",)


def test_optional_diversity_and_missing_validation_policy():
    c = context()
    q = make_strategy("difficulty_aware", diversity_weight=.4).select(c)
    assert len(q.image_ids) == 4 and q.diagnostics["diversity_shortlist_size"] == 16
    c = QueryContext(c.feature_store, batch_budget=3, predictions=PredictionBatch(c.train_pool_ids, np.full((16, 8), 1/8)))
    q = make_strategy("difficulty_aware").select(c)
    assert q.diagnostics["fallback_reason"] == "no_validation_difficulty_use_entropy"
    with pytest.raises(ValueError, match="validation"):
        make_strategy("difficulty_aware", missing_validation="raise").select(c)


def test_predictions_join_by_id_and_stale_metadata_fails():
    c = context()
    p = c.predictions.rows(c.unlabeled_ids)
    assert p[1, 0] == 1
    with pytest.raises(ValueError, match="stale"):
        QueryContext(c.feature_store, predictions=c.predictions, pool_version=4)
    with pytest.raises(ValueError, match="checkpoint"):
        QueryContext(c.feature_store, predictions=PredictionBatch(c.train_pool_ids, p, "other"), difficulty=c.difficulty)
    with pytest.raises(ValueError, match="partition"):
        QueryContext(c.feature_store, labeled_ids=("id_00",), unlabeled_ids=c.train_pool_ids)


def test_label_free_context_and_empty_budget_require_no_predictions():
    c = context()
    forbidden = {"labels", "truth", "targets", "oracle", "dataset", "validation_examples"}
    assert not forbidden.intersection(f.name for f in fields(QueryContext))
    for name in ["random", "entropy", "bald", "coreset", "supclust", "difficulty_aware"]:
        assert make_strategy(name).select(QueryContext(c.feature_store, batch_budget=0)).image_ids == ()
    with pytest.raises(ValueError):
        QueryContext(c.feature_store, batch_budget=True)
    with pytest.raises(ValueError):
        QueryBatch(("duplicate", "duplicate"))
    with pytest.raises(ValueError):
        QueryBatch(("a",), (np.nan,))


def test_mc_dropout_preserves_rng_batchnorm_and_module_modes():
    import torch
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(.5), torch.nn.Linear(4, 8))
    model.train()
    x = torch.randn(3, 4)
    states = [m.training for m in model.modules()]
    running = model[0].running_mean.clone()
    rng = torch.random.get_rng_state().clone()
    a = mc_dropout_probabilities(model, x, passes=3, seed=91)
    b = mc_dropout_probabilities(model, x, passes=3, seed=91)
    assert a.shape == (3, 3, 8) and np.array_equal(a, b)
    assert not np.array_equal(a[0], a[1])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(running, model[0].running_mean)
    assert states == [m.training for m in model.modules()]
    with pytest.raises(ValueError, match="dropout"):
        mc_dropout_probabilities(torch.nn.Linear(4, 8), x)
