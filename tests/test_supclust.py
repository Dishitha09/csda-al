"""Paper-equation and geometric edge-case tests; no medical training."""
import numpy as np
import pytest
from csda_al.features.cache import FeatureStore
from csda_al.acquisition import QueryContext, SUPClustAcquisition, make_strategy
from csda_al.acquisition.supclust import typicality_scores, center_weights, sup_scores


def pool(dtype=np.float64):
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(4, 6))*4
    x = np.concatenate([c+rng.normal(scale=.2, size=(12, 6)) for c in centers]).astype(dtype)
    return FeatureStore(tuple(f"image_{i:03d}" for i in range(len(x))), x)


def test_equations_use_euclidean_distances_and_exclude_own_center():
    assert np.allclose(typicality_scores(np.array([[0.], [2.], [5.]]), 1), [.5, .5, 1/3])
    centers = np.array([[0., 0.], [3., 0.], [0., 4.]])
    expected = np.exp([-3., -4.])
    expected /= expected.sum()
    w = center_weights(centers, 0)
    assert w[0] == 0
    assert np.allclose(w[1:], expected)
    score = sup_scores(np.array([[1., 0.]]), centers, 0)[0]
    assert score == pytest.approx(1/(expected[0]*2+expected[1]*np.sqrt(17)))


def test_first_and_subsequent_round_cluster_rule():
    store = pool()
    first = SUPClustAcquisition().select(QueryContext(store, batch_budget=8))
    assert len(first.image_ids) == 8
    assert first.diagnostics["requested_clusters"] == 8
    second = SUPClustAcquisition().select(QueryContext(store, first.image_ids, batch_budget=8))
    assert len(second.image_ids) == 8
    assert second.diagnostics["requested_clusters"] == 16
    assert set(first.image_ids).isdisjoint(second.image_ids)
    assert len({x["cluster"] for x in second.diagnostics["candidates"]}) == 8


def test_input_permutation_determinism_and_float_precision():
    store = pool()
    baseline = SUPClustAcquisition().select(QueryContext(store, batch_budget=8))
    order = np.random.default_rng(93).permutation(len(store.image_ids))
    shuffled = FeatureStore(tuple(store.image_ids[i] for i in order), store.features[order])
    assert baseline.image_ids == SUPClustAcquisition().select(QueryContext(shuffled, batch_budget=8)).image_ids
    assert baseline.image_ids == SUPClustAcquisition().select(QueryContext(pool(np.float32), batch_budget=8)).image_ids


def test_duplicate_embeddings_and_single_center_have_audited_fallback():
    store = FeatureStore(("a", "b", "c", "d"), np.ones((4, 2)))
    selected = SUPClustAcquisition().select(QueryContext(store, batch_budget=3))
    assert selected.image_ids == ("a", "b", "c")
    assert selected.scores == (None, None, None)
    assert selected.diagnostics["requested_clusters"] == 3
    assert selected.diagnostics["effective_clusters"] == 1
    assert selected.diagnostics["fallback_ids"] == ["a", "b", "c"]
    assert set(selected.diagnostics["fallback_reasons"]) == {"SUP_undefined_single_center"}
    with pytest.raises(ValueError, match="degenerate"):
        SUPClustAcquisition(strict=True).select(QueryContext(store, batch_budget=3))


def test_tiny_pool_exhaustion_singletons_and_zero_budget():
    one = FeatureStore(("only",), np.ones((1, 2)))
    q = SUPClustAcquisition().select(QueryContext(one, batch_budget=8))
    assert q.image_ids == ("only",) and q.scores == (None,)
    assert SUPClustAcquisition().select(QueryContext(one, batch_budget=0)).image_ids == ()
    assert SUPClustAcquisition().select(QueryContext(one, labeled_ids=("only",))).image_ids == ()
    two = FeatureStore(("a", "b"), np.array([[1., 0.], [0., 1.]]))
    q = SUPClustAcquisition().select(QueryContext(two, batch_budget=2))
    assert len(q.image_ids) == 2
    assert all(c["singleton"] for c in q.diagnostics["candidates"])
    assert np.isfinite(q.scores).all()
    assert typicality_scores(np.ones((3, 2))).tolist() == [1e12]*3


@pytest.mark.parametrize("variant", ["without_SUP", "without_typicality", "typiclust", "kmeans"])
def test_named_variants_are_distinct_audited_methods(variant):
    context = QueryContext(pool(), batch_budget=4)
    strategy = SUPClustAcquisition(variant=variant)
    q = strategy.select(context)
    assert len(q.image_ids) == 4 and len(q.config_fingerprint) == 64
    assert q.method == variant
    assert q.image_ids == strategy.select(context).image_ids
    if variant == "without_SUP":
        assert q.scores == (None,)*4


def test_typicality_filter_restricts_even_without_sup_ablation():
    ids = tuple(f"{prefix}{i:02d}" for prefix in ("a", "b") for i in range(30))
    # Each cluster has 29 coincident dense points and one outlier. Top 10% retains
    # exactly the first three dense IDs. Seed 42 draws index 0 then index 2.
    x = np.array([[0.]]*29+[[2.]]+[[10.]]*29+[[12.]])
    store = FeatureStore(ids, x)
    c = QueryContext(store, batch_budget=2)
    q = make_strategy("supclust_without_sup", typical_fraction=.1).select(c)
    assert q.image_ids == ("a00", "b02")
    assert make_strategy("typiclust").select(c).image_ids == ("a00", "b00")
    assert make_strategy("supclust_without_typicality").select(c).image_ids == ("a29", "b00")


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"temperature": np.inf},
                                    {"typical_fraction": 0}, {"k_neighbors": True},
                                    {"n_init": 1.5}, {"epsilon": -1}])
def test_invalid_hyperparameters_fail_before_clustering(kwargs):
    with pytest.raises(ValueError):
        SUPClustAcquisition(**kwargs)
