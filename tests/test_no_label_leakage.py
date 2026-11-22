import inspect
import numpy as np
import pytest

from csda_al.acquisition import QueryContext, PredictionBatch, make_strategy
from csda_al.data.oracle import LabelOracle
from csda_al.features.cache import FeatureStore
from csda_al.imbalance.class_difficulty import compute_class_difficulty


@pytest.mark.parametrize("strategy", ["random", "supclust", "typiclust", "kmeans", "coreset",
                                      "entropy", "difficulty_only", "difficulty_aware"])
def test_hidden_truth_permutation_cannot_change_selection(strategy):
    ids = tuple(f"id{i:02d}" for i in range(16))
    store = FeatureStore(ids, np.random.default_rng(7).normal(size=(16, 6)))
    probs = np.random.default_rng(8).dirichlet(np.ones(8), size=16)
    predictions = PredictionBatch(ids, probs, "model")
    difficulty = compute_class_difficulty([0, 1], [0, 0], model_checkpoint_id="model")
    context = QueryContext(store, batch_budget=4, predictions=predictions, difficulty=difficulty)
    oracle_a = LabelOracle({i: n % 8 for n, i in enumerate(ids)})
    oracle_b = LabelOracle({i: (n+3) % 8 for n, i in enumerate(ids)})
    query_a = make_strategy(strategy).select(context)
    query_b = make_strategy(strategy).select(context)
    assert query_a.to_dict() == query_b.to_dict()
    assert oracle_a.unique_annotations == oracle_b.unique_annotations == 0
    # The only subsequent truth access is through the explicit controller action.
    assert oracle_a.annotate(query_a.image_ids).acquired_labels != oracle_b.annotate(query_b.image_ids).acquired_labels


def test_context_and_engine_have_no_test_or_truth_inputs():
    from csda_al.engine.active_loop import ActiveLearningEngine
    assert not {"oracle", "truth", "labels", "targets", "dataset"}.intersection(inspect.signature(QueryContext).parameters)
    assert not any("test" in name for name in inspect.signature(ActiveLearningEngine).parameters)
    with pytest.raises(ValueError, match="validation"):
        compute_class_difficulty([0], [0], split="test")
