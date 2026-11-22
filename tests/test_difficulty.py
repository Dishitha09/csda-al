"""Known-array checks of validation recall, smoothing and predicted membership."""
import json
import numpy as np
import pytest
from csda_al.imbalance import compute_class_difficulty, expected_difficulty, normalize_difficulty
from csda_al.imbalance.diagnostics import post_annotation_composition


def test_recall_smoothing_and_prior_are_explicit():
    d = compute_class_difficulty([0, 0, 1], [0, 1, 0], model_checkpoint_id="model-v1")
    assert d.values[0] == pytest.approx(.5)
    assert d.values[1] == pytest.approx(.75)
    assert d.values[2] == pytest.approx(.5)
    assert d.validation_support.tolist() == [2, 1, 0, 0, 0, 0, 0, 0]
    assert d.raw_recall[:2].tolist() == [.5, 0]
    assert d.prior_mask.tolist() == [False, False, True, True, True, True, True, True]
    assert d.validation_annotation_cost == 3
    assert d.to_dict()["raw_recall"][2] is None
    json.dumps(d.to_dict(), allow_nan=False)
    with pytest.raises(ValueError):
        d.values[0] = 1


def test_unsmoothed_supported_recall_and_zero_labels():
    d = compute_class_difficulty([0, 0, 1], [0, 1, 1], smoothing=0)
    assert d.values[:2].tolist() == [.5, 0]
    assert compute_class_difficulty([], []) is None
    with pytest.raises(ValueError, match="zero support"):
        compute_class_difficulty([0], [0], missing_policy="raise")


def test_predicted_membership_not_argmax_buckets():
    p = np.zeros((2, 8))
    p[0, :2] = [.75, .25]
    p[1, :2] = [.25, .75]
    difficulty = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
    assert np.allclose(expected_difficulty(p, difficulty), [.25, .75])
    normalized, diagnostic = normalize_difficulty([.25, .75])
    assert normalized.tolist() == [0., 1.]
    assert diagnostic["minimum"] == .25 and diagnostic["maximum"] == .75
    normalized, diagnostic = normalize_difficulty([.5, .5])
    assert normalized.tolist() == [0., 0.] and diagnostic["constant_difficulty_signal"]
    assert normalize_difficulty([.5, .5], method="identity")[0].tolist() == [.5, .5]


@pytest.mark.parametrize("kwargs", [{"split": "test"}, {"validation_annotation_cost": 0},
                                    {"smoothing": -1}, {"recall_prior": 2},
                                    {"num_classes": 9}, {"missing_policy": "silent"}])
def test_invalid_or_test_provenance_fails(kwargs):
    with pytest.raises(ValueError):
        compute_class_difficulty([0], [0], **kwargs)


def test_composition_is_explicitly_retrospective():
    result = post_annotation_composition([1, 1, 7])
    assert result["counts"] == [0, 2, 0, 0, 0, 0, 0, 1]
    assert result["phase"] == "post_annotation_only"
