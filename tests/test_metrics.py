"""Analytic evaluation checks; these arrays are not experimental ISIC results."""

import csv
import json

import numpy as np
import pytest

from csda_al.data.metadata import CLASSES
from csda_al.evaluation import (
    EvaluationLock,
    classification_metrics,
    compare_aulc,
    evaluate_predictions,
    export_result,
    frequency_groups,
    head_middle_tail_metrics,
    label_savings,
    normalized_aulc,
    plot_learning_curves,
)


def test_classification_metrics_hand_calculation():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 2])
    assert result["label_order"] == list(CLASSES)
    assert result["confusion_matrix"][0] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert result["confusion_matrix"][1] == [0, 1, 1, 0, 0, 0, 0, 0]
    assert result["support"] == [2, 2, 0, 0, 0, 0, 0, 0]
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["macro_precision"] == pytest.approx(1.5 / 8)
    assert result["macro_recall"] == pytest.approx(1 / 8)
    assert result["macro_f1"] == pytest.approx(7 / 48)
    assert result["weighted_precision"] == pytest.approx(0.75)
    assert result["weighted_recall"] == pytest.approx(0.5)
    assert result["weighted_f1"] == pytest.approx(7 / 12)
    assert all(result[f"micro_{name}"] == 0.5 for name in ("precision", "recall", "f1"))
    assert result["per_class"][2]["predicted_count"] == 1
    assert result["per_class"][2]["supported"] is False
    json.dumps(result, allow_nan=False)


def test_missing_classes_and_empty_evaluation_are_explicit():
    result = classification_metrics([7, 7], [7, 7])
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1 / 8
    assert result["zero_support_class_indices"] == list(range(7))
    empty = classification_metrics([], [])
    assert empty["accuracy"] is None
    assert empty["balanced_accuracy"] is None
    assert empty["macro_f1"] == 0
    assert empty["n_examples"] == 0
    assert empty["support"] == [0] * 8
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize(
    "truth,predicted",
    [([0], [8]), ([-1], [0]), ([0.0], [0]), ([True], [0]), ([0, 1], [0]), ([[0]], [0])],
)
def test_invalid_classes_or_alignment_rejected(truth, predicted):
    with pytest.raises(ValueError):
        classification_metrics(truth, predicted)


def test_group_metrics_use_only_acquired_counts_and_disclose_unobserved_classes():
    acquired = [0, 0, 0, 0, 1, 1, 2]
    groups = frequency_groups(acquired, head_min=4, middle_min=2)
    assert groups["head"] == [0]
    assert groups["middle"] == [1]
    assert groups["tail"] == [2, 3, 4, 5, 6, 7]
    assert groups["unobserved"] == [3, 4, 5, 6, 7]
    report = head_middle_tail_metrics(
        [0, 0, 1, 1], [0, 1, 1, 2], acquired, head_min=4, middle_min=2,
    )
    assert report["groups"]["head"]["balanced_accuracy"] == 0.5
    assert report["groups"]["middle"]["balanced_accuracy"] == 0.5
    assert report["groups"]["tail"]["balanced_accuracy"] is None
    with pytest.raises(ValueError):
        frequency_groups(acquired, head_min=2, middle_min=2)


def test_evaluator_probabilities_costs_and_atomic_exports(tmp_path):
    probabilities = np.eye(8)[[0, 1, 2]]
    result = evaluate_predictions(
        [0, 1, 7], probabilities, label_budget=8, validation_annotation_cost=3,
        checkpoint_id="synthetic-checkpoint",
    )
    assert result.per_class_support == (1, 1, 0, 0, 0, 0, 0, 1)
    assert result.to_dict()["total_annotation_cost"] == 11
    assert result.metrics["accuracy"] == pytest.approx(2 / 3)
    paths = export_result(result, tmp_path, synthetic=True)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["provenance"] == "synthetic_demonstration_no_real_training"
    assert payload["label_budget"] == 8
    assert payload["validation_annotation_cost"] == 3
    with paths["per_class_csv"].open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 8
    with paths["confusion_csv"].open(newline="", encoding="utf-8") as stream:
        matrix = list(csv.reader(stream))
    assert len(matrix) == 9
    assert matrix[0][1:] == list(CLASSES)
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        export_result(result, tmp_path, prefix="../escape")


@pytest.mark.parametrize("values", [np.ones((1, 8)), np.ones((1, 7)) / 7, np.array([[np.nan] * 8]), [-1]])
def test_invalid_evaluation_probabilities_or_predictions_rejected(values):
    with pytest.raises(ValueError):
        evaluate_predictions([0], values)


def test_test_evaluation_requires_matching_locked_identities():
    with pytest.raises(ValueError, match="EvaluationLock"):
        evaluate_predictions([0], [0], split="test")
    lock = EvaluationLock("checkpoint-1", "config-1", "split-1")
    with pytest.raises(ValueError, match="identities"):
        evaluate_predictions(
            [0], [0], split="test", lock=lock,
            checkpoint_id="changed", config_fingerprint="config-1", split_fingerprint="split-1",
        )
    result = evaluate_predictions(
        [0], [0], split="test", lock=lock,
        checkpoint_id="checkpoint-1", config_fingerprint="config-1", split_fingerprint="split-1",
    )
    assert result.metrics["accuracy"] == 1
    with pytest.raises(ValueError, match="locked"):
        evaluate_predictions(
            [0], [0], split="test", lock=EvaluationLock("a", "b", "c", locked=False),
            checkpoint_id="a", config_fingerprint="b", split_fingerprint="c",
        )
    with pytest.raises(ValueError):
        EvaluationLock("", "b", "c")
    with pytest.raises(ValueError):
        evaluate_predictions([0], [0], split="train")
    with pytest.raises(ValueError):
        evaluate_predictions([0], [0], label_budget=-1)


def test_normalized_aulc_uses_common_intervals_without_extrapolation():
    assert normalized_aulc([0, 10, 20], [0, 0.5, 1]) == 0.5
    assert normalized_aulc([0, 10, 20], [0, 0.5, 1], budget_interval=(5, 15)) == 0.5
    comparison = compare_aulc({
        "linear": ([0, 10, 20], [0, 0.5, 1]),
        "constant": ([5, 15, 25], [0.5, 0.5, 0.5]),
    })
    assert comparison["budget_interval"] == [5, 20]
    assert comparison["normalized_aulc"] == {"linear": 0.625, "constant": 0.5}
    with pytest.raises(ValueError, match="extrapolation"):
        normalized_aulc([0, 10], [0, 1], budget_interval=(0, 11))
    with pytest.raises(ValueError):
        compare_aulc({"a": ([0, 1], [0, 1]), "b": ([2, 3], [0, 1])})


@pytest.mark.parametrize("budgets,scores", [([0], [0]), ([0, 0], [0, 1]), ([2, 1], [0, 1]), ([0, 1], [0, 2]), ([0, 1], [0, np.nan])])
def test_invalid_learning_curves_rejected(budgets, scores):
    with pytest.raises(ValueError):
        normalized_aulc(budgets, scores)


def test_label_savings_distinguishes_estimates_and_unreached_targets():
    result = label_savings([0, 10, 20], [0, 0.5, 1], [0, 5, 10], [0, 0.5, 1], 0.75)
    assert result["status"] == "reached"
    assert result["baseline_budget"] == 15
    assert result["candidate_budget"] == 7.5
    assert result["labels_saved"] == 7.5
    assert result["fraction_saved"] == 0.5
    missing = label_savings([0, 10], [0, 0.5], [0, 10], [0, 0.7], 0.6)
    assert missing["status"] == "baseline_not_reached"
    assert missing["labels_saved"] is None
    assert missing["fraction_saved"] is None
    negative = label_savings([0, 10], [0, 1], [0, 20], [0, 1], 0.5)
    assert negative["labels_saved"] == -5


def test_synthetic_plot_is_a_standalone_artifact(tmp_path):
    path = plot_learning_curves(
        {"analytic fixture": ([0, 10, 20], [0, 0.5, 1])},
        tmp_path / "synthetic_curve.png", synthetic=True,
    )
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
