"""Known-count privileged preparation QA, without training or selection."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from csda_al.data.analysis import (
    analyze_class_distribution, analyze_dataset, analyze_metadata, export_analysis,
)
from csda_al.data.metadata import CLASSES, ImageRecord


def records_and_truth():
    ids = tuple(f"image_{i}" for i in range(8))
    records = [ImageRecord(i, Path(f"unused/{i}.jpg")) for i in ids]
    labels = dict(zip(ids, [0, 0, 0, 0, 1, 1, 2, 3], strict=True))
    splits = dict(zip(ids, ["train"]*4+["validation"]*2+["test"]*2, strict=True))
    return records, labels, splits


def test_hand_counts_entropy_effective_classes_and_missing_denominators():
    result = analyze_class_distribution([0, 0, 0, 0, 1, 1, 2, 3])["overall"]
    assert result["support"] == [4, 2, 1, 1, 0, 0, 0, 0]
    assert result["num_images"] == 8
    assert result["shannon_entropy_bits"] == pytest.approx(1.75)
    assert result["effective_class_count"] == pytest.approx(2**1.75)
    assert result["normalized_entropy"] == pytest.approx(1.75/3)
    assert result["imbalance_ratio"] is None
    assert result["imbalance_ratio_observed"] == 4
    assert result["minimum_class_count"] == 0
    assert result["minimum_positive_class_count"] == 1
    assert result["missing_classes"] == ["BKL", "DF", "VASC", "SCC"]


def test_uniform_eight_classes_and_empty_inventory_are_explicit():
    uniform = analyze_class_distribution(list(range(8)))["overall"]
    assert uniform["imbalance_ratio"] == 1
    assert uniform["shannon_entropy_bits"] == pytest.approx(3)
    assert uniform["effective_class_count"] == pytest.approx(8)
    assert uniform["normalized_entropy"] == pytest.approx(1)
    empty = analyze_class_distribution([])["overall"]
    assert empty["support"] == [0]*8
    assert empty["shannon_entropy_bits"] is None
    assert empty["effective_class_count"] is None
    json.dumps(empty, allow_nan=False)


def test_id_based_class_and_split_counts_do_not_require_contiguous_rows():
    _, labels, assignments = records_and_truth()
    labels = dict(reversed(list(labels.items())))
    result = analyze_class_distribution(labels, assignments)
    assert result["split_counts"] == {"train": 4, "validation": 2, "test": 2}
    assert result["splits"]["train"]["support"] == [4, 0, 0, 0, 0, 0, 0, 0]
    assert result["splits"]["validation"]["class_counts"]["NV"] == 2
    assert result["splits"]["test"]["support"] == [0, 0, 1, 1, 0, 0, 0, 0]
    seq = analyze_class_distribution([0, 1], ["train", "val"])
    assert seq["split_counts"]["validation"] == 1


@pytest.mark.parametrize("labels", [[True], [-1], [8], [0.5]])
def test_invalid_labels_do_not_silently_change_inventory(labels):
    with pytest.raises(ValueError):
        analyze_class_distribution(labels)


def test_assignment_missing_extra_and_unknown_split_are_rejected():
    with pytest.raises(ValueError, match="exactly"):
        analyze_class_distribution({"a": 0}, {"b": "train"})
    with pytest.raises(ValueError, match="align"):
        analyze_class_distribution([0, 1], ["train"])
    with pytest.raises(ValueError, match="split"):
        analyze_class_distribution([0], ["development"])
    assert analyze_class_distribution([np.int64(0)])["overall"]["num_images"] == 1


def test_metadata_null_tokens_coverage_and_numeric_values_without_mutation():
    rows = [{"image": "a", "lesion_id": "nan", "sex": "female", "age_approx": "20"},
            {"image": "b", "lesion_id": " NULL ", "sex": "unknown", "age_approx": "40"},
            {"image": "b", "lesion_id": "L1", "sex": None, "age_approx": "NaN"}]
    before = copy.deepcopy(rows)
    summary = analyze_metadata(rows)
    assert rows == before
    assert not summary["patient_id_column_present"]
    assert summary["columns"]["lesion_id"]["missing"] == 2
    assert summary["columns"]["sex"]["missing"] == 2
    assert summary["columns"]["age_approx"]["numeric"] == {"minimum": 20, "median": 30, "maximum": 40}
    assert summary["duplicate_image_id_count"] == 1
    assert summary["duplicate_image_row_count"] == 1
    assert analyze_metadata([])["num_rows"] == 0


def test_group_overlap_and_transitive_connected_components_are_reported():
    records = [ImageRecord("a", Path("a.jpg"), patient_id="P1", lesion_id="L1"),
               ImageRecord("b", Path("b.jpg"), patient_id="P1", lesion_id="L2"),
               ImageRecord("c", Path("c.jpg"), lesion_id="L2")]
    summary = analyze_dataset(records, {"a": 0, "b": 1, "c": 2},
                              {"a": "train", "b": "train", "c": "test"})
    qa = summary["grouping"]
    assert qa["patient_id"]["overlapping_groups"] == 0
    assert qa["lesion_id"]["overlapping_groups"] == 1
    assert qa["connected_components"] == {"count": 1, "largest_size": 3, "cross_split_components": 1}
    assert not qa["group_disjoint"]


def test_missing_group_placeholders_never_form_a_shared_cross_split_group():
    rows = [{"image_id": "a", "patient_id": "unknown", "lesion_id": "nan"},
            {"image_id": "b", "patient_id": " NULL ", "lesion_id": None}]
    summary = analyze_dataset(rows, {"a": 0, "b": 1}, {"a": "train", "b": "test"})
    assert summary["grouping"]["group_disjoint"]
    assert summary["grouping"]["patient_id"]["unique_groups"] == 0
    assert not summary["grouping"]["patient_independence_established"]


def test_predeclared_minority_uses_only_frozen_train_counts_and_fixed_ties():
    ids = tuple(f"i{i}" for i in range(8))
    records = [ImageRecord(i, Path(i)) for i in ids]
    summary = analyze_dataset(records, dict(zip(ids, range(8), strict=True)), {i: "train" for i in ids})
    minority = summary["minority_definition"]
    assert minority["classes"] == list(CLASSES[:3])
    assert minority["tie_break"] == "fixed_class_order"
    assert not minority["allowed_as_selector_input"]
    assert minority["purpose"] == "future_evaluation_reporting_only"


def test_preparation_label_availability_is_separate_from_charged_annotations():
    records, labels, assignments = records_and_truth()
    summary = analyze_dataset(records, labels, assignments)
    accounting = summary["annotation_availability"]
    assert accounting["simulator_train_labels_available"] == 4
    assert accounting["acquired_train_labels"] == accounting["charged_train_annotations"] == 0
    assert accounting["validation_labels_available"] == accounting["validation_annotation_cost_if_all_used"] == 2
    assert accounting["validation_annotations_charged_at_preparation"] == 0
    assert not accounting["validation_labels_used_for_acquisition"]
    assert not accounting["test_labels_used_for_model_or_acquisition"]
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match="universes"):
        analyze_dataset(records, {"missing": 0}, assignments)


def test_actual_summary_exports_csv_totals_and_standalone_log_count_figures(tmp_path):
    records, labels, assignments = records_and_truth()
    summary = analyze_dataset(records, labels, assignments,
                              [{"image": r.image_id, "sex": "unknown"} for r in records])
    paths = export_analysis(summary, tmp_path, dataset_name="Synthetic QA fixture")
    saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert saved["overall"]["support"] == [4, 2, 1, 1, 0, 0, 0, 0]
    with Path(paths["class_csv"]).open(newline="", encoding="utf-8") as stream:
        assert sum(int(row["count"]) for row in csv.DictReader(stream)) == 8
    with Path(paths["split_class_csv"]).open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 24
    for name in ("class_counts_png", "split_counts_png"):
        assert Path(paths[name]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for name in ("class_counts_svg", "split_counts_svg"):
        assert Path(paths[name]).read_text(encoding="utf-8").lstrip().startswith("<?xml")
