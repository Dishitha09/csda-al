"""Privileged dataset-preparation statistics, never acquisition inputs.

These functions may inspect the supplied truth CSV only after a label-independent
split has been frozen. They report dataset observations, not model results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from .metadata import CLASSES, SPLITS, ImageRecord, normalize_group_id
from ..utils.io import atomic_json


def _label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Dataset labels must be integer class indices 0..7")
    if not 0 <= value < len(CLASSES):
        raise ValueError("Dataset label index is outside the eight ISIC classes")
    return int(value)


def _split(value: Any) -> str:
    value = "validation" if value == "val" else value
    if value not in SPLITS:
        raise ValueError(f"Unknown dataset split: {value}")
    return str(value)


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=len(CLASSES))
    total = int(counts.sum())
    positive = counts[counts > 0]
    proportions = counts / total if total else np.zeros(len(CLASSES))
    entropy = float(-sum(p*math.log2(p) for p in proportions if p > 0)) if total else None
    maximum = int(counts.max())
    minimum = int(counts.min())
    min_positive = int(positive.min()) if len(positive) else None
    return {
        "num_images": total, "num_declared_classes": len(CLASSES),
        "class_order": list(CLASSES), "support": counts.tolist(),
        "class_counts": {name: int(counts[i]) for i, name in enumerate(CLASSES)},
        "per_class": [{"class_index": i, "class_name": name, "count": int(counts[i]),
                       "proportion": float(proportions[i])} for i, name in enumerate(CLASSES)],
        "observed_classes": [CLASSES[i] for i in np.flatnonzero(counts)],
        "missing_classes": [CLASSES[i] for i in np.flatnonzero(counts == 0)],
        "minimum_class_count": minimum, "minimum_positive_class_count": min_positive,
        "maximum_class_count": maximum,
        "imbalance_ratio": maximum/minimum if minimum else None,
        "imbalance_ratio_observed": maximum/min_positive if min_positive else None,
        "imbalance_ratio_policy": "maximum/minimum across eight classes; null if minimum is zero",
        "shannon_entropy_bits": entropy,
        "normalized_entropy": entropy/math.log2(len(CLASSES)) if entropy is not None else None,
        "effective_class_count": 2**entropy if entropy is not None else None,
    }


def analyze_class_distribution(
    labels: Mapping[str, int] | Sequence[int],
    assignments: Mapping[str, str] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Count fixed-order classes and optional splits; no label-based splitting.

    A mapping uses persistent image IDs; a sequence can use aligned split names.
    Missing/extra mapping IDs fail instead of silently changing denominators.
    Empty observations have undefined entropy/ratios serialized as null.
    """
    if isinstance(labels, Mapping):
        ids = tuple(labels)
        if any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError("Label mapping requires nonempty string image IDs")
        values = [_label(labels[i]) for i in ids]
    else:
        values = [_label(v) for v in labels]
        ids = tuple(str(i) for i in range(len(values)))
    result: dict[str, Any] = {"overall": _distribution(values), "splits": {}, "split_counts": {}}
    if assignments is None:
        return result
    if isinstance(assignments, Mapping):
        if set(assignments) != set(ids):
            raise ValueError("Split assignments must cover exactly the label IDs")
        split_values = [_split(assignments[i]) for i in ids]
    else:
        if len(assignments) != len(values):
            raise ValueError("Split names and labels must align")
        split_values = [_split(v) for v in assignments]
    for split in SPLITS:
        group = [label for label, marker in zip(values, split_values, strict=True) if marker == split]
        result["splits"][split] = _distribution(group)
        result["split_counts"][split] = len(group)
    return result


def _metadata_value(value: Any) -> str | None:
    return normalize_group_id(None if value is None else str(value))


def analyze_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report normalized missingness, category/numeric summaries and ID coverage.

    It uses the same explicit null-token policy as grouping. Missing group IDs
    remain missing; this function never infers patient identity from demographics.
    """
    columns = sorted({str(key) for row in rows for key in row})
    summary: dict[str, Any] = {}
    for name in columns:
        normalized = [_metadata_value(row.get(name)) for row in rows]
        observed = [v for v in normalized if v is not None]
        counts = Counter(observed)
        missing = len(rows)-len(observed)
        details: dict[str, Any] = {
            "observed": len(observed), "missing": missing,
            "missing_fraction": missing/len(rows) if rows else None,
            "unique_observed_values": len(counts),
        }
        if name not in {"image", "image_id", "patient_id", "lesion_id"}:
            details["most_common_values"] = [
                {"value": value, "count": count}
                for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ]
        numeric: list[float] = []
        for value in observed:
            try:
                number = float(value)
            except (TypeError, ValueError):
                break
            if not math.isfinite(number):
                break
            numeric.append(number)
        if observed and len(numeric) == len(observed):
            details["numeric"] = {"minimum": min(numeric), "median": float(np.median(numeric)),
                                  "maximum": max(numeric)}
        summary[name] = details
    image_ids = [_metadata_value(row.get("image", row.get("image_id"))) for row in rows]
    available = [i for i in image_ids if i is not None]
    duplicates = {i: n for i, n in Counter(available).items() if n > 1}
    return {"num_rows": len(rows), "columns": summary,
            "patient_id_column_present": "patient_id" in columns,
            "lesion_id_column_present": "lesion_id" in columns,
            "image_ids_missing": len(rows)-len(available),
            "image_ids_unique": len(set(available)),
            "duplicate_image_id_count": len(duplicates),
            "duplicate_image_row_count": sum(n-1 for n in duplicates.values()),
            "missing_policy": "strip whitespace; metadata.normalize_group_id explicit null tokens"}


def _record_value(record: ImageRecord | Mapping[str, Any], name: str) -> Any:
    return record.get(name) if isinstance(record, Mapping) else getattr(record, name)


def _grouping_qa(records: Sequence[ImageRecord | Mapping[str, Any]], assignments: Mapping[str, str]) -> dict:
    ids = [str(_record_value(record, "image_id")) for record in records]
    parent = {i: i for i in ids}

    def find(image_id: str) -> str:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    report: dict[str, Any] = {}
    for field in ("patient_id", "lesion_id"):
        groups: dict[str, list[str]] = defaultdict(list)
        for record in records:
            group_id = _metadata_value(_record_value(record, field))
            if group_id is not None:
                groups[group_id].append(str(_record_value(record, "image_id")))
        overlaps = []
        group_counts_by_split: Counter = Counter()
        for group_id, members in groups.items():
            markers = sorted({assignments[i] for i in members})
            group_counts_by_split.update(markers)
            if len(markers) > 1:
                overlaps.append({"group_id": group_id, "splits": markers, "num_images": len(members)})
            first = members[0]
            for member in members[1:]:
                parent[find(member)] = find(first)
        observed_count = sum(map(len, groups.values()))
        report[field] = {
            "images_with_group_id": observed_count, "images_missing_group_id": len(records)-observed_count,
            "unique_groups": len(groups), "groups_by_split": dict(group_counts_by_split),
            "overlapping_groups": len(overlaps), "overlap_examples": sorted(overlaps, key=lambda x: str(x["group_id"]))[:20],
        }
    components: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        components[find(image_id)].append(image_id)
    component_overlaps = sum(len({assignments[i] for i in members}) > 1 for members in components.values())
    report["connected_components"] = {"count": len(components),
                                       "largest_size": max(map(len, components.values()), default=0),
                                       "cross_split_components": component_overlaps}
    report["group_disjoint"] = component_overlaps == 0
    report["patient_independence_established"] = False
    report["patient_independence_note"] = (
        "Known patient/lesion groups checked; unavailable patient IDs cannot establish patient independence."
    )
    return report


def analyze_dataset(
    records: Sequence[ImageRecord | Mapping[str, Any]],
    label_mapping: Mapping[str, int],
    split_assignments: Mapping[str, str],
    metadata_raw: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze a frozen dataset under the privileged preparation boundary.

    Full class counts are permitted for inventory/reporting only. This output,
    including the minority definition, must never be supplied to a selector.
    """
    ids = [str(_record_value(record, "image_id")) for record in records]
    if len(ids) != len(set(ids)) or any(not i.strip() or i == "None" for i in ids):
        raise ValueError("Dataset analysis requires unique nonempty record IDs")
    if set(ids) != set(label_mapping) or set(ids) != set(split_assignments):
        raise ValueError("Record, label and split ID universes must match exactly")
    assignments = {i: _split(split_assignments[i]) for i in ids}
    summary = analyze_class_distribution(label_mapping, assignments)
    summary["schema_version"] = 1
    summary["provenance"] = "actual_dataset_inventory_no_model_training"
    summary["access_boundary"] = "privileged_preparation_and_reporting_only_never_selector_input"
    summary["grouping"] = _grouping_qa(records, assignments)
    if metadata_raw is not None:
        summary["metadata"] = analyze_metadata(metadata_raw)
        raw_ids = {_metadata_value(row.get("image", row.get("image_id"))) for row in metadata_raw}
        raw_ids.discard(None)
        summary["metadata"]["manifest_ids_without_metadata"] = len(set(ids)-raw_ids)
        summary["metadata"]["metadata_ids_outside_manifest"] = len(raw_ids-set(ids))
    else:
        summary["metadata"] = None
    train_counts = summary["splits"]["train"]["support"]
    minority = sorted(range(len(CLASSES)), key=lambda i: (train_counts[i], i))[:3]
    summary["minority_definition"] = {
        "policy": "bottom_three_classes_by_frozen_train_count", "source_split": "train",
        "class_indices": minority, "classes": [CLASSES[i] for i in minority],
        "counts": [train_counts[i] for i in minority], "tie_break": "fixed_class_order",
        "includes_zero_support_classes": True, "purpose": "future_evaluation_reporting_only",
        "allowed_as_selector_input": False,
    }
    summary["annotation_availability"] = {
        "simulator_train_labels_available": summary["split_counts"]["train"],
        "acquired_train_labels": 0, "charged_train_annotations": 0,
        "validation_labels_available": summary["split_counts"]["validation"],
        "validation_labels_used_for_acquisition": False,
        "validation_annotation_cost_if_all_used": summary["split_counts"]["validation"],
        "validation_annotations_charged_at_preparation": 0,
        "test_labels_available_for_locked_assessment": summary["split_counts"]["test"],
        "test_labels_used_for_model_or_acquisition": False,
        "cost_policy": "availability is not supervision use; charge validation labels once if later used",
    }
    return summary


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_analysis(summary: Mapping[str, Any], output_dir: str | Path,
                    *, dataset_name: str = "ISIC 2019") -> dict[str, str]:
    """Export actual-count JSON/CSVs and standalone log-count PNG/SVG figures.

    The caller chooses a privileged output directory. Figures contain no model
    metrics; zero counts are annotated as zero rather than fabricated positive bars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"json": folder/"dataset_analysis.json", "class_csv": folder/"class_distribution.csv",
             "split_class_csv": folder/"split_class_distribution.csv", "split_csv": folder/"split_distribution.csv"}
    atomic_json(paths["json"], dict(summary))
    _csv(paths["class_csv"], summary["overall"]["per_class"])
    split_rows = [{"split": split, **row} for split, stats in summary["splits"].items()
                  for row in stats["per_class"]]
    _csv(paths["split_class_csv"], split_rows)
    total = summary["overall"]["num_images"]
    _csv(paths["split_csv"], [{"split": split, "count": count,
                              "proportion": count/total if total else 0.0}
                             for split, count in summary["split_counts"].items()])
    if summary.get("metadata") is not None:
        paths["metadata_csv"] = folder/"metadata_missingness.csv"
        _csv(paths["metadata_csv"], [{"column": name, "observed": stats["observed"],
                                     "missing": stats["missing"], "missing_fraction": stats["missing_fraction"],
                                     "unique_observed_values": stats["unique_observed_values"]}
                                    for name, stats in summary["metadata"]["columns"].items()])

    def figure(labels: Sequence[str], values: Sequence[int], stem: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(labels))
        heights = [float(v) if v else float("nan") for v in values]
        ax.bar(x, heights, color="#246B8E", width=.65)
        ax.set_yscale("log")
        tick_formatter = ScalarFormatter()
        tick_formatter.set_scientific(False)
        tick_formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(tick_formatter)
        ax.yaxis.set_minor_formatter(NullFormatter())
        positive = [v for v in values if v > 0]
        if positive:
            ax.set_ylim(max(.5, min(positive)*.5), max(positive)*2)
        else:
            ax.set_ylim(.5, 2)
        for i, value in enumerate(values):
            y = value if value > 0 else ax.get_ylim()[0]
            ax.annotate(f"{value:,}", (i, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_xlim(-.6, len(labels)-.4)
        ax.set_ylabel("Images (log scale)")
        ax.set_title(f"{dataset_name}: {title}")
        ax.grid(axis="y", which="major", alpha=.25)
        ax.set_axisbelow(True)
        fig.text(.5, .015, "Dataset inventory only - no model training or performance results", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for extension in ("png", "svg"):
            key = f"{stem}_{extension}"
            paths[key] = folder/f"{stem}.{extension}"
            fig.savefig(paths[key], dpi=180)
        plt.close(fig)

    figure(summary["overall"]["class_order"], summary["overall"]["support"], "class_counts", "class counts")
    if summary["split_counts"]:
        figure(list(summary["split_counts"]), list(summary["split_counts"].values()), "split_counts", "frozen split counts")
    return {key: str(path) for key, path in paths.items() if path.exists()}
