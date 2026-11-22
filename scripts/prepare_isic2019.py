"""Stage 1 privileged preparation; never pretrains or fits a model."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from csda_al.data.analysis import analyze_dataset, export_analysis
from csda_al.data.archive import extract_and_scan, file_sha256
from csda_al.data.duplicates import detect_duplicate_groups
from csda_al.data.metadata import CLASSES, load_metadata, load_truth
from csda_al.data.prepared import prepare_bundle, load_bundle_manifest, load_prepared_records
from csda_al.data.split import grouped_split, save_split, apply_split
from csda_al.utils.io import atomic_json


def leakage_components(inventory: list[dict], duplicates: dict) -> tuple[dict[str, str], dict]:
    """Conservatively keep every perceptual candidate together without dropping IDs.

    Perceptual edges are not asserted to prove identical lesions. Their use only
    overconstrains grouping to avoid a potential cross-split leak.
    """
    parent = {row["image_id"]: row["image_id"] for row in inventory}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)

    for key in ("exact_byte_groups", "exact_pixel_groups"):
        for members in duplicates[key]:
            for image_id in members[1:]:
                union(members[0], image_id)
    for pair in duplicates["near_candidates"]:
        union(pair["image_id1"], pair["image_id2"])
    mapping = {image_id: "leakage_" + find(image_id) for image_id in sorted(parent)}
    sizes = Counter(mapping.values())
    return mapping, {"component_count": len(sizes), "non_singleton_components": sum(n > 1 for n in sizes.values()),
                     "images_in_non_singleton_components": sum(n for n in sizes.values() if n > 1),
                     "largest_component": max(sizes.values()),
                     "component_size_histogram": dict(sorted(Counter(sizes.values()).items())),
                     "near_policy": "conservatively_group_all_unconfirmed_dual_hash_candidates",
                     "all_image_ids_retained": True, "images_dropped": 0}


def prepare(archive: Path, output: Path, *, workers: int, reuse_scan: bool, seed: int) -> dict:
    output = output.resolve()
    analysis = output / "analysis"
    if reuse_scan:
        summary = json.loads((analysis / "archive_scan_summary.json").read_text(encoding="utf-8"))
        if archive.stat().st_size != summary["archive_bytes"] or file_sha256(archive) != summary["archive_sha256"]:
            raise ValueError("Source archive differs from the completed scan")
        with (analysis / "image_inventory.csv").open(newline="", encoding="utf-8") as stream:
            inventory = list(csv.DictReader(stream))
        for row in inventory:
            for key in ("width", "height", "size_bytes", "exif_orientation"):
                row[key] = int(row[key])
            row["valid"] = row["valid"].lower() == "true"
        if not all(r["valid"] for r in inventory) or len(inventory) != summary["image_count"]:
            raise ValueError("Scan inventory is incomplete")
        print(f"Reusing completed CRC/decode scan of {len(inventory)} images", flush=True)
    else:
        inventory, summary = extract_and_scan(archive, output, workers=workers,
                                               progress=lambda message: print(message, flush=True))
    private = output / "private_source"
    records = load_metadata(output / "images", private / "ISIC_2019_Training_Metadata.csv")
    with (private / "ISIC_2019_Training_Metadata.csv").open(newline="", encoding="utf-8-sig") as stream:
        metadata = list(csv.DictReader(stream))
    if len(metadata) != len(records) or {r.image_id for r in records} != {r["image_id"] for r in inventory}:
        raise ValueError("Metadata rows, image IDs and scan IDs do not match exactly")
    print("Detecting exact and unconfirmed perceptual duplicate groups without labels", flush=True)
    duplicates = detect_duplicate_groups(inventory, near_threshold=4)
    groups, component_summary = leakage_components(inventory, duplicates)
    atomic_json(analysis / "duplicate_screening.json", duplicates)
    atomic_json(analysis / "duplicate_component_summary.json", component_summary)
    hashes = {r["image_id"]: r["byte_sha256"] for r in inventory}
    split_records, manifest = grouped_split(records, seed=seed, fractions=(.7, .15, .15),
                                             additional_groups=groups, precomputed_sha256=hashes)
    save_split(analysis / "split_manifest.json", manifest)
    restored, _ = apply_split(records, analysis / "split_manifest.json",
                              additional_groups=groups, precomputed_sha256=hashes)
    if restored != split_records:
        raise ValueError("Saved split did not round-trip deterministically")
    assignments = {r.image_id: r.split for r in split_records}
    # All assignments are frozen before truth is opened. Only privileged
    # preparation/reporting and role-specific private stores may read targets.
    truth = load_truth(private / "ISIC_2019_Training_GroundTruth.csv")
    if set(truth) != set(assignments):
        raise ValueError("Truth IDs must match every image/metadata ID exactly")
    with (private / "archive_members.csv").open(newline="", encoding="utf-8") as stream:
        provenance = list(csv.DictReader(stream))
    folder_mismatches = [row["image_id"] for row in provenance
                         if Path(row["source_member"]).parts[0] != CLASSES[truth[row["image_id"]]]]
    if folder_mismatches:
        raise ValueError("Source class folders disagree with official one-hot targets")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for image_id, group in groups.items():
        group_splits[group].add(assignments[image_id])
    if any(len(markers) != 1 for markers in group_splits.values()):
        raise ValueError("Duplicate/potential-duplicate component crosses a split")
    conflicts = []
    for members in duplicates["exact_pixel_groups"]:
        if len({truth[i] for i in members}) > 1:
            conflicts.append({"image_ids": members, "class_indices": [truth[i] for i in members]})
    atomic_json(analysis / "exact_pixel_label_conflicts.json", conflicts)
    stats = analyze_dataset(split_records, truth, assignments, metadata_raw=metadata)
    stats["source_archive"] = summary
    stats["duplicates"] = {"exact_byte_groups": len(duplicates["exact_byte_groups"]),
                            "exact_pixel_groups": len(duplicates["exact_pixel_groups"]),
                            "unconfirmed_near_pairs": len(duplicates["near_candidates"]),
                            "exact_pixel_conflicting_label_groups": len(conflicts),
                            "source_class_folder_target_mismatches": len(folder_mismatches),
                            "cross_split_duplicate_components": 0, **component_summary}
    paths = export_analysis(stats, analysis)
    bundle = output / "prepared"
    if not bundle.exists():
        prepare_bundle(split_records, truth, manifest, bundle, output)
    elif load_bundle_manifest(bundle)["split_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Existing frozen bundle differs; use a new output directory")
    print("Verifying role manifests and every prepared image SHA-256", flush=True)
    counts = {s: len(load_prepared_records(bundle, s, verify_images=True))
              for s in ("train", "validation", "test")}
    if counts != stats["split_counts"]:
        raise ValueError("Prepared role manifests differ from frozen split counts")
    result = {"status": "PASS", "stage": 1, "real_model_training": False,
              "images": len(records), "split_counts": counts,
              "split_fingerprint": manifest["fingerprint"], "bundle": str(bundle),
              "outputs": paths, "duplicates": stats["duplicates"],
              "validation_cost_if_all_labels_used": counts["validation"],
              "acquired_train_annotations": 0}
    atomic_json(analysis / "preparation_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-scan", action="store_true", help="Require a prior completed scan and matching archive hash")
    args = parser.parse_args()
    result = prepare(args.archive, args.output, workers=args.workers, reuse_scan=args.reuse_scan, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
