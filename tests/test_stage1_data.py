"""Stage 1 preparation, duplicate grouping and runtime truth isolation tests."""
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from csda_al.data.metadata import ImageRecord, load_metadata, normalize_group_id
from csda_al.data.oracle import LabelOracle
from csda_al.data.prepared import (
    PUBLIC_FIELDS, load_bundle_manifest, load_prepared_records, load_validation_truth, prepare_bundle,
)
from csda_al.data.split import apply_split, grouped_split, save_split
from csda_al.utils.io import atomic_json, fingerprint, read_json


def _records(root, count=12):
    images = root / "images"
    images.mkdir(parents=True)
    records = []
    for index in range(count):
        path = images / f"image_{index:03d}.png"
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(path)
        records.append(ImageRecord(f"image_{index:03d}", path))
    return records


def _bundle(tmp_path):
    root = tmp_path / "dataset"
    records = _records(root)
    records, manifest = grouped_split(records, seed=19, fractions=(0.5, 0.25, 0.25))
    truth = {record.image_id: index % 8 for index, record in enumerate(records)}
    bundle = root / "prepared"
    prepare_bundle(records, truth, manifest, bundle, root)
    return root, bundle, records, truth, manifest


@pytest.mark.parametrize("token", [None, "", "  ", "nan", " NaN ", "None", "NULL", "na", "n/a", "<NA>", "NaT", "unknown", "UNK"])
def test_declared_missing_group_tokens_are_missing_not_shared_ids(token):
    assert normalize_group_id(token) is None
    record = ImageRecord("safe_id", "safe_id.png", patient_id=token, lesion_id=token)
    assert record.patient_id is None and record.lesion_id is None
    assert normalize_group_id("  patient_17 ") == "patient_17"


def test_literal_missing_patient_tokens_do_not_collapse_dataset(tmp_path):
    records = _records(tmp_path)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "patient_id", "lesion_id"])
        writer.writeheader()
        writer.writerows({"image": record.image_id, "patient_id": "nan", "lesion_id": "unknown"} for record in records)
    records = load_metadata(tmp_path / "images", metadata)
    _, manifest = grouped_split(records, seed=7, fractions=(0.5, 0.25, 0.25))
    assert manifest["counts"] == {"train": 6, "validation": 3, "test": 3}
    assert manifest["patient_ids_available"] == manifest["lesion_ids_available"] == 0


@pytest.mark.parametrize("image_id", ["../escape", "folder/image", "C:drive", " has_space", "has space", "..", ""])
def test_image_ids_cannot_smuggle_paths_or_whitespace(image_id):
    with pytest.raises(ValueError, match="safe identifier"):
        ImageRecord(image_id, "image.png")


def test_all_linkage_sources_form_transitive_disjoint_groups_and_roundtrip(tmp_path):
    records = _records(tmp_path, 8)
    records[0] = replace(records[0], patient_id="patient_a")
    records[1] = replace(records[1], patient_id="patient_a", lesion_id="lesion_a")
    records[2] = replace(records[2], lesion_id="lesion_a")
    extra = {records[2].image_id: "confirmed_duplicate_1", records[3].image_id: "confirmed_duplicate_1"}
    records[4].image_path.write_bytes(records[5].image_path.read_bytes())
    hashes = {record.image_id: hashlib.sha256(record.image_path.read_bytes()).hexdigest() for record in records}
    result, manifest = grouped_split(records, seed=2, additional_groups=extra, precomputed_sha256=hashes)
    assert len({record.split for record in result[:4]}) == 1
    assert result[4].split == result[5].split
    assert manifest["schema_version"] == 3 and manifest["content_checked"] is True
    assert manifest["additional_groups"] == extra
    path = tmp_path / "split.json"
    save_split(path, manifest)
    restored, same = apply_split(records, path)
    assert restored == result and same == manifest
    _, reordered = grouped_split(list(reversed(records)), seed=2, additional_groups=extra, precomputed_sha256=hashes)
    assert reordered["fingerprint"] == manifest["fingerprint"]
    with pytest.raises(ValueError, match="mandatory"):
        grouped_split(records, check_content=False)
    with pytest.raises(ValueError, match="groups changed"):
        apply_split(records, path, additional_groups={})
    records[0].image_path.write_bytes(records[6].image_path.read_bytes())
    with pytest.raises(ValueError, match="dataset fingerprint"):
        apply_split(records, path)


def test_invalid_duplicate_maps_hash_coverage_and_split_counts_rejected(tmp_path):
    records = _records(tmp_path)
    with pytest.raises(ValueError, match="known IDs"):
        grouped_split(records, additional_groups={"not_present": "g"})
    with pytest.raises(ValueError, match="Precomputed SHA256"):
        grouped_split(records, precomputed_sha256={records[0].image_id: "x"})
    _, manifest = grouped_split(records)
    manifest["counts"]["train"] += 1
    manifest["fingerprint"] = fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"})
    path = tmp_path / "bad_split.json"
    save_split(path, manifest)
    with pytest.raises(ValueError, match="counts"):
        apply_split(records, path)


def test_prepared_public_manifests_are_label_free_and_truth_stores_are_disjoint(tmp_path):
    _, bundle, records, truth, manifest = _bundle(tmp_path)
    info = load_bundle_manifest(bundle)
    assert info["split_fingerprint"] == manifest["fingerprint"]
    assert info["validation_annotation_cost"] == 3
    public_ids = {}
    for split in ("train", "validation", "test"):
        loaded = load_prepared_records(bundle, split, verify_images=True)
        public_ids[split] = {record.image_id for record in loaded}
        assert all(record.split == split and not hasattr(record, "label") for record in loaded)
        with (bundle / "public" / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert tuple(reader.fieldnames) == PUBLIC_FIELDS
            assert all(row["relative_path"].startswith("images/") for row in reader)
    assert set.union(*public_ids.values()) == set(truth)
    assert not public_ids["train"] & public_ids["validation"]
    assert not public_ids["train"] & public_ids["test"]
    assert not public_ids["validation"] & public_ids["test"]
    validation = load_validation_truth(bundle)
    assert validation == {record.image_id: truth[record.image_id] for record in records if record.split == "validation"}
    oracle = LabelOracle.from_prepared_train(bundle, max_annotations=2, ledger_path=tmp_path / "ledger.json")
    first_id = next(iter(public_ids["train"]))
    receipt = oracle.annotate([first_id])
    assert receipt.acquired_labels == (truth[first_id],)
    assert oracle.unique_annotations == 1
    with pytest.raises(ValueError, match="TRAIN"):
        oracle.annotate([next(iter(public_ids["test"]))])


def test_runtime_train_and_validation_providers_never_open_sealed_test_truth(tmp_path, monkeypatch):
    _, bundle, _, _, _ = _bundle(tmp_path)
    original = Path.open
    opened = []

    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        if path.name == "test_labels.csv":
            raise AssertionError("Final-test truth opened during runtime preparation loading")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        load_prepared_records(bundle, "train")
        load_prepared_records(bundle, "validation")
        load_validation_truth(bundle)
        oracle = LabelOracle.from_prepared_train(bundle)
        assert oracle.unique_annotations == 0
    assert not any(path.endswith("test_labels.csv") for path in opened)


def test_public_image_paths_cannot_escape_root_even_with_recomputed_manifest_hash(tmp_path):
    _, bundle, _, _, _ = _bundle(tmp_path)
    csv_path = bundle / "public" / "train.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["relative_path"] = "../outside.jpg"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    info = read_json(bundle / "bundle.json")
    info["public_manifests"]["train"]["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    info["fingerprint"] = fingerprint({key: value for key, value in info.items() if key != "fingerprint"})
    atomic_json(bundle / "bundle.json", info)
    with pytest.raises(ValueError, match="neutral"):
        load_prepared_records(bundle, "train")


def test_private_checksum_failure_and_wrong_split_role_fail_before_label_reveal(tmp_path):
    _, bundle, _, _, _ = _bundle(tmp_path)
    path = bundle / "private" / "train_oracle.csv"
    path.write_text(path.read_text(encoding="utf-8") + "forged,0,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        LabelOracle.from_prepared_train(bundle)
    # Validation remains independently loadable and never needs TRAIN truth.
    assert len(load_validation_truth(bundle)) == 3


def test_existing_bundle_or_diagnostic_directory_is_not_silently_published(tmp_path):
    root, bundle, records, truth, manifest = _bundle(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        prepare_bundle(records, truth, manifest, bundle, root)
    outside = tmp_path / "not_within_dataset"
    with pytest.raises(ValueError, match="direct child"):
        prepare_bundle(records, truth, manifest, outside, root)
    bad_root = tmp_path / "bad_dataset"
    records = _records(bad_root)
    diagnosis_folder = bad_root / "MEL"
    diagnosis_folder.mkdir()
    path = diagnosis_folder / records[0].image_path.name
    path.write_bytes(records[0].image_path.read_bytes())
    records[0] = replace(records[0], image_path=path)
    records, manifest = grouped_split(records)
    truth = {record.image_id: 0 for record in records}
    with pytest.raises(ValueError, match="neutral"):
        prepare_bundle(records, truth, manifest, bad_root / "prepared", bad_root)


def _prepared_config(tmp_path, bundle):
    from csda_al.config import Config
    raw_truth = tmp_path / "full_ground_truth_must_not_open.csv"
    raw_truth.write_text("DO NOT READ FULL TRUTH AT RUNTIME\n", encoding="utf-8")
    default = Config()
    return replace(default, dataset=replace(default.dataset, prepared_dir=str(bundle),
                   ground_truth_csv=str(raw_truth)),
                   experiment=replace(default.experiment, output_dir=str(tmp_path / "run"))), raw_truth


def test_prepared_training_inputs_open_only_train_oracle_and_paid_validation(tmp_path, monkeypatch):
    from csda_al import cli
    _, bundle, _, truth, _ = _bundle(tmp_path)
    config, raw_truth = _prepared_config(tmp_path, bundle)
    original = Path.open
    opened = []

    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        if path.resolve() == raw_truth.resolve() or path.name in {"test.csv", "test_labels.csv"}:
            raise AssertionError("TRAIN/validation preparation opened test/full truth inputs")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        train, validation, validation_truth, oracle, split_hash = cli.prepared_training_inputs(config)
    assert len(train) == 6 and len(validation) == len(validation_truth) == 3
    assert set(validation_truth) == {record.image_id for record in validation}
    assert all(validation_truth[image_id] == truth[image_id] for image_id in validation_truth)
    assert oracle.unique_annotations == 0
    assert split_hash == load_bundle_manifest(bundle)["split_fingerprint"]
    assert any(path.endswith("train_oracle.csv") for path in opened)
    assert any(path.endswith("validation_labels.csv") for path in opened)


def test_ssl_data_boundary_is_train_only_and_never_opens_any_target_store(tmp_path, monkeypatch):
    from csda_al import cli
    _, bundle, _, _, _ = _bundle(tmp_path)
    config, raw_truth = _prepared_config(tmp_path, bundle)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if (path.resolve() == raw_truth.resolve() or "private" in path.parts
                or path.name in {"validation.csv", "test.csv"}):
            raise AssertionError("SSL data boundary opened targets or held-out manifest")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        train, targets, _ = cli._data(config, synthetic=False, truth_needed=False)
    assert len(train) == 6 and all(record.split == "train" for record in train)
    assert targets == {}


def test_prepared_inspect_reads_public_metadata_without_private_labels_or_models(tmp_path, monkeypatch, capsys):
    from csda_al import cli
    _, bundle, _, _, _ = _bundle(tmp_path)
    config, raw_truth = _prepared_config(tmp_path, bundle)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.resolve() == raw_truth.resolve() or "private" in path.parts:
            raise AssertionError("Inspection opened private target data")
        return original(path, *args, **kwargs)

    def no_model(*args, **kwargs):
        raise AssertionError("Inspection instantiated a training model")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        patch.setattr(cli, "SimCLR", no_model)
        patch.setattr(cli, "Classifier", no_model)
        assert cli.main(["inspect-data", "--config", str(config_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["images"] == 12
    assert result["truth_labels_loaded"] is False
    assert result["split_counts"] == {"train": 6, "validation": 3, "test": 3}


def test_real_prepared_run_reaches_only_reviewed_data_boundary_before_model_call(tmp_path, monkeypatch):
    from csda_al import cli
    _, bundle, _, _, _ = _bundle(tmp_path)
    config, raw_truth = _prepared_config(tmp_path, bundle)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.resolve() == raw_truth.resolve() or path.name in {"test.csv", "test_labels.csv"}:
            raise AssertionError("AL runtime opened test/full truth inputs")
        return original(path, *args, **kwargs)

    class StopBeforeModel(RuntimeError):
        pass

    def stop_representation(*args, **kwargs):
        raise StopBeforeModel("Data boundary reached; no model/training executed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        patch.setattr(cli, "_representation", stop_representation)
        args = SimpleNamespace(command="run-al", config=str(config_path), output=None,
                               dry_run=False, synthetic=False, resume=False)
        with pytest.raises(StopBeforeModel, match="no model/training"):
            cli._dispatch(args)
    assert not (Path(config.experiment.output_dir) / "oracle_ledger.json").exists()


def test_optional_pretraining_input_integrity_check_detects_changed_image_bytes(tmp_path):
    _, bundle, _, _, _ = _bundle(tmp_path)
    train = load_prepared_records(bundle, "train")
    Image.new("RGB", (8, 8), "white").save(train[0].image_path)
    with pytest.raises(ValueError, match="byte checksum"):
        load_prepared_records(bundle, "train", verify_images=True)
