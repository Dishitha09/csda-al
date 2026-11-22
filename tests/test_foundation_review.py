"""Independent integrity checks for split identities, pool state, and costs."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from csda_al.data.metadata import ImageRecord, load_metadata
from csda_al.data.oracle import LabelOracle
from csda_al.data.split import apply_split, grouped_split, save_split
from csda_al.engine.pool_state import PoolState
from csda_al.utils.io import atomic_json, fingerprint, read_json


def _records(tmp_path, count=8):
    records = []
    for index in range(count):
        path = tmp_path / f"image_{index}.png"
        Image.fromarray(np.full((10, 10, 3), index, dtype=np.uint8)).save(path)
        records.append(ImageRecord(f"image_{index}", path))
    return records


def test_metadata_enumerates_real_files_once_regardless_of_extension_case(tmp_path):
    Image.new("RGB", (10, 10), "red").save(tmp_path / "sample.PNG")
    records = load_metadata(tmp_path)
    assert len(records) == 1
    assert records[0].image_id == "sample"
    assert records[0].image_path.name == "sample.PNG"
    Image.new("RGB", (10, 10), "blue").save(tmp_path / "sample.jpg")
    with pytest.raises(ValueError, match="exactly one"):
        load_metadata(tmp_path)


def test_saved_split_detects_changed_content_with_same_ids(tmp_path):
    records = _records(tmp_path)
    _, manifest = grouped_split(records)
    path = tmp_path / "split.json"
    save_split(path, manifest)
    Image.new("RGB", (10, 10), "red").save(records[0].image_path)
    with pytest.raises(ValueError, match="dataset fingerprint"):
        apply_split(records, path)


def test_saved_split_detects_changed_group_metadata_and_duplicate_input(tmp_path):
    records = _records(tmp_path)
    _, manifest = grouped_split(records)
    path = tmp_path / "split.json"
    save_split(path, manifest)
    changed = [replace(records[0], patient_id="new-patient"), *records[1:]]
    with pytest.raises(ValueError, match="dataset fingerprint"):
        apply_split(changed, path)
    with pytest.raises(ValueError, match="unique"):
        apply_split([*records, records[0]], path)


def test_split_fingerprint_is_portable_across_order_and_file_location(tmp_path):
    records = _records(tmp_path)
    _, first = grouped_split(records, seed=1)
    target = tmp_path / "relocated"
    target.mkdir()
    copied = []
    for record in reversed(records):
        path = target / record.image_path.name
        path.write_bytes(record.image_path.read_bytes())
        copied.append(replace(record, image_path=path))
    _, second = grouped_split(copied, seed=1)
    assert first["fingerprint"] == second["fingerprint"]


@pytest.mark.parametrize("field,value", [("query_round", False), ("query_round", 0.0), ("state_version", True), ("state_version", 1.5)])
def test_pool_counters_reject_boolean_or_fractional_values(field, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        PoolState(("a", "b"), **{field: value})
    with pytest.raises(ValueError, match="empty"):
        PoolState(("a",), selected_history=((),), query_round=1, state_version=1)


@pytest.mark.parametrize("cost_name,wrong_value", [("new_unique_cost", 100), ("cumulative_training_cost", 100)])
def test_oracle_rejects_self_consistent_but_impossible_ledger_costs(tmp_path, cost_name, wrong_value):
    path = tmp_path / "oracle.json"
    oracle = LabelOracle({"a": 0, "b": 1}, ledger_path=path)
    oracle.annotate(["a"], request_id="first")
    ledger = read_json(path)
    receipt = ledger["first"]
    receipt[cost_name] = wrong_value
    receipt["receipt_hash"] = fingerprint({key: value for key, value in receipt.items() if key != "receipt_hash"})
    atomic_json(path, ledger)
    with pytest.raises(ValueError, match="cost"):
        LabelOracle({"a": 0, "b": 1}, ledger_path=path)


def test_oracle_reload_uses_cumulative_cost_not_sorted_request_key_order(tmp_path):
    path = tmp_path / "oracle.json"
    truth = {"a": 0, "b": 1, "c": 2}
    oracle = LabelOracle(truth, ledger_path=path)
    oracle.annotate(["a"], request_id="z-first")
    oracle.annotate(["b", "c"], request_id="a-second")
    reloaded = LabelOracle(truth, ledger_path=path)
    assert reloaded.unique_annotations == 3
    assert reloaded.annotate(["a"], request_id="z-first").new_unique_cost == 0


def test_stale_oracle_instance_cannot_overwrite_new_receipts(tmp_path):
    path = tmp_path / "oracle.json"
    truth = {"a": 0, "b": 1}
    first = LabelOracle(truth, ledger_path=path)
    stale = LabelOracle(truth, ledger_path=path)
    first.annotate(["a"], request_id="first")
    with pytest.raises(ValueError, match="changed"):
        stale.annotate(["b"], request_id="second")
    assert LabelOracle(truth, ledger_path=path).unique_annotations == 1


def test_oracle_train_id_contract_rejects_duplicates_and_empty_request_identity():
    with pytest.raises(ValueError, match="unique"):
        LabelOracle({"a": 0}, train_ids=("a", "a"))
    oracle = LabelOracle({"a": 0})
    with pytest.raises(ValueError, match="request_id"):
        oracle.annotate(["a"], request_id="")
    assert oracle.unique_annotations == 0
