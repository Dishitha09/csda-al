from dataclasses import replace
import csv

import numpy as np
import pytest
from PIL import Image

from csda_al.config import load_config
from csda_al.data.metadata import CLASSES, ImageRecord, load_metadata, load_truth
from csda_al.data.split import grouped_split, save_split, apply_split
from csda_al.data.dataset import ImageDataset, LabeledImageDataset
from csda_al.data.oracle import LabelOracle
from csda_al.engine.budget import AnnotationBudget
from csda_al.engine.pool_state import PoolState


def records_at(tmp_path, n=12):
    records = []
    for i in range(n):
        p = tmp_path / f"image_{i}.png"
        Image.fromarray(np.full((12, 12, 3), i, dtype=np.uint8)).save(p)
        records.append(ImageRecord(f"image_{i}", p))
    return records


def test_named_csv_mapping_and_image_only(tmp_path):
    records_at(tmp_path, 3)
    records = load_metadata(tmp_path)
    assert len(records) == 3 and not hasattr(records[0], "label")
    path = tmp_path / "truth.csv"
    order = tuple(reversed(CLASSES))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", *order])
        writer.writeheader()
        writer.writerow({"image": "image_0", **{c: int(c == "BCC") for c in order}})
    assert load_truth(path) == {"image_0": 2}
    assert len(ImageDataset(records)[0]) == 2
    with pytest.raises(ValueError, match="acquired_only"):
        LabeledImageDataset(records, {r.image_id: 0 for r in records})


def test_group_transitivity_content_and_roundtrip(tmp_path):
    records = records_at(tmp_path)
    records[0] = replace(records[0], patient_id="p")
    records[1] = replace(records[1], patient_id="p", lesion_id="l")
    records[2] = replace(records[2], lesion_id="l")
    records[3].image_path.write_bytes(records[4].image_path.read_bytes())
    result, manifest = grouped_split(records, seed=5)
    assert len({r.split for r in result[:3]}) == 1
    assert result[3].split == result[4].split
    result2, manifest2 = grouped_split(list(reversed(records)), seed=5)
    assert manifest["assignments"] == manifest2["assignments"]
    path = tmp_path / "split.json"
    save_split(path, manifest)
    restored, _ = apply_split(records, path)
    assert result == restored


def test_oracle_all_or_nothing_replay_and_restart(tmp_path):
    ledger = tmp_path / "ledger.json"
    oracle = LabelOracle({"a": 0, "b": 7, "c": 2}, max_annotations=2, ledger_path=ledger)
    with pytest.raises(ValueError):
        oracle.annotate(["a", "test"], request_id="bad")
    assert oracle.unique_annotations == 0 and not ledger.exists()
    receipt = oracle.annotate(["a"], request_id="one")
    assert receipt.acquired_labels == (0,) and receipt.new_unique_cost == 1
    replay = oracle.annotate(["a"], request_id="one")
    assert replay.new_unique_cost == 0 and oracle.unique_annotations == 1
    restored = LabelOracle({"a": 0, "b": 7, "c": 2}, max_annotations=2, ledger_path=ledger)
    assert restored.annotate(["a"], request_id="one").new_unique_cost == 0
    with pytest.raises(ValueError):
        restored.annotate(["a"], request_id="different")
    with pytest.raises(ValueError):
        restored.annotate(["b"], request_id="one")
    with pytest.raises(ValueError):
        restored.annotate(["b", "c"], request_id="over")
    assert restored.unique_annotations == 1


def test_budget_and_state_invariants():
    budget = AnnotationBudget(10, validation_annotation_cost=3, total_label_cap=8)
    assert budget.remaining(4) == 1
    with pytest.raises(ValueError):
        budget.remaining(6)
    state = PoolState(("a", "b", "c")).acquire(("a",))
    assert set(state.unlabeled_ids) == {"b", "c"}
    assert PoolState.from_dict(state.to_dict()) == state
    with pytest.raises(ValueError):
        state.acquire(("a",))
    with pytest.raises(ValueError):
        PoolState(("a", "b"), ("a",))


def test_config_strict_and_relative_paths(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  image_dir: images\nexperiment:\n  max_annotations: 10\n", encoding="utf-8")
    config = load_config(path)
    assert config.dataset.image_dir == str(tmp_path / "images")
    assert len(config.fingerprint) == 64
    path.write_text("experiment:\n  made_up: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("experiment:\n  max_annotations: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_truth_and_missing_data(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_metadata(tmp_path / "absent")
    path = tmp_path / "truth.csv"
    path.write_text("image," + ",".join(CLASSES) + "\na,0,0,0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="one-hot"):
        load_truth(path)
