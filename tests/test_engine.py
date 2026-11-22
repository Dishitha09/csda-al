"""Real tiny CPU wiring and crash/replay checks; NO REAL ISIC TRAINING."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from csda_al.config import Config
from csda_al.data.dataset import ImageDataset
from csda_al.data.oracle import LabelOracle
from csda_al.data.synthetic import create_synthetic_dataset
from csda_al.data.transforms import build_image_transform
from csda_al.engine.active_loop import ActiveLearningEngine
from csda_al.features import FeatureStore, extract_features
from csda_al.models import Classifier
from csda_al.training import SupervisedTrainer
from csda_al.utils.io import read_json
from csda_al.utils.seed import seed_everything


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FailsAfterOneUpdate(SupervisedTrainer):
    def fit(self, loader, epochs=1, max_steps=None):
        super().fit(loader, epochs=epochs, max_steps=max_steps)
        raise RuntimeError("injected training failure after optimizer mutation")


def make_engine(root, run_name="run", *, trainer_class=SupervisedTrainer,
                validation_size=8, extract=False, diversity=False):
    seed_everything(19)
    records, truth = create_synthetic_dataset(root / "images", seed=17, train_size=12,
                                             validation_size=validation_size, test_size=0, image_size=32)
    train = [r for r in records if r.split == "train"]
    validation = [r for r in records if r.split == "validation"]
    default = Config()
    config = replace(
        default,
        dataset=replace(default.dataset, image_size=32),
        classifier=replace(default.classifier, epochs=1, batch_size=2, dropout=0.1),
        acquisition=replace(default.acquisition, initial_budget=2, batch_budget=2, k_neighbors=3,
                            diversity=diversity),
        experiment=replace(default.experiment, seed=19, output_dir=str(root / run_name),
                           max_rounds=2, max_annotations=4, num_workers=0),
    )
    model = Classifier(dropout=config.classifier.dropout)
    if extract:
        store = extract_features(model, DataLoader(ImageDataset(train, build_image_transform(32)), batch_size=4),
                                 encoder_version="synthetic_untrained_resnet18", split_fingerprint="split-fixture")
    else:
        # Embeddings are generated independently of truth; actual ResNet extraction
        # is covered by the first end-to-end test and model/feature tests.
        values = np.random.default_rng(41).normal(size=(len(train), 12)).astype(np.float32)
        store = FeatureStore(tuple(r.image_id for r in train), values,
                             encoder_version="synthetic_generated", split_fingerprint="split-fixture")
    oracle = LabelOracle({r.image_id: truth[r.image_id] for r in train}, max_annotations=4,
                         ledger_path=root / run_name / "oracle_ledger.json")
    trainer = trainer_class(model, lr=config.classifier.lr, freeze_stage="head")
    engine = ActiveLearningEngine(config, train, validation, {r.image_id: truth[r.image_id] for r in validation},
                                  store, oracle, trainer, synthetic=True)
    return engine, oracle


def assert_same_model(actual, expected):
    for name, value in expected.trainer.model.state_dict().items():
        torch.testing.assert_close(actual.trainer.model.state_dict()[name], value, rtol=0, atol=0, msg=name)


def test_synthetic_two_rounds_real_resnet_and_paid_validation(tmp_path):
    engine, oracle = make_engine(tmp_path, extract=True, diversity=True)
    summary = engine.run(max_train_steps=1)
    assert summary["synthetic"] and summary["no_real_training"]
    assert summary["rounds"] == summary["training_steps"] == 2
    assert summary["training_annotations"] == oracle.unique_annotations == 4
    assert summary["validation_annotations"] == 8 and summary["total_annotation_cost"] == 12
    assert summary["history"][0]["method"] == "supclust"
    assert summary["history"][1]["method"] == "difficulty_aware"
    assert summary["history"][1]["diagnostics"]["configuration"]["diversity_weight"] == 0.25
    queried = [i for batch in summary["history"] for i in batch["image_ids"]]
    assert len(queried) == len(set(queried)) == 4 and all("_train_" in i for i in queried)
    state = read_json(engine.output / "state.json")
    assert state["phase"] == "ready" and len(state["acquired_labels"]) == 4
    reports = sorted((engine.output / "validation").glob("round_*.json"))
    assert len(reports) == 2
    result = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert result["provenance"] == "synthetic_demonstration_no_real_training"
    assert len(result["metrics"]["confusion_matrix"]) == 8
    resumed, resumed_oracle = make_engine(tmp_path, extract=True, diversity=True)
    repeated = resumed.run(resume=True, max_train_steps=1)
    assert repeated["history"] == summary["history"] and resumed_oracle.unique_annotations == 4
    assert_same_model(resumed, engine)


@pytest.mark.parametrize("fault", ["after_oracle_before_commit", "training_after_update"])
def test_resume_replays_paid_annotations_without_double_charge_and_same_weights(tmp_path, monkeypatch, fault):
    baseline, _ = make_engine(tmp_path, "baseline")
    expected = baseline.run(max_train_steps=1)
    trainer_class = FailsAfterOneUpdate if fault == "training_after_update" else SupervisedTrainer
    interrupted, interrupted_oracle = make_engine(tmp_path, "interrupted", trainer_class=trainer_class)
    if fault == "after_oracle_before_commit":
        from csda_al.engine import active_loop
        original = active_loop.atomic_json

        def fail_on_pool_commit(path, value):
            if str(path).endswith("state.json") and value.get("phase") == "training_pending":
                raise RuntimeError("injected failure after durable oracle before pool commit")
            return original(path, value)

        with monkeypatch.context() as patch:
            patch.setattr(active_loop, "atomic_json", fail_on_pool_commit)
            with pytest.raises(RuntimeError, match="after durable oracle"):
                interrupted.run(max_train_steps=1)
        assert read_json(interrupted.output / "state.json")["phase"] == "prepared"
    else:
        with pytest.raises(RuntimeError, match="after optimizer mutation"):
            interrupted.run(max_train_steps=1)
        assert read_json(interrupted.output / "state.json")["phase"] == "training_pending"
    assert interrupted_oracle.unique_annotations == 2
    resumed, oracle = make_engine(tmp_path, "interrupted")
    result = resumed.run(resume=True, max_train_steps=1)
    assert oracle.unique_annotations == result["training_annotations"] == 4
    assert result["training_steps"] == 2
    assert [q["image_ids"] for q in result["history"]] == [q["image_ids"] for q in expected["history"]]
    assert_same_model(resumed, baseline)


def test_resume_completes_validation_export_without_retraining(tmp_path, monkeypatch):
    from csda_al.engine import active_loop
    engine, _ = make_engine(tmp_path)
    with monkeypatch.context() as patch:
        def fail_export(*args, **kwargs):
            raise RuntimeError("injected evaluation export failure")
        patch.setattr(active_loop, "export_result", fail_export)
        with pytest.raises(RuntimeError, match="export failure"):
            engine.run(max_train_steps=1)
    state = read_json(engine.output / "state.json")
    assert state["phase"] == "evaluation_pending" and state["trainer_steps"] == 1
    resumed, oracle = make_engine(tmp_path)
    result = resumed.run(resume=True, max_train_steps=1)
    assert result["training_steps"] == 2 and oracle.unique_annotations == 4
    assert (resumed.output / "validation" / "round_0001.json").exists()
    assert (resumed.output / "validation" / "round_0002.json").exists()


def test_no_validation_has_explicit_entropy_fallback(tmp_path):
    engine, _ = make_engine(tmp_path, validation_size=0)
    result = engine.run(max_train_steps=1)
    assert result["validation_annotations"] == 0
    assert result["history"][1]["method"] == "entropy"
    events = (engine.output / "events.jsonl").read_text(encoding="utf-8")
    assert "no_validation_fallback" in events


def test_resume_rejects_checkpoint_tamper_or_changed_training_limit(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.run(max_train_steps=1)
    resumed, _ = make_engine(tmp_path)
    with pytest.raises(ValueError, match="training-step limit"):
        resumed.run(resume=True, max_train_steps=2)
    state = read_json(engine.output / "state.json")
    with (engine.output / state["checkpoint"]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        resumed.run(resume=True, max_train_steps=1)


@pytest.mark.parametrize("changed", ["validation_labels", "feature_provenance"])
def test_resume_rejects_changed_validation_labels_and_feature_provenance(tmp_path, changed):
    engine, _ = make_engine(tmp_path)
    engine.run(max_train_steps=1)
    resumed, _ = make_engine(tmp_path)
    labels = dict(resumed.validation_labels)
    store = resumed.store
    if changed == "validation_labels":
        first = next(iter(labels))
        labels[first] = (labels[first] + 1) % 8
    else:
        store = replace(store, preprocessing_id="different-transform-identity")
    resumed = ActiveLearningEngine(resumed.config, tuple(resumed.records.values()), resumed.validation_records,
                                   labels, store, resumed.oracle, resumed.trainer, synthetic=True)
    with pytest.raises(ValueError, match="provenance"):
        resumed.run(resume=True, max_train_steps=1)


def test_persistent_engine_requires_durable_oracle(tmp_path):
    engine, _ = make_engine(tmp_path)
    oracle = LabelOracle({i: 0 for i in engine.records}, max_annotations=4)
    with pytest.raises(ValueError, match="durable oracle"):
        ActiveLearningEngine(engine.config, tuple(engine.records.values()), engine.validation_records,
                             engine.validation_labels, engine.store, oracle, engine.trainer, synthetic=True)


def test_resume_rejects_equal_cost_ledger_with_different_annotated_ids(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.run(max_train_steps=1)
    acquired = set(read_json(engine.output / "state.json")["pool"]["labeled_ids"])
    other_ids = tuple(i for i in engine.records if i not in acquired)[:4]
    wrong_ledger = LabelOracle({i: 0 for i in engine.records}, max_annotations=4,
                              ledger_path=tmp_path / "unrelated_oracle.json")
    wrong_ledger.annotate(other_ids, request_id="unrelated")
    assert wrong_ledger.unique_annotations == 4
    resumed = ActiveLearningEngine(engine.config, tuple(engine.records.values()), engine.validation_records,
                                   engine.validation_labels, engine.store, wrong_ledger, engine.trainer, synthetic=True)
    with pytest.raises(ValueError, match="Oracle ledger"):
        resumed.run(resume=True, max_train_steps=1)
