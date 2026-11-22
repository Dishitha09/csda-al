"""Model, feature-cache, and frozen-backbone behavioral checks."""
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from csda_al.engine.checkpointing import load_checkpoint, save_checkpoint
from csda_al.features import FeatureStore, extract_features, nearest_neighbors
from csda_al.models import Classifier, SimCLR
from csda_al.training import SupervisedTrainer


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Images(Dataset):
    split = "train"
    acquired_only = True

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(index))
        return (image, index % 2, f"im-{index}") if self.labeled else (image, f"im-{index}")


def test_feature_store_alignment_immutability_and_roundtrip(tmp_path):
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    store = FeatureStore(("b", "a"), values, encoder_version="sha123")
    values[0] = 99
    assert store.rows(("a", "b")).tolist() == [[3.0, 4.0], [1.0, 2.0]]
    with pytest.raises(ValueError):
        store.features[0] = 0
    with pytest.raises(KeyError):
        store.rows(("unknown",))
    with pytest.raises(ValueError):
        store.rows(("a", "a"))
    path = tmp_path / "features.npz"
    store.save(path)
    restored = FeatureStore.load(path, expected_encoder_version="sha123")
    assert restored.digest == store.digest
    with pytest.raises(ValueError, match="Stale"):
        FeatureStore.load(path, expected_encoder_version="other")
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    manifest = json.loads(payload["manifest"].item())
    manifest["sha256"] = "bad"
    payload["manifest"] = np.asarray(json.dumps(manifest))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        FeatureStore.load(path)


@pytest.mark.parametrize("ids,features,kwargs", [
    (("a", "a"), np.ones((2, 2)), {}),
    (("a",), np.asarray([[float("nan")]]), {}),
    (("a",), np.zeros((1, 2)), {"normalization": "l2"}),
    (("a",), np.ones((2, 2)), {}),
])
def test_invalid_feature_store(ids, features, kwargs):
    with pytest.raises(ValueError):
        FeatureStore(ids, features, **kwargs)


def test_classifier_head_fit_preserves_frozen_encoder_and_buffers(tmp_path):
    torch.manual_seed(8)
    model = Classifier()
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    head_before = model.head.weight.detach().clone()
    trainer = SupervisedTrainer(model, freeze_stage="head")
    stats = trainer.fit(DataLoader(Images(), batch_size=2), max_steps=1)
    assert stats["steps"] == 1 and np.isfinite(stats["loss"])
    for name, value in model.backbone.state_dict().items():
        assert torch.equal(before[name], value), name
    assert not torch.equal(head_before, model.head.weight)
    ids, probs = trainer.predict(DataLoader(Images(labeled=False), batch_size=2))
    assert ids == ("im-0", "im-1", "im-2", "im-3") and probs.shape == (4, 8)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
    features = extract_features(model, DataLoader(Images(labeled=False), batch_size=2))
    assert features.features.shape == (4, 512)
    assert not features.features.flags.writeable
    trainer.set_freeze_stage("layer4")
    assert model.backbone.network.layer4[0].conv1.weight.requires_grad
    assert not model.backbone.network.layer1[0].conv1.weight.requires_grad
    trainer.set_freeze_stage("all")
    assert all(p.requires_grad for p in model.parameters())
    path = tmp_path / "classifier.pt"
    save_checkpoint(path, model, metadata={"pool_version": 1}, optimizer=trainer.optimizer)
    restored = Classifier()
    load_checkpoint(path, restored, expected_metadata={"pool_version": 1})
    restored.eval()
    model.eval()
    with torch.no_grad():
        image = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(model(image), restored(image))


def test_supervised_boundary_rejects_unacquired_or_validation():
    trainer = SupervisedTrainer(Classifier())
    data = Images()
    data.acquired_only = False
    with pytest.raises(ValueError, match="acquired"):
        trainer.fit(DataLoader(data, batch_size=2), max_steps=1)
    data.acquired_only = True
    data.split = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        trainer.fit(DataLoader(data, batch_size=2), max_steps=1)


def test_ssl_initialization_and_nearest_neighbor_self_exclusion():
    ssl = SimCLR()
    model = Classifier()
    model.initialize_from_ssl(ssl)
    assert torch.equal(ssl.backbone.network.conv1.weight, model.backbone.network.conv1.weight)
    x = np.asarray([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    distances, indices = nearest_neighbors(x, 9)
    assert distances.shape == indices.shape == (3, 2)
    assert all(i not in row for i, row in enumerate(indices))
    d, indices = nearest_neighbors(x[:1], 2)
    assert d.shape == indices.shape == (1, 0)
