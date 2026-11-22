"""Synthetic CPU checks only: these are not representation benchmarks."""
import math

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from csda_al.data.transforms import build_ssl_transform
from csda_al.engine.checkpointing import load_checkpoint, save_checkpoint
from csda_al.models import SimCLR, nt_xent_loss
from csda_al.training import train_ssl


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_nt_xent_known_value_and_pairing():
    same = torch.ones(2, 3, requires_grad=True)
    loss = nt_xent_loss(same, same, temperature=0.2)
    assert loss.item() == pytest.approx(math.log(3))
    loss.backward()
    assert torch.isfinite(same.grad).all()
    points = torch.eye(3)
    assert nt_xent_loss(points, points) < nt_xent_loss(points, points.roll(1, 0))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_nt_xent_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        nt_xent_loss(torch.ones(2, 3), torch.ones(2, 3), temperature)


def test_nt_xent_rejects_no_negatives_or_nonfinite():
    with pytest.raises(ValueError):
        nt_xent_loss(torch.ones(1, 3), torch.ones(1, 3))
    with pytest.raises(ValueError):
        nt_xent_loss(torch.full((2, 3), float("nan")), torch.ones(2, 3))


def test_two_view_transform_is_independent_and_seeded():
    pixels = np.random.default_rng(7).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    transform = build_ssl_transform(32)
    torch.manual_seed(11)
    a, b = transform(image)
    torch.manual_seed(11)
    aa, bb = transform(image)
    assert a.shape == b.shape == (3, 32, 32)
    assert not torch.equal(a, b)
    assert torch.equal(a, aa) and torch.equal(b, bb)


class Views(Dataset):
    split = "train"

    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        image = torch.rand((3, 32, 32), generator=generator)
        return (image, image.flip(-1)), f"ssl-{index}"


def test_resnet_ssl_one_step_and_checkpoint(tmp_path):
    torch.manual_seed(3)
    model = SimCLR(projection_dim=16)
    before = model.projection_head[-1].weight.detach().clone()
    output = train_ssl(model, DataLoader(Views(), batch_size=2), max_steps=1)
    assert output["steps"] == 1 and output["target_labels_used"] == 0
    assert math.isfinite(output["loss"])
    assert not torch.equal(before, model.projection_head[-1].weight)
    model.eval()
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(images)
        assert model.encode(images).shape == (2, 512)
        assert expected.shape == (2, 16)
    path = tmp_path / "ssl.pt"
    save_checkpoint(path, model, metadata={"synthetic": True})
    restored = SimCLR(projection_dim=16).eval()
    assert load_checkpoint(path, restored) == {"synthetic": True}
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(images))


def test_ssl_rejects_target_bearing_batches():
    model = SimCLR(projection_dim=8)
    with pytest.raises(ValueError, match="never targets"):
        train_ssl(model, [(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]), ("a", "b"))], max_steps=1)
