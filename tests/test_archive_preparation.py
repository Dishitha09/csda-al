"""Archive and neutral-image checks without any model training."""
import csv
import importlib.util
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from csda_al.data.archive import extract_and_scan
from csda_al.data.dataset import ImageDataset
from csda_al.data.metadata import CLASSES, ImageRecord
from csda_al.data.duplicates import image_fingerprints


def tiny_archive(path, count=2, *, corrupt=False, extra=None):
    gt = io.StringIO(newline="")
    writer = csv.writer(gt)
    writer.writerow(["image", *CLASSES, "UNK"])
    meta = io.StringIO(newline="")
    metadata = csv.writer(meta)
    metadata.writerow(["image", "lesion_id"])
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(count):
            image_id = f"ISIC_{i:07d}"
            writer.writerow([image_id, *[int(c == i % 8) for c in range(8)], 0])
            metadata.writerow([image_id, f"lesion_{i}"])
            data = io.BytesIO()
            Image.fromarray(np.random.default_rng(i).integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(data, format="JPEG")
            archive.writestr(f"{CLASSES[i % 8]}/{image_id}.jpg", b"not a JPEG" if corrupt and i == 0 else data.getvalue())
        archive.writestr("ISIC_2019_Training_GroundTruth.csv", gt.getvalue())
        archive.writestr("ISIC_2019_Training_Metadata.csv", meta.getvalue())
        if extra:
            archive.writestr(*extra)


def test_full_decode_crc_preserves_bytes_and_removes_class_paths(tmp_path):
    archive = tmp_path / "source.zip"
    tiny_archive(archive)
    rows, summary = extract_and_scan(archive, tmp_path / "prepared", workers=2)
    assert summary["full_image_decode_count"] == 2
    assert summary["crc_checked_members"] == 4 and summary["invalid_images"] == 0
    with zipfile.ZipFile(archive) as source:
        for i, row in enumerate(rows):
            path = tmp_path / "prepared" / row["relative_path"]
            assert path.parent.name == "images"
            assert not set(Path(row["relative_path"]).parts).intersection(CLASSES)
            assert path.read_bytes() == source.read(f"{CLASSES[i]}/{row['image_id']}.jpg")
    # Same source can be rechecked without overwriting a different file.
    target = tmp_path / "prepared/images/ISIC_0000000.jpg"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="invalid"):
        extract_and_scan(archive, tmp_path / "prepared", workers=1)
    assert target.read_bytes() == b"changed"


def test_unsafe_and_duplicate_members_are_rejected(tmp_path):
    archive = tmp_path / "source.zip"
    tiny_archive(archive, extra=("../outside.jpg", b"bad"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_and_scan(archive, tmp_path / "safe")
    assert not (tmp_path / "outside.jpg").exists()
    archive2 = tmp_path / "duplicate.zip"
    tiny_archive(archive2, extra=("NV/ISIC_0000000.jpg", b"bad"))
    with pytest.raises(ValueError, match="duplicate image ID"):
        extract_and_scan(archive2, tmp_path / "safe2")


def test_corrupt_image_has_no_success_summary(tmp_path):
    archive = tmp_path / "bad.zip"
    tiny_archive(archive, corrupt=True)
    with pytest.raises(ValueError, match="invalid"):
        extract_and_scan(archive, tmp_path / "dataset", workers=1)
    assert (tmp_path / "dataset/analysis/invalid_images.json").is_file()
    assert not (tmp_path / "dataset/analysis/archive_scan_summary.json").exists()


def test_exif_loader_matches_fingerprint_orientation(tmp_path):
    path = tmp_path / "ISIC_exif.jpg"
    image = Image.new("RGB", (30, 20), (110, 20, 70))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    record = ImageRecord("ISIC_exif", path)
    loaded, image_id = ImageDataset([record])[0]
    with Image.open(path) as original:
        info = image_fingerprints(original)
    assert image_id == "ISIC_exif"
    assert loaded.size == (20, 30) == (info["width"], info["height"])


def test_conservative_components_retain_unconfirmed_pairs(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_isic2019.py"
    spec = importlib.util.spec_from_file_location("stage1_prepare_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"image_id": i} for i in ("a", "b", "c", "d")]
    duplicates = {"exact_byte_groups": [["a", "b"]], "exact_pixel_groups": [],
                  "near_candidates": [{"image_id1": "b", "image_id2": "c", "confirmed_duplicate": False}]}
    groups, summary = module.leakage_components(rows, duplicates)
    assert groups["a"] == groups["b"] == groups["c"] != groups["d"]
    assert summary["images_dropped"] == 0 and summary["largest_component"] == 3
