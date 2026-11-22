"""Check the deliverable contract against disposable architecture fixtures."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def _packager():
    script = Path(__file__).resolve().parents[1] / "scripts" / "package_project.py"
    spec = importlib.util.spec_from_file_location("csda_packaging_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_project(tmp_path, packager):
    root = tmp_path / "fixture_source"
    for relative in packager.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Disposable packaging-contract fixture\n", encoding="utf-8")
    return root


def test_archive_excludes_data_weights_caches_and_matches_manifest(tmp_path):
    packager = _packager()
    root = _fixture_project(tmp_path, packager)
    forbidden = [
        ".test-tmp-models/run/model.pt", ".venv/secret.txt", "src/csda_al.egg-info/PKG-INFO",
        "outputs/pool.json", "src/csda_al/__pycache__/module.pyc", "weights.PTH",
        "somewhere/features.npz", "image.PNG", "data/labels.csv", ".env",
        "reference_code/source.py", "paper.pdf",
    ]
    for relative in forbidden:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Excluded fixture\n", encoding="utf-8")
    (root / "docs" / "diagram.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    output = tmp_path / "contract.zip"
    report = packager.build_archive(root, output)
    assert report["crc"] == report["manifest_hashes"] == report["extracted_hashes"] == "PASS"
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert all(name.startswith("csda_al/") for name in names)
        assert "csda_al/docs/diagram.svg" in names
        assert not any(f"csda_al/{relative}" in names for relative in forbidden)
        manifest = json.loads(archive.read("csda_al/ARCHIVE_MANIFEST.json"))
        assert {entry["member"] for entry in manifest["files"]} == names - {"csda_al/ARCHIVE_MANIFEST.json"}
        assert manifest["self_hash_included"] is False
    assert not (root / "ARCHIVE_MANIFEST.json").exists()
    assert not list(tmp_path.glob("csda_verify_*"))


def test_manifest_detects_valid_crc_payload_tampering(tmp_path):
    packager = _packager()
    root = _fixture_project(tmp_path, packager)
    output = tmp_path / "original.zip"
    packager.build_archive(root, output)
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(changed, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "csda_al/README.md":
                data += b"Payload changed after manifest creation\n"
            target.writestr(info, data)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        packager.verify_archive(changed)


def test_missing_required_file_cannot_replace_existing_deliverable(tmp_path):
    packager = _packager()
    root = _fixture_project(tmp_path, packager)
    (root / "src" / "csda_al" / "data" / "oracle.py").unlink()
    output = tmp_path / "existing.zip"
    output.write_bytes(b"Existing artifact remains untouched")
    with pytest.raises(ValueError, match="missing required files"):
        packager.build_archive(root, output)
    assert output.read_bytes() == b"Existing artifact remains untouched"
    with pytest.raises(ValueError, match="outside"):
        packager.build_archive(root, root / "invalid.zip")
