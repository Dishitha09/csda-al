"""Package and verify the original CSDA-AL architecture without data or weights.

Run from any directory: ``python path/to/scripts/package_project.py --run-tests``.
The default destination is beside the project directory. Verification reads the
archive, checks CRC and SHA256, extracts into an isolated temporary directory,
checks imported module locations, and optionally runs the extracted tests.
Only Python's standard library is needed to build the ZIP; importing/testing the
extracted software uses the current interpreter and its installed dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import zipfile

ARCHIVE_ROOT = "csda_al"
MANIFEST_NAME = f"{ARCHIVE_ROOT}/ARCHIVE_MANIFEST.json"
DEFAULT_ARCHIVE_NAME = "CSDA_AL_Complete_Architecture.zip"

REQUIRED_FILES = tuple("""
README.md
pyproject.toml
requirements.txt
requirements-lock.txt
LICENSE_NOTES.md
.gitignore
ARCHITECTURE_COMPLETION_REPORT.md
configs/dataset.yaml
configs/simclr.yaml
configs/classifier.yaml
configs/acquisition.yaml
configs/experiment.yaml
src/csda_al/__init__.py
src/csda_al/config.py
src/csda_al/cli.py
src/csda_al/data/metadata.py
src/csda_al/data/split.py
src/csda_al/data/dataset.py
src/csda_al/data/transforms.py
src/csda_al/data/oracle.py
src/csda_al/models/backbone.py
src/csda_al/models/simclr.py
src/csda_al/models/classifier.py
src/csda_al/features/extract.py
src/csda_al/features/cache.py
src/csda_al/features/nearest_neighbors.py
src/csda_al/acquisition/base.py
src/csda_al/acquisition/random.py
src/csda_al/acquisition/entropy.py
src/csda_al/acquisition/coreset.py
src/csda_al/acquisition/typiclust.py
src/csda_al/acquisition/supclust.py
src/csda_al/acquisition/bald.py
src/csda_al/acquisition/difficulty_aware.py
src/csda_al/imbalance/class_difficulty.py
src/csda_al/imbalance/probability_weighting.py
src/csda_al/imbalance/diagnostics.py
src/csda_al/engine/pool_state.py
src/csda_al/engine/budget.py
src/csda_al/engine/active_loop.py
src/csda_al/engine/checkpointing.py
src/csda_al/training/ssl_train.py
src/csda_al/training/supervised_train.py
src/csda_al/training/finetune.py
src/csda_al/evaluation/metrics.py
src/csda_al/evaluation/learning_curves.py
src/csda_al/evaluation/report.py
src/csda_al/utils/seed.py
src/csda_al/utils/logging.py
src/csda_al/utils/io.py
tests/test_dataset.py
tests/test_no_label_leakage.py
tests/test_simclr.py
tests/test_supclust.py
tests/test_difficulty.py
tests/test_acquisition.py
tests/test_engine.py
tests/test_metrics.py
tests/test_cli_smoke.py
tests/test_packaging.py
examples/synthetic_smoke_test.py
examples/sample_config.yaml
docs/ARCHITECTURE.md
docs/RESEARCH_TRACEABILITY.md
docs/DATA_CONTRACTS.md
docs/IMPLEMENTATION_DECISIONS.md
docs/EVALUATION_PROTOCOL.md
docs/RESEARCH_LIMITATIONS.md
docs/MODULE_GUIDE.md
docs/diagrams/csda_al.mmd
scripts/package_project.py
""".split())

EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "__pycache__", "node_modules", "build", "dist", "outputs", "output",
    "reference_resources", "reference_code", ".idea", ".vscode", ".ipynb_checkpoints",
})
EXCLUDED_ROOT_DIRECTORIES = frozenset({"data", "datasets", "artifacts", "tmp", "temp", "audit_work"})
EXCLUDED_EXTENSIONS = frozenset({
    ".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".npy",
    ".h5", ".hdf5", ".pkl", ".pickle", ".onnx", ".bin", ".dcm", ".dicom",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".ipynb", ".pdf", ".pem", ".key",
    ".tmp", ".lockfile",
})


def _excluded_directory(name: str) -> bool:
    lower = name.casefold()
    return (
        lower in EXCLUDED_DIRECTORIES or lower.startswith((".venv", ".test-tmp"))
        or lower.endswith((".egg-info", ".dist-info"))
    )


def should_include(relative_path: str | Path) -> bool:
    """Return whether a relative source file belongs to the architecture ZIP."""
    path = PurePosixPath(str(relative_path).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"..", "."} or ":" in part for part in path.parts):
        return False
    if path.parts[0].casefold() in EXCLUDED_ROOT_DIRECTORIES:
        return False
    if any(_excluded_directory(part) for part in path.parts[:-1]):
        return False
    name = path.name.casefold()
    if name == "archive_manifest.json" or name == ".env" or name.startswith((".env.", "credentials", "secrets")):
        return False
    if name in {".run.lock", "thumbs.db", ".ds_store", "cachedir.tag"}:
        return False
    return path.suffix.casefold() not in EXCLUDED_EXTENSIONS


def _payload(project_dir: Path) -> dict[str, bytes]:
    payload = {}
    for current, directories, filenames in os.walk(project_dir, topdown=True, followlinks=False):
        folder = Path(current)
        directories[:] = sorted(
            name for name in directories
            if not _excluded_directory(name)
            and not (folder == project_dir and name.casefold() in EXCLUDED_ROOT_DIRECTORIES)
            and not (folder / name).is_symlink()
        )
        for name in sorted(filenames):
            path = folder / name
            relative = path.relative_to(project_dir).as_posix()
            if not should_include(relative):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(project_dir):
                raise ValueError(f"Refusing source link outside the project: {relative}")
            payload[f"{ARCHIVE_ROOT}/{relative}"] = path.read_bytes()
    return dict(sorted(payload.items()))


def _check_required(names: Iterable[str], required_files: Iterable[str]) -> None:
    missing = {f"{ARCHIVE_ROOT}/{path}" for path in required_files} - set(names)
    if missing:
        raise ValueError("Project is incomplete; missing required files: " + ", ".join(sorted(missing)))


def _write_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    # Fixed metadata makes repeated packaging byte-reproducible for unchanged
    # inputs with the same Python/zlib versions. No local absolute paths leak.
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, content, compresslevel=9)


def _verify_imports(extracted_root: Path, environment: dict[str, str]) -> str:
    command = (
        "from pathlib import Path; import csda_al, csda_al.cli; "
        "root=(Path.cwd()/'src').resolve(); "
        "assert Path(csda_al.__file__).resolve().is_relative_to(root); "
        "assert Path(csda_al.cli.__file__).resolve().is_relative_to(root); "
        "print('PASS: package and CLI imports resolve inside the extracted archive')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], cwd=extracted_root, env=environment,
        capture_output=True, text=True, timeout=120, check=False,
    )
    if result.returncode:
        raise RuntimeError("Extracted archive import check failed:\n" + result.stdout + result.stderr)
    return result.stdout.strip()


def verify_archive(
    archive_path: str | Path,
    *,
    run_tests: bool = False,
    required_files: Iterable[str] = REQUIRED_FILES,
) -> dict[str, Any]:
    """Verify CRC, every manifest hash, isolated extraction, and imports.

    The manifest describes every payload member except itself, explicitly
    avoiding a recursive self-hash. Extraction is temporary and always removed.
    """
    path = Path(archive_path).resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member names")
        for info in archive.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if (not parts or parts[0] != ARCHIVE_ROOT or len(parts) < 2 or "\\" in name
                    or any(part in {"..", "."} or ":" in part for part in parts)
                    or PurePosixPath(name).is_absolute() or info.is_dir()
                    or PurePosixPath(name).as_posix() != name
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe or unexpected archive member: {name}")
            if name != MANIFEST_NAME and not should_include("/".join(parts[1:])):
                raise ValueError(f"Archive contains excluded content: {name}")
        _check_required(names, required_files)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Archive CRC verification failed: {bad_member}")
        if MANIFEST_NAME not in names:
            raise ValueError("Archive manifest is missing")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        if (manifest.get("schema_version") != 1 or manifest.get("archive_root") != ARCHIVE_ROOT
                or manifest.get("self_hash_included") is not False):
            raise ValueError("Unsupported archive manifest schema")
        entries = manifest.get("files")
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            raise ValueError("Invalid archive manifest file entries")
        by_name = {item.get("member"): item for item in entries}
        if len(by_name) != len(entries) or set(by_name) != set(names) - {MANIFEST_NAME}:
            raise ValueError("Archive member set differs from its manifest")
        for name, entry in by_name.items():
            data = archive.read(name)
            if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                raise ValueError(f"Archive member size/SHA256 mismatch: {name}")
        with tempfile.TemporaryDirectory(prefix="csda_verify_", dir=path.parent) as directory:
            extraction = Path(directory).resolve()
            for name in names:
                destination = (extraction / name).resolve()
                if not destination.is_relative_to(extraction):
                    raise ValueError("Archive extraction would escape its destination")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
                if name != MANIFEST_NAME:
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != by_name[name]["sha256"]:
                        raise ValueError(f"Extracted file SHA256 mismatch: {name}")
            extracted_root = extraction / ARCHIVE_ROOT
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(extracted_root / "src")
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            imports = _verify_imports(extracted_root, environment)
            test_output = None
            if run_tests:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "--basetemp", str(extraction / "pytest_temp")],
                    cwd=extracted_root, env=environment, capture_output=True,
                    text=True, timeout=600, check=False,
                )
                test_output = result.stdout + result.stderr
                if result.returncode:
                    raise RuntimeError("Extracted archive tests failed:\n" + test_output)
    return {
        "archive": str(path), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload_files": len(entries), "archive_members": len(names),
        "crc": "PASS", "manifest_hashes": "PASS", "extracted_hashes": "PASS",
        "imports": imports, "extracted_tests": "PASS" if run_tests else "not_requested",
        "test_output": test_output,
    }


def build_archive(
    project_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    run_tests: bool = False,
    required_files: Iterable[str] = REQUIRED_FILES,
) -> dict[str, Any]:
    """Write only a complete, verified ZIP; retain an existing ZIP on failure."""
    root = Path(project_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Project directory not found: {root}")
    destination = Path(output_path).resolve() if output_path else root.parent / DEFAULT_ARCHIVE_NAME
    if destination.is_relative_to(root):
        raise ValueError("Archive destination must be outside the source project directory")
    required = tuple(required_files)
    payload = _payload(root)
    _check_required(payload, required)
    manifest = {
        "schema_version": 1, "archive_root": ARCHIVE_ROOT, "self_hash_included": False,
        "manifest_scope": "all_payload_members_except_ARCHIVE_MANIFEST.json",
        "files": [{"member": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in payload.items()],
        "policy": "original_sources_configs_tests_docs_only_no_data_weights_caches_or_reference_repositories",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in payload.items():
                _write_member(archive, name, data)
            _write_member(archive, MANIFEST_NAME, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        report = verify_archive(temporary, run_tests=run_tests, required_files=required)
        os.replace(temporary, destination)
        report["archive"] = str(destination)
        return report
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help=f"Default: ../{DEFAULT_ARCHIVE_NAME}")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest against the isolated extracted sources")
    parser.add_argument("--verify-only", type=Path, metavar="ZIP", help="Verify an existing ZIP without rebuilding it")
    args = parser.parse_args(argv)
    try:
        report = (verify_archive(args.verify_only, run_tests=args.run_tests) if args.verify_only
                  else build_archive(args.project_dir, args.output, run_tests=args.run_tests))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
