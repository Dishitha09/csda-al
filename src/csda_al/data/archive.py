"""Verified neutral extraction of a class-directory ISIC archive.

This is privileged dataset preparation. Source member names stay in a private
provenance file; runtime manifests and image paths do not encode diagnoses.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Callable
import zipfile

from PIL import Image

from .duplicates import image_fingerprints
from .metadata import CLASSES
from ..utils.io import atomic_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_and_scan(archive_path: str | Path, output_root: str | Path, *, workers: int = 4,
                     progress: Callable[[str], None] | None = None) -> tuple[list[dict], dict]:
    """CRC-read every member; fully decode RGB images and compute fingerprints.

    Preserves source bytes and never overwrites a different existing image.
    A completion record is written only after all images validate successfully.
    """
    archive_path = Path(archive_path).resolve()
    output_root = Path(output_root).resolve()
    if not archive_path.is_file():
        raise ValueError(f"Archive not found: {archive_path}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    images = output_root / "images"
    private = output_root / "private_source"
    audit = output_root / "analysis"
    for folder in (images, private, audit):
        folder.mkdir(parents=True, exist_ok=True)
    csv_names = {"ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Metadata.csv"}
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len({i.filename for i in members}) != len(members):
            raise ValueError("Duplicate archive member names")
        jobs = []
        ids: set[str] = set()
        provenance = []
        crc_csv = 0
        for info in members:
            member = PurePosixPath(info.filename)
            if (member.is_absolute() or ".." in member.parts or "\\" in info.filename
                    or any(":" in part for part in member.parts)):
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if info.filename in csv_names:
                content = archive.read(info)  # ZipFile checks the CRC after the complete read.
                path = private / info.filename
                if path.exists() and path.read_bytes() != content:
                    raise ValueError(f"Refusing to overwrite changed source CSV: {path}")
                path.write_bytes(content)
                crc_csv += 1
                continue
            if len(member.parts) != 2 or member.parts[0] not in CLASSES or member.suffix.lower() != ".jpg":
                raise ValueError(f"Unexpected archive member: {info.filename}")
            image_id = member.stem
            if not image_id.startswith("ISIC_") or image_id.casefold() in ids:
                raise ValueError(f"Invalid or duplicate image ID: {image_id}")
            ids.add(image_id.casefold())
            jobs.append((info, image_id))
            provenance.append({"image_id": image_id, "source_member": info.filename,
                               "crc32": f"{info.CRC:08x}", "uncompressed_bytes": info.file_size})
        if crc_csv != 2 or not jobs:
            raise ValueError("Archive must include images and both official training CSV files")

        def process(job: tuple[zipfile.ZipInfo, str]) -> dict:
            info, image_id = job
            target = images / f"{image_id}.jpg"
            descriptor, temporary_name = tempfile.mkstemp(prefix=image_id + ".", suffix=".tmp", dir=images)
            temporary = Path(temporary_name)
            try:
                digest = hashlib.sha256()
                with os.fdopen(descriptor, "wb") as destination, archive.open(info) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                byte_hash = digest.hexdigest()
                if temporary.stat().st_size != info.file_size:
                    raise ValueError(f"Archive size mismatch for {image_id}")
                with Image.open(temporary) as image:
                    original_mode = image.mode
                    orientation = image.getexif().get(274, 1)
                    image.load()  # Full decoder validation; truncated images are not allowed.
                    fingerprints = image_fingerprints(image)
                if target.exists():
                    if file_sha256(target) != byte_hash:
                        raise ValueError(f"Existing neutral image differs from archive: {image_id}")
                else:
                    os.replace(temporary, target)
                return {"image_id": image_id, "relative_path": f"images/{image_id}.jpg",
                        "size_bytes": info.file_size, "byte_sha256": byte_hash,
                        "original_mode": original_mode, "exif_orientation": int(orientation),
                        "valid": True, **fingerprints}
            finally:
                temporary.unlink(missing_ok=True)

        with (audit / "image_scan_progress.jsonl").open("w", encoding="utf-8") as journal:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process, job): job[1] for job in jobs}
                errors = []
                for completed, future in enumerate(as_completed(futures), 1):
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {"image_id": futures[future], "valid": False,
                               "error": f"{type(exc).__name__}: {exc}"}
                        errors.append(row)
                    rows.append(row)
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    if completed % 500 == 0 or completed == len(jobs):
                        journal.flush()
                        if progress:
                            progress(f"Decoded and CRC-checked {completed}/{len(jobs)} images; errors={len(errors)}")
        provenance.sort(key=lambda row: row["image_id"])
        with (private / "archive_members.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(provenance[0]))
            writer.writeheader()
            writer.writerows(provenance)
        if errors:
            atomic_json(audit / "invalid_images.json", errors)
            raise ValueError(f"Image scan failed: {len(errors)} images invalid; see analysis/invalid_images.json")
    rows.sort(key=lambda row: row["image_id"])
    with (audit / "image_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if progress:
        progress("Computing complete source archive SHA-256")
    summary = {"source_archive": str(archive_path), "archive_bytes": archive_path.stat().st_size,
               "archive_sha256": file_sha256(archive_path), "image_count": len(rows),
               "csv_count": crc_csv, "crc_checked_members": len(rows) + crc_csv,
               "full_image_decode_count": len(rows), "invalid_images": 0,
               "neutral_image_root": str(images), "source_bytes_preserved": True,
               "fingerprint_recipe": rows[0]["fingerprint_recipe"], "workers": workers}
    atomic_json(audit / "archive_scan_summary.json", summary)
    return rows, summary
