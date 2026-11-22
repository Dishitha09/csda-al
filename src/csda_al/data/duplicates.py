"""Label-independent exact image identity and conservative near-duplicate candidates.

Perceptual similarity is a leakage-screening signal, never proof that two lesions
are the same. All original IDs are retained; this module never deletes images.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from numbers import Integral
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

FINGERPRINT_RECIPE = "exif_transpose_rgb_sha256_phash32_dct8_no_dc_dhash9x8_v1"


def _hex_bits(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def image_fingerprints(image: Image.Image) -> dict[str, Any]:
    """Hash a fully decoded image without mutating it or consulting labels.

    Exact-pixel identity includes RGB dimensions and EXIF-normalized pixels.
    Perceptual hashes use luminance and can collide for different-color lesions.
    pHash uses a 32x32 DCT, top 8x8 block, median excluding DC, DC bit zero.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb.load()
    width, height = rgb.size
    if width < 1 or height < 1:
        raise ValueError("Image must have positive dimensions")
    pixel_hash = hashlib.sha256(f"RGB:{width}x{height}\0".encode("ascii") + rgb.tobytes()).hexdigest()
    gray = rgb.convert("L")
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    spectrum = dctn(small, type=2, norm="ortho")[:8, :8]
    pbits = spectrum > np.median(spectrum.reshape(-1)[1:])
    pbits[0, 0] = False
    dsmall = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return {"pixel_sha256": pixel_hash, "phash": _hex_bits(pbits),
            "dhash": _hex_bits(dsmall[:, 1:] > dsmall[:, :-1]),
            "width": width, "height": height, "mode": "RGB",
            "fingerprint_recipe": FINGERPRINT_RECIPE}


def _hash64(value: Any, name: str) -> int:
    if isinstance(value, str):
        if len(value) != 16 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError(f"{name} must be a 16-character hexadecimal hash")
        return int(value, 16)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an unsigned 64-bit integer or 16 hex characters")
    number = int(value)
    if not 0 <= number < 2**64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer or 16 hex characters")
    return number


class _BKNode:
    def __init__(self, value: int, index: int):
        self.value = value
        self.indices = [index]
        self.children: dict[int, _BKNode] = {}


class _HammingTree:
    """Exact metric-radius search without an N-by-N distance allocation."""
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def insert(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value, index)
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                node.indices.append(index)
                return
            if distance not in node.children:
                node.children[distance] = _BKNode(value, index)
                return
            node = node.children[distance]

    def search(self, value: int, radius: int) -> Iterable[tuple[int, int]]:
        pending = [self.root] if self.root is not None else []
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                yield from ((index, distance) for index in node.indices)
            lower, upper = distance - radius, distance + radius
            pending.extend(child for edge, child in node.children.items() if lower <= edge <= upper)


def detect_duplicate_groups(inventory_rows: Iterable[Mapping[str, Any]], *, near_threshold: int = 4) -> dict:
    """Return confirmed hash groups and unconfirmed perceptual candidates.

    Required row fields: image_id, byte_sha256, pixel_sha256, phash, dhash,
    width, height. Hashes use helper output; byte SHA is computed by the scanner.
    Candidates require BOTH Hamming distances <= threshold and relative aspect
    ratio difference <=2%. Stable lexicographic image-ID order resolves output.
    Exact byte/pixel pairs are not duplicated in the near-candidate list.
    """
    if isinstance(near_threshold, (bool, np.bool_)) or not isinstance(near_threshold, Integral):
        raise ValueError("near_threshold must be an integer")
    if not 0 <= near_threshold <= 64:
        raise ValueError("near_threshold must be in [0,64]")
    keys = ("image_id", "byte_sha256", "pixel_sha256", "phash", "dhash", "width", "height")
    # Deliberately read only image-inventory fields. Do not even copy a caller's
    # unrelated labels/predictions/clinical metadata into duplicate processing.
    rows = sorted(({key: row.get(key) for key in keys} for row in inventory_rows),
                  key=lambda r: str(r.get("image_id", "")))
    ids = [row.get("image_id") for row in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Inventory image IDs must be unique nonempty strings")
    byte_groups: dict[str, list[str]] = defaultdict(list)
    pixel_groups: dict[str, list[str]] = defaultdict(list)
    phashes, dhashes, aspects = [], [], []
    for row in rows:
        for key in ("byte_sha256", "pixel_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
                raise ValueError(f"{key} must contain a SHA-256 hex digest")
            row[key] = value.lower()
        for key in ("width", "height"):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), Integral) or row[key] < 1:
                raise ValueError("Inventory dimensions must be positive integers")
        phashes.append(_hash64(row.get("phash"), "phash"))
        dhashes.append(_hash64(row.get("dhash"), "dhash"))
        aspects.append(row["width"] / row["height"])
        byte_groups[row["byte_sha256"]].append(row["image_id"])
        pixel_groups[row["pixel_sha256"]].append(row["image_id"])
    tree = _HammingTree()
    candidates = []
    search_matches = 0
    for right, row in enumerate(rows):
        for left, pdistance in tree.search(phashes[right], int(near_threshold)):
            search_matches += 1
            other = rows[left]
            if other["byte_sha256"] == row["byte_sha256"] or other["pixel_sha256"] == row["pixel_sha256"]:
                continue
            ddistance = (dhashes[left] ^ dhashes[right]).bit_count()
            relative_aspect_difference = abs(aspects[left] - aspects[right]) / max(aspects[left], aspects[right])
            if ddistance <= near_threshold and relative_aspect_difference <= 0.02:
                candidates.append({"image_id1": other["image_id"], "image_id2": row["image_id"],
                                   "phash_hamming": pdistance, "dhash_hamming": ddistance,
                                   "relative_aspect_difference": float(relative_aspect_difference),
                                   "confirmed_duplicate": False})
        tree.insert(phashes[right], right)
    candidates.sort(key=lambda item: (item["image_id1"], item["image_id2"]))
    return {"image_count": len(rows),
            "exact_byte_groups": sorted((group for group in byte_groups.values() if len(group) > 1), key=lambda g: g[0]),
            "exact_pixel_groups": sorted((group for group in pixel_groups.values() if len(group) > 1), key=lambda g: g[0]),
            "near_candidates": candidates,
            "policy": {"near_threshold": int(near_threshold), "require_both_hashes": True,
                       "max_relative_aspect_difference": 0.02, "near_candidates_are_confirmed": False,
                       "retention": "all_original_image_ids_retained", "search": "exact_hamming_bk_tree",
                       "fingerprint_recipe": FINGERPRINT_RECIPE},
            "diagnostics": {"phash_radius_matches": search_matches,
                            "near_candidate_count": len(candidates), "pairwise_matrix_allocated": False,
                            "worst_case": "quadratic candidate output possible for colliding hashes"}}
