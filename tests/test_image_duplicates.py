"""Image identity and near-duplicate screening without diagnostic labels."""
from collections.abc import Mapping
import hashlib
import io
import itertools

import numpy as np
from PIL import Image
import pytest

from csda_al.data.duplicates import detect_duplicate_groups, image_fingerprints


def row(image_id, *, phash=0, dhash=0, width=32, height=32):
    return {"image_id": image_id, "byte_sha256": hashlib.sha256((image_id + "bytes").encode()).hexdigest(),
            "pixel_sha256": hashlib.sha256((image_id + "pixels").encode()).hexdigest(),
            "phash": f"{phash:016x}", "dhash": f"{dhash:016x}", "width": width, "height": height}


def encoded_row(image_id, image, compression):
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=compression)
    payload = stream.getvalue()
    with Image.open(io.BytesIO(payload)) as source:
        data = image_fingerprints(source)
    return {"image_id": image_id, "byte_sha256": hashlib.sha256(payload).hexdigest(), **data}


def test_exact_pixels_across_different_encodings_and_original_ids_retained():
    pixels = np.random.default_rng(8).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    a, b = encoded_row("a", image, 0), encoded_row("b", image, 9)
    assert a["byte_sha256"] != b["byte_sha256"] and a["pixel_sha256"] == b["pixel_sha256"]
    result = detect_duplicate_groups([b, a])
    assert result["image_count"] == 2
    assert result["exact_byte_groups"] == [] and result["exact_pixel_groups"] == [["a", "b"]]
    assert result["near_candidates"] == []
    assert result["policy"]["retention"] == "all_original_image_ids_retained"


def test_exact_byte_identity_and_deterministic_output():
    a, b, c = row("a"), row("b"), row("c", phash=2, dhash=2)
    b["byte_sha256"], b["pixel_sha256"] = a["byte_sha256"], a["pixel_sha256"]
    expected = detect_duplicate_groups([a, b, c])
    assert expected["exact_byte_groups"] == expected["exact_pixel_groups"] == [["a", "b"]]
    for ordering in itertools.permutations([a, b, c]):
        assert detect_duplicate_groups(ordering) == expected


def test_grayscale_hash_false_friends_are_unconfirmed_and_never_dropped():
    # Different RGB images with the same luminance: perceptual hashes alone do
    # not prove that these are duplicate images or the same clinical lesion.
    red = Image.new("RGB", (32, 32), (255, 0, 0))
    green = Image.new("RGB", (32, 32), (0, 130, 0))
    a, b = encoded_row("red", red, 1), encoded_row("green", green, 1)
    assert a["pixel_sha256"] != b["pixel_sha256"]
    assert a["phash"] == b["phash"] and a["dhash"] == b["dhash"]
    result = detect_duplicate_groups([a, b])
    assert result["image_count"] == 2 and result["exact_pixel_groups"] == []
    assert len(result["near_candidates"]) == 1
    assert result["near_candidates"][0]["confirmed_duplicate"] is False
    assert result["policy"]["near_candidates_are_confirmed"] is False


def test_dual_hash_and_aspect_ratio_filters_avoid_single_hash_matches():
    a = row("a")
    far_dhash = row("b", phash=1, dhash=(1 << 64) - 1)
    far_phash = row("c", phash=31, dhash=0)
    different_aspect = row("d", width=64)
    assert detect_duplicate_groups([a, far_dhash, far_phash, different_aspect])["near_candidates"] == []
    near = row("z", phash=15, dhash=15)
    candidate = detect_duplicate_groups([a, near])["near_candidates"]
    assert len(candidate) == 1 and candidate[0]["phash_hamming"] == candidate[0]["dhash_hamming"] == 4


def test_bk_tree_candidates_match_small_bruteforce_reference():
    rng = np.random.default_rng(9)
    phashes = [int(x) for x in rng.integers(0, 2**64, size=80, dtype=np.uint64)]
    phashes += [1 << i for i in range(24)]
    dhashes = phashes.copy()
    rows = [row(f"im-{i:03}", phash=p, dhash=d) for i, (p, d) in enumerate(zip(phashes, dhashes))]
    expected = {(a["image_id"], b["image_id"])
                for a, b in itertools.combinations(rows, 2)
                if (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() <= 2}
    result = detect_duplicate_groups(rows, near_threshold=2)
    actual = {(pair["image_id1"], pair["image_id2"]) for pair in result["near_candidates"]}
    assert actual == expected and len(actual) >= 276
    assert result["diagnostics"]["pairwise_matrix_allocated"] is False


class LabelTrap(Mapping):
    def __init__(self, value):
        self.value = {**value, "hidden_label": 7}

    def __getitem__(self, key):
        if key == "hidden_label":
            raise AssertionError("Duplicate screening must not read diagnostic labels")
        return self.value[key]

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)


def test_diagnostic_labels_are_not_even_read():
    assert detect_duplicate_groups([LabelTrap(row("a")), LabelTrap(row("b"))])["image_count"] == 2


def test_fingerprints_do_not_mutate_source_and_dimensions_are_part_of_identity():
    image = Image.new("RGB", (8, 16), (10, 20, 30))
    before = image.tobytes()
    fingerprints = image_fingerprints(image)
    other = image_fingerprints(Image.new("RGB", (16, 8), (10, 20, 30)))
    assert image.tobytes() == before and image.size == (8, 16)
    assert fingerprints["pixel_sha256"] != other["pixel_sha256"]
    assert fingerprints["width"] == 8 and fingerprints["height"] == 16 and fingerprints["mode"] == "RGB"


@pytest.mark.parametrize("threshold", [-1, 65, True, 1.5])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        detect_duplicate_groups([], near_threshold=threshold)


@pytest.mark.parametrize("mutation", [
    {"image_id": ""}, {"width": 0}, {"height": True}, {"phash": "bad"},
    {"dhash": -1}, {"byte_sha256": "not-a-digest"}, {"pixel_sha256": ""},
])
def test_invalid_inventory_rows_fail_explicitly(mutation):
    with pytest.raises(ValueError):
        detect_duplicate_groups([{**row("a"), **mutation}])


def test_duplicate_ids_rejected_and_empty_inventory_is_explicit():
    with pytest.raises(ValueError):
        detect_duplicate_groups([row("a"), row("a")])
    result = detect_duplicate_groups([])
    assert result["image_count"] == 0 and result["near_candidates"] == []
