"""Prediction-based expected difficulty; no hidden known-class buckets."""
import numpy as np


def expected_difficulty(probabilities, difficulty) -> np.ndarray:
    from csda_al.acquisition.base import probability_array
    p = probability_array(probabilities)
    d = np.asarray(difficulty.values if hasattr(difficulty, "values") else difficulty, dtype=float)
    if d.shape != (8,) or not np.isfinite(d).all() or (d < 0).any() or (d > 1).any():
        raise ValueError("difficulty must contain eight finite values in [0,1]")
    return p @ d


def normalize_difficulty(values, *, method: str = "minmax") -> tuple[np.ndarray, dict]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any() or (x > 1+1e-12).any():
        raise ValueError("expected difficulty must be a finite [0,1] vector")
    if method not in {"minmax", "identity"}:
        raise ValueError("unknown difficulty normalization")
    low, high = (float(x.min()), float(x.max())) if len(x) else (0.0, 0.0)
    constant = high-low <= 1e-12
    scaled = x.copy() if method == "identity" else (np.zeros_like(x) if constant else (x-low)/(high-low))
    return scaled, {"normalization": method, "minimum": low, "maximum": high,
                    "constant_difficulty_signal": constant}
