"""Optional FAISS neighbors with a scikit-learn CPU fallback."""
import importlib
from typing import Any
import numpy as np
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(features: np.ndarray, k: int, query: np.ndarray | None = None,
                      exclude_self: bool = True, use_faiss: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features)
    q = x if query is None else np.asarray(query)
    if x.ndim != 2 or len(x) == 0 or q.ndim != 2 or q.shape[1] != x.shape[1]:
        raise ValueError("Expected nonempty aligned feature matrices")
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError("Neighbor features must be finite")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    self_query = query is None and exclude_self
    effective = min(k, len(x) - int(self_query))
    if effective == 0:
        return np.empty((len(q), 0)), np.empty((len(q), 0), dtype=np.int64)
    count = effective + int(self_query)
    faiss_backend: Any = None
    if use_faiss:
        try:
            faiss_backend = importlib.import_module("faiss")
        except ImportError:
            pass
    if faiss_backend is not None:
        index = faiss_backend.IndexFlatL2(x.shape[1])
        index.add(np.ascontiguousarray(x, dtype=np.float32))
        squared, indices = index.search(np.ascontiguousarray(q, dtype=np.float32), count)
        distances = np.sqrt(np.maximum(squared, 0))
    else:
        distances, indices = NearestNeighbors(n_neighbors=count, metric="euclidean").fit(x).kneighbors(q)
    out_d, out_i = [], []
    for row, (dist, ind) in enumerate(zip(distances, indices)):
        pairs = sorted((float(d), int(i)) for d, i in zip(dist, ind) if not self_query or i != row)
        pairs = pairs[:effective]
        out_d.append([p[0] for p in pairs])
        out_i.append([p[1] for p in pairs])
    return np.asarray(out_d), np.asarray(out_i, dtype=np.int64)
