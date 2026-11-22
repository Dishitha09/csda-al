"""Maximum Euclidean within-cluster typicality baseline.

Uses the shared SUPClust cluster eligibility contract; it does not silently copy
the reference's 500-cluster cap, squared FAISS distances or tiny-cluster removal.
"""
from .supclust import SUPClustAcquisition, typicality_scores


class TypiClustAcquisition(SUPClustAcquisition):
    def __init__(self, **kwargs):
        super().__init__(variant="typiclust", **kwargs)


__all__ = ["TypiClustAcquisition", "typicality_scores"]
