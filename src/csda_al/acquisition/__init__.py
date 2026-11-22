"""Original acquisition strategies with sealed-label-independent contracts."""
from .base import AcquisitionStrategy, PredictionBatch, QueryBatch, QueryContext
from .random import RandomInitialAcquisition, RandomAcquisition
from .entropy import EntropyAcquisition, entropy_scores
from .coreset import CoreSetAcquisition
from .supclust import SUPClustAcquisition, KMeansRepresentativeAcquisition
from .typiclust import TypiClustAcquisition
from .bald import BALDAcquisition, bald_scores, mc_dropout_probabilities
from .difficulty_aware import DifficultyAwareAcquisition, DifficultyOnlyAcquisition, EntropyDifficultyAcquisition
from .bengar import BengarClassBalancedAcquisition
from .flex_sampling import FlexibleSamplingAcquisition


def make_strategy(name: str, **config) -> AcquisitionStrategy:
    """Build a strategy; unknown names/configuration fail explicitly."""
    normalized = name.lower().replace("-", "_")
    classes = {"supclust": SUPClustAcquisition, "typiclust": TypiClustAcquisition,
               "kmeans": KMeansRepresentativeAcquisition,
               "kmeans_representative": KMeansRepresentativeAcquisition,
               "random": RandomInitialAcquisition, "entropy": EntropyAcquisition,
               "coreset": CoreSetAcquisition, "core_set": CoreSetAcquisition,
               "bald": BALDAcquisition, "difficulty_only": DifficultyOnlyAcquisition,
               "difficulty_aware": DifficultyAwareAcquisition,
               "entropy_difficulty": DifficultyAwareAcquisition,
               "bengar": BengarClassBalancedAcquisition,
               "bengar_class_balanced": BengarClassBalancedAcquisition,
               "class_balanced": BengarClassBalancedAcquisition,
               "flexible_sampling": FlexibleSamplingAcquisition,
               "flex_sampling": FlexibleSamplingAcquisition,
               "flexsampling": FlexibleSamplingAcquisition}
    if normalized in {"without_sup", "supclust_without_sup"}:
        return SUPClustAcquisition(variant="without_SUP", **config)
    if normalized in {"without_typicality", "supclust_without_typicality"}:
        return SUPClustAcquisition(variant="without_typicality", **config)
    if normalized not in classes:
        raise ValueError(f"unknown acquisition strategy: {name}")
    return classes[normalized](**config)


__all__ = ["AcquisitionStrategy", "PredictionBatch", "QueryBatch", "QueryContext",
           "RandomInitialAcquisition", "RandomAcquisition", "EntropyAcquisition",
           "CoreSetAcquisition", "SUPClustAcquisition", "KMeansRepresentativeAcquisition",
           "TypiClustAcquisition", "BALDAcquisition", "DifficultyAwareAcquisition",
           "DifficultyOnlyAcquisition", "EntropyDifficultyAcquisition",
           "BengarClassBalancedAcquisition", "FlexibleSamplingAcquisition",
           "make_strategy", "entropy_scores", "bald_scores", "mc_dropout_probabilities"]
