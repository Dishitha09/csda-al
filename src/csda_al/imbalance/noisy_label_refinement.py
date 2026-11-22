"""Active Label Refinement pipeline (Khanal et al., MICCAI 2024).

Implements Co-teaching dual-model small-loss selection, synthetic label corruption,
and active label refinement under label noise.
This pipeline addresses noisy-label learning, kept modularly separated from clean-pool active learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csda_al.data.metadata import CLASSES


def inject_symmetric_label_noise(
    labels: Sequence[int] | np.ndarray,
    noise_rate: float,
    num_classes: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject symmetric label noise into a label array.

    Returns:
        (corrupted_labels, is_corrupted_boolean_mask)
    """
    if not 0.0 <= noise_rate < 1.0:
        raise ValueError("noise_rate must be in [0, 1)")
    rng = np.random.RandomState(seed)
    y = np.asarray(labels, dtype=np.int64).copy()
    n = len(y)
    is_corrupted = np.zeros(n, dtype=bool)

    if noise_rate == 0.0:
        return y, is_corrupted

    num_corrupt = int(round(noise_rate * n))
    corrupt_indices = rng.choice(n, size=num_corrupt, replace=False)

    for idx in corrupt_indices:
        original = y[idx]
        other_classes = [c for c in range(num_classes) if c != original]
        y[idx] = rng.choice(other_classes)
        is_corrupted[idx] = True

    return y, is_corrupted


def co_teaching_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    targets: torch.Tensor,
    remember_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Co-teaching loss (Han et al. NeurIPS 2018 / Khanal et al. MICCAI 2024).

    Each model selects its small-loss samples to train the peer model.
    """
    loss1 = F.cross_entropy(logits1, targets, reduction="none")
    loss2 = F.cross_entropy(logits2, targets, reduction="none")

    n_samples = logits1.size(0)
    num_remember = max(1, int(remember_rate * n_samples))

    # Model 1 selects small loss for Model 2
    ind1_sorted = torch.argsort(loss1)
    ind1_update = ind1_sorted[:num_remember]

    # Model 2 selects small loss for Model 1
    ind2_sorted = torch.argsort(loss2)
    ind2_update = ind2_sorted[:num_remember]

    # Exchange loss
    peer_loss1 = loss1[ind2_update].mean()
    peer_loss2 = loss2[ind1_update].mean()

    return peer_loss1, peer_loss2, ind1_update, ind2_update


class ActiveLabelRefiner:
    """Khanal et al. (MICCAI 2024) Active Label Refinement module.

    Identifies suspected noisy samples via model disagreement and predictive entropy,
    and queries the oracle to refine (correct) their labels.
    """

    def __init__(self, num_classes: int = 8, uncertainty_type: str = "entropy"):
        self.num_classes = num_classes
        self.uncertainty_type = uncertainty_type

    def identify_suspects(
        self,
        probs1: np.ndarray,
        probs2: np.ndarray,
        noisy_labels: np.ndarray,
        refinement_budget: int,
    ) -> list[int]:
        """Rank and select samples suspected of label noise for oracle refinement."""
        pred1 = probs1.argmax(axis=1)
        pred2 = probs2.argmax(axis=1)

        # Disagreement between model predictions and noisy label
        disagree = (pred1 != noisy_labels) | (pred2 != noisy_labels)

        # Average predictive entropy
        p_avg = 0.5 * (probs1 + probs2)
        entropy = -np.sum(p_avg * np.log(np.clip(p_avg, 1e-12, 1.0)), axis=1)

        # Suspect score: disagreement weighted by uncertainty
        score = np.where(disagree, entropy + 1.0, entropy)

        order = np.argsort(-score)
        return order[:refinement_budget].tolist()
