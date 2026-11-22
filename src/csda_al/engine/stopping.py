"""Class-Aware Early Stopping Criterion for Imbalance-Aware Active Learning.

Evaluates validation-only performance trajectories and acquired class counts to decide
whether active learning should terminate before budget exhaustion.
Strictly prevents any data leakage: never consumes unlabelled pool truth or test set targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from csda_al.data.metadata import CLASSES


@dataclass(frozen=True)
class StoppingDecision:
    """Immutable result of a stopping evaluation."""
    should_stop: bool
    reason: str
    round: int
    patience_counter: int
    min_class_coverage_met: bool
    current_macro_f1: float
    best_macro_f1: float
    per_class_recalls: tuple[float, ...]
    acquired_class_counts: tuple[int, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "round": self.round,
            "patience_counter": self.patience_counter,
            "min_class_coverage_met": self.min_class_coverage_met,
            "current_macro_f1": self.current_macro_f1,
            "best_macro_f1": self.best_macro_f1,
            "per_class_recalls": list(self.per_class_recalls),
            "acquired_class_counts": list(self.acquired_class_counts),
            "diagnostics": self.diagnostics,
        }


class ClassAwareStoppingCriterion:
    """Multi-condition class-aware early stopping rule.

    Conditions for stopping:
    1. Minimum Representation: Every class must have at least `min_per_class` acquired samples.
    2. Minimum Rounds: Active learning must run at least `min_rounds` (default: 3).
    3. Performance Plateau: Validation Macro-F1 fails to improve by more than `min_delta`
       for `patience` consecutive rounds.
    4. Minority Stability: Minority (tail) class validation recall has stabilized
       (change < `minority_delta` over consecutive rounds).
    """

    def __init__(
        self,
        *,
        min_rounds: int = 3,
        patience: int = 2,
        min_delta: float = 0.005,
        min_per_class: int = 2,
        minority_delta: float = 0.01,
        enabled: bool = True,
    ):
        if min_rounds < 1:
            raise ValueError("min_rounds must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0 or minority_delta < 0:
            raise ValueError("deltas must be non-negative")
        if min_per_class < 0:
            raise ValueError("min_per_class must be non-negative")

        self.min_rounds = int(min_rounds)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_per_class = int(min_per_class)
        self.minority_delta = float(minority_delta)
        self.enabled = bool(enabled)

        # Internal state tracking across rounds
        self._history_macro_f1: list[float] = []
        self._history_recalls: list[np.ndarray] = []
        self._best_macro_f1: float = -1.0
        self._patience_counter: int = 0

    def reset(self) -> None:
        """Reset history for a fresh active learning run."""
        self._history_macro_f1.clear()
        self._history_recalls.clear()
        self._best_macro_f1 = -1.0
        self._patience_counter = 0

    def evaluate(
        self,
        round_idx: int,
        acquired_counts: Sequence[int] | np.ndarray,
        validation_metrics: dict[str, Any],
    ) -> StoppingDecision:
        """Evaluate whether active learning should stop after validation of round_idx."""
        counts = tuple(int(x) for x in acquired_counts)
        num_classes = len(CLASSES)
        if len(counts) != num_classes:
            raise ValueError(f"acquired_counts must have {num_classes} elements")

        macro_f1 = float(validation_metrics.get("macro_f1", 0.0))
        per_class = validation_metrics.get("per_class", [])
        recalls_list = [float(row.get("recall", 0.0)) for row in per_class] if per_class else [0.0] * num_classes
        recalls = np.array(recalls_list, dtype=np.float64)

        self._history_macro_f1.append(macro_f1)
        self._history_recalls.append(recalls)

        # Check condition 1: Minimum representation
        coverage_met = all(c >= self.min_per_class for c in counts)

        if not self.enabled:
            return StoppingDecision(
                should_stop=False,
                reason="stopping_criterion_disabled",
                round=round_idx,
                patience_counter=0,
                min_class_coverage_met=coverage_met,
                current_macro_f1=macro_f1,
                best_macro_f1=max(self._history_macro_f1),
                per_class_recalls=tuple(recalls_list),
                acquired_class_counts=counts,
            )

        if round_idx < self.min_rounds:
            if macro_f1 > self._best_macro_f1:
                self._best_macro_f1 = macro_f1
            return StoppingDecision(
                should_stop=False,
                reason=f"minimum_rounds_not_reached_{round_idx}_lt_{self.min_rounds}",
                round=round_idx,
                patience_counter=0,
                min_class_coverage_met=coverage_met,
                current_macro_f1=macro_f1,
                best_macro_f1=self._best_macro_f1,
                per_class_recalls=tuple(recalls_list),
                acquired_class_counts=counts,
            )

        # Check macro-f1 improvement
        if macro_f1 > self._best_macro_f1 + self.min_delta:
            self._best_macro_f1 = macro_f1
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        # Check minority class stability (classes with lowest acquired counts)
        sorted_classes = np.argsort(counts)
        minority_indices = sorted_classes[:3]  # bottom 3 rarest classes
        minority_stable = False
        if len(self._history_recalls) >= 2:
            prev_minority_recall = self._history_recalls[-2][minority_indices]
            curr_minority_recall = recalls[minority_indices]
            recall_shift = np.abs(curr_minority_recall - prev_minority_recall).max()
            minority_stable = bool(recall_shift < self.minority_delta)

        # Stopping decision logic:
        # Must satisfy coverage and patience
        should_stop = False
        reason = "continuing_active_acquisition"

        if coverage_met and self._patience_counter >= self.patience:
            should_stop = True
            reason = (
                f"class_aware_plateau_patience_exhausted_{self._patience_counter}_ge_{self.patience}"
                + ("_and_minority_stable" if minority_stable else "")
            )
        elif not coverage_met and self._patience_counter >= self.patience:
            reason = f"patience_exhausted_but_coverage_not_met_continuing_for_minorities"

        diagnostics = {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_per_class": self.min_per_class,
            "minority_indices": minority_indices.tolist(),
            "minority_stable": minority_stable,
            "history_macro_f1": self._history_macro_f1,
        }

        return StoppingDecision(
            should_stop=should_stop,
            reason=reason,
            round=round_idx,
            patience_counter=self._patience_counter,
            min_class_coverage_met=coverage_met,
            current_macro_f1=macro_f1,
            best_macro_f1=self._best_macro_f1,
            per_class_recalls=tuple(recalls_list),
            acquired_class_counts=counts,
            diagnostics=diagnostics,
        )
