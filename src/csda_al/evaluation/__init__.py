"""Split-aware evaluation and annotation-budget comparisons."""

from .learning_curves import (
    compare_aulc,
    label_savings,
    normalized_aulc,
    plot_learning_curves,
)
from .metrics import classification_metrics, frequency_groups, head_middle_tail_metrics
from .report import EvaluationLock, EvaluationResult, evaluate_predictions, export_result

__all__ = [
    "EvaluationLock", "EvaluationResult", "classification_metrics", "compare_aulc",
    "evaluate_predictions", "export_result", "frequency_groups",
    "head_middle_tail_metrics", "label_savings", "normalized_aulc",
    "plot_learning_curves",
]
