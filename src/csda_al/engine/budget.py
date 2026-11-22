"""Separate TRAIN acquisition and paid validation budgets."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AnnotationBudget:
    max_training_annotations: int
    validation_annotation_cost: int = 0
    total_label_cap: int | None = None

    def __post_init__(self) -> None:
        for value in (self.max_training_annotations, self.validation_annotation_cost, self.total_label_cap):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError("Annotation costs and caps must be nonnegative integers")
        if self.total_label_cap is not None and self.total_label_cap < self.validation_annotation_cost:
            raise ValueError("Total label cap cannot cover the purchased validation labels")

    def remaining(self, acquired: int) -> int:
        if isinstance(acquired, bool) or not isinstance(acquired, int) or acquired < 0:
            raise ValueError("Acquired annotation count must be nonnegative integer")
        remaining = self.max_training_annotations - acquired
        if self.total_label_cap is not None:
            remaining = min(remaining, self.total_label_cap - self.validation_annotation_cost - acquired)
        if remaining < 0:
            raise ValueError("Committed annotations exceed budget")
        return remaining
