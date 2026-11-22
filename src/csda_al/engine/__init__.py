"""Fixed-budget pool management, stopping criteria, and active-learning orchestration."""
from .budget import AnnotationBudget
from .checkpointing import load_checkpoint, save_checkpoint
from .pool_state import PoolState
from .stopping import ClassAwareStoppingCriterion, StoppingDecision

__all__ = [
    "AnnotationBudget",
    "load_checkpoint",
    "save_checkpoint",
    "PoolState",
    "ClassAwareStoppingCriterion",
    "StoppingDecision",
]
