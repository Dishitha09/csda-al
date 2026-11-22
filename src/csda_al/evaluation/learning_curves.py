"""Annotation-budget curves with no extrapolation or invented observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

Curve = tuple[Sequence[float], Sequence[float]]


def _curve(budgets: Sequence[float], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("a curve needs at least two aligned budget and score points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("curve values must be finite")
    if np.any(x < 0) or np.any(np.diff(x) <= 0):
        raise ValueError("budgets must be nonnegative and strictly increasing")
    if np.any(y < 0) or np.any(y > 1):
        raise ValueError("normalized metric scores must be within [0, 1]")
    return x, y


def normalized_aulc(
    budgets: Sequence[float],
    scores: Sequence[float],
    *,
    budget_interval: tuple[float, float] | None = None,
) -> float:
    """Trapezoidal area divided by interval width; interpolate only internally.

    Scores must already be normalized to [0, 1]. Curves being compared must use
    the same metric, label-cost definition, and interval; ``compare_aulc``
    enforces the shared interval across a mapping of curves.
    """
    x, y = _curve(budgets, scores)
    left, right = budget_interval if budget_interval is not None else (float(x[0]), float(x[-1]))
    if not np.isfinite([left, right]).all() or not left < right:
        raise ValueError("budget interval must have finite, increasing endpoints")
    if left < x[0] or right > x[-1]:
        raise ValueError("budget interval lies outside observed budgets; extrapolation is forbidden")
    inside = (x > left) & (x < right)
    aligned_x = np.concatenate(([left], x[inside], [right]))
    aligned_y = np.interp(aligned_x, x, y)
    # Explicit trapezoid expression supports NumPy 1.x and 2.x unchanged.
    area: float = float(np.sum(np.diff(aligned_x) * (aligned_y[:-1] + aligned_y[1:]) / 2))
    return float(area / (right - left))


def compare_aulc(
    curves: Mapping[str, Curve],
    *,
    budget_interval: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compare curves on one explicit or common-overlap budget interval."""
    if not curves:
        raise ValueError("at least one curve is required")
    checked = {name: _curve(*curve) for name, curve in curves.items()}
    interval = budget_interval or (
        max(float(x[0]) for x, _ in checked.values()),
        min(float(x[-1]) for x, _ in checked.values()),
    )
    values = {name: normalized_aulc(x.tolist(), y.tolist(), budget_interval=interval)
              for name, (x, y) in checked.items()}
    return {
        "budget_interval": [float(interval[0]), float(interval[1])],
        "interpolation": "piecewise_linear_no_extrapolation",
        "normalized_aulc": values,
    }


def _first_target_budget(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    hits = np.flatnonzero(y >= target)
    if not len(hits):
        return None
    index = int(hits[0])
    if index == 0:
        return float(x[0])
    fraction = (target - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def label_savings(
    baseline_budgets: Sequence[float], baseline_scores: Sequence[float],
    candidate_budgets: Sequence[float], candidate_scores: Sequence[float],
    target_score: float,
) -> dict[str, Any]:
    """Compare first interpolated attainment of a predeclared target score.

    Positive savings favor the candidate. An unattained target returns explicit
    status and ``None`` savings; it is never extrapolated. A crossing between
    measured points is an estimate, not an observed integer annotation count.
    """
    if not np.isfinite(target_score) or not 0 <= target_score <= 1:
        raise ValueError("target_score must be a finite value within [0, 1]")
    bx, by = _curve(baseline_budgets, baseline_scores)
    cx, cy = _curve(candidate_budgets, candidate_scores)
    baseline = _first_target_budget(bx, by, target_score)
    candidate = _first_target_budget(cx, cy, target_score)
    if baseline is None and candidate is None:
        status = "neither_reached"
    elif baseline is None:
        status = "baseline_not_reached"
    elif candidate is None:
        status = "candidate_not_reached"
    else:
        status = "reached"
    savings = baseline - candidate if baseline is not None and candidate is not None else None
    return {
        "target_score": float(target_score), "status": status,
        "baseline_budget": baseline, "candidate_budget": candidate,
        "labels_saved": savings,
        "fraction_saved": savings / baseline if savings is not None and baseline else None,
        "interpolation": "first_piecewise_linear_crossing_no_extrapolation",
    }


def plot_learning_curves(
    curves: Mapping[str, Curve],
    output_path: str | Path,
    *,
    metric_name: str = "Balanced accuracy",
    label_cost_name: str = "Acquired training annotations",
    synthetic: bool = False,
) -> Path:
    """Save a standalone PNG/PDF plot of supplied observations.

    Callers must mark synthetic inputs. The function creates no measurements.
    """
    if not curves:
        raise ValueError("at least one curve is required")
    checked = {name: _curve(*curve) for name, curve in curves.items()}
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output_path)
    if destination.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("plot output must be .png, .pdf or .svg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(7.5, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for name, (x, y) in checked.items():
        axes.plot(x, y, marker="o", label=name)
    axes.set(xlabel=label_cost_name, ylabel=metric_name, ylim=(0, 1))
    axes.set_title("SYNTHETIC DEMONSTRATION - NO REAL TRAINING" if synthetic else "Observed evaluation learning curves")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(destination, dpi=160)
    return destination
