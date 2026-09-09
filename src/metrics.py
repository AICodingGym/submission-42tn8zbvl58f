"""Evaluation and score-discretization helpers for essay scoring."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

try:
    from src.config import SCORE_MAX, SCORE_MIN
except ModuleNotFoundError:  # Support `python src/metrics.py` imports.
    from config import SCORE_MAX, SCORE_MIN  # type: ignore[no-redef]


SCORE_LABELS = tuple(range(SCORE_MIN, SCORE_MAX + 1))
DEFAULT_THRESHOLDS = tuple(
    score + 0.5 for score in range(SCORE_MIN, SCORE_MAX)
)


def _as_integer_scores(values: Iterable[int | float], name: str) -> np.ndarray:
    """Validate and return a one-dimensional integer score array."""

    array = np.asarray(list(values))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    numeric = array.astype(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains non-finite values")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} must contain integer scores")
    if ((rounded < SCORE_MIN) | (rounded > SCORE_MAX)).any():
        raise ValueError(
            f"{name} contains scores outside {SCORE_MIN}..{SCORE_MAX}"
        )
    return rounded.astype(np.int8)


def quadratic_weighted_kappa(
    y_true: Iterable[int | float],
    y_pred: Iterable[int | float],
) -> float:
    """Compute the competition metric with the full ordered label range."""

    true_scores = _as_integer_scores(y_true, "y_true")
    predicted_scores = _as_integer_scores(y_pred, "y_pred")
    if true_scores.shape != predicted_scores.shape:
        raise ValueError("y_true and y_pred must have the same length")
    score = float(
        cohen_kappa_score(
            true_scores,
            predicted_scores,
            labels=list(SCORE_LABELS),
            weights="quadratic",
        )
    )
    if not np.isfinite(score):
        raise ValueError("QWK is undefined for the supplied score arrays")
    return score


def apply_ordered_thresholds(
    predictions: Iterable[int | float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    """Map continuous predictions to integer scores using strict thresholds."""

    values = np.asarray(list(predictions), dtype=float)
    if values.ndim != 1:
        raise ValueError("predictions must be one-dimensional")
    if values.size == 0:
        raise ValueError("predictions must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    threshold_array = np.asarray(thresholds, dtype=float)
    expected_count = SCORE_MAX - SCORE_MIN
    if threshold_array.shape != (expected_count,):
        raise ValueError(f"exactly {expected_count} thresholds are required")
    if not np.isfinite(threshold_array).all():
        raise ValueError("thresholds contain non-finite values")
    if not np.all(np.diff(threshold_array) > 0):
        raise ValueError("thresholds must be strictly increasing")
    return (
        SCORE_MIN
        + (values[:, None] > threshold_array[None, :]).sum(axis=1)
    ).astype(np.int8)
