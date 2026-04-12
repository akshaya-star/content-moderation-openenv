"""Task 1: single-message moderation — exact decision match."""

from __future__ import annotations

from typing import Any, Dict

_MIN = 0.01
_MAX = 0.99


def _clamp(score: float) -> float:
    """Return score strictly in (0, 1) — never 0.0, never 1.0."""
    return max(_MIN, min(_MAX, float(score)))


def grade(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Returns float strictly in (0.0, 1.0). Exact match on expected_decision.
    prediction keys: decision
    ground_truth keys: expected_decision
    """
    try:
        exp = (ground_truth.get("expected_decision") or "").strip().upper()
        got = (prediction.get("decision") or "").strip().upper()
        if not exp or not got:
            score = _MIN
        else:
            score = _MAX if got == exp else _MIN
        return _clamp(score)
    except Exception:
        return _MIN
