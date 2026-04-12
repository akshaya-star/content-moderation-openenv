"""Task 1: single-message moderation — exact decision match."""

from __future__ import annotations

from typing import Any, Dict


def grade(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Returns float in [0.0, 1.0]. Exact match on expected_decision.
    prediction keys: decision
    ground_truth keys: expected_decision
    """
    try:
        exp = (ground_truth.get("expected_decision") or "").strip().upper()
        got = (prediction.get("decision") or "").strip().upper()
        if not exp or not got:
            score = 0.01
        else:
            score = 0.99 if got == exp else 0.01
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.0
