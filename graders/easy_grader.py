"""Task 1: single-message moderation — exact decision match."""

from __future__ import annotations

from typing import Any, Dict


def grade(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Returns float in [0.0, 1.0]. Exact match on expected_decision.
    prediction keys: decision
    ground_truth keys: expected_decision
    """
    exp = (ground_truth.get("expected_decision") or "").strip().upper()
    got = (prediction.get("decision") or "").strip().upper()
    if not exp or not got:
        return 0.0
    return 1.0 if got == exp else 0.0
