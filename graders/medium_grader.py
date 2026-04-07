"""Task 2: multi-label moderation — weighted decision, category, severity."""

from __future__ import annotations

from typing import Any, Dict


def _norm_cat(s: str) -> str:
    return (s or "").strip().lower()


def _norm_sev(s: str) -> str:
    return (s or "").strip().lower()


def grade(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    0.5 decision + 0.3 category + 0.2 severity (normalized categories/labels).
    prediction: decision, category (optional), severity (optional)
    ground_truth: expected_decision, labels (list), severity
    """
    w_dec, w_cat, w_sev = 0.5, 0.3, 0.2
    exp_dec = (ground_truth.get("expected_decision") or "").strip().upper()
    got_dec = (prediction.get("decision") or "").strip().upper()
    dec_score = 1.0 if exp_dec and got_dec == exp_dec else 0.0

    gt_labels = {_norm_cat(x) for x in (ground_truth.get("labels") or [])}
    pred_cat = _norm_cat(prediction.get("category") or "")
    if not gt_labels or gt_labels == {"none"}:
        cat_score = 1.0 if pred_cat == "none" else 0.0
    else:
        cat_score = 1.0 if pred_cat in gt_labels else 0.0

    exp_sev = _norm_sev(ground_truth.get("severity") or "")
    got_sev = _norm_sev(prediction.get("severity") or "")
    sev_score = 1.0 if exp_sev and got_sev == exp_sev else 0.0
    if not exp_sev:
        sev_score = 1.0

    raw = w_dec * dec_score + w_cat * cat_score + w_sev * sev_score
    return max(0.0, min(1.0, float(raw)))
