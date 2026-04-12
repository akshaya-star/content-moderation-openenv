"""Task 3: queue moderation — per-message, consistency, FP / missed penalties."""

from __future__ import annotations

from typing import Any, Dict, List

_MIN = 0.01
_MAX = 0.99


def _clamp(score: float) -> float:
    """Return score strictly in (0, 1) — never 0.0, never 1.0."""
    return max(_MIN, min(_MAX, float(score)))


def _decision_match(pred: str, exp: str) -> bool:
    return pred.strip().upper() == exp.strip().upper()


def grade(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Returns float strictly in (0.0, 1.0).
    prediction:
      decisions: list of {id, decision} in order
    ground_truth:
      messages: list with id, expected_decision, consistency_group, is_safe (optional)
    """
    try:
        messages: List[Dict[str, Any]] = ground_truth.get("messages") or []
        preds: List[Dict[str, Any]] = prediction.get("decisions") or []

        by_id_pr = {p.get("id"): p for p in preds if p.get("id")}

        n = len(messages)
        if n == 0:
            return _MAX

        per_msg_scores: List[float] = []
        fp_pen = 0.0
        miss_pen = 0.0

        for m in messages:
            mid = m["id"]
            exp = (m.get("expected_decision") or "").upper()
            is_safe = bool(m.get("is_safe", exp == "APPROVE"))
            pr = by_id_pr.get(mid, {})
            got = (pr.get("decision") or "").strip().upper()

            if not got:
                per_msg_scores.append(0.0)
                continue

            ok = _decision_match(got, exp)
            per_msg_scores.append(1.0 if ok else 0.0)

            if is_safe and got in ("FLAG", "BLOCK"):
                fp_pen += 0.12
            if not is_safe and exp in ("FLAG", "BLOCK") and got == "APPROVE":
                miss_pen += 0.18

        # Consistency: same consistency_group and same harm should have same non-APPROVE decision
        consistency_bonus = 0.0
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for m in messages:
            g = m.get("consistency_group")
            if g:
                groups.setdefault(str(g), []).append(m)

        for _gid, group_msgs in groups.items():
            harmful = [
                m
                for m in group_msgs
                if (m.get("expected_decision") or "").upper() in ("FLAG", "BLOCK")
            ]
            if len(harmful) < 2:
                continue
            decisions = []
            for m in harmful:
                pr = by_id_pr.get(m["id"], {})
                decisions.append((pr.get("decision") or "").strip().upper())
            if all(d == decisions[0] for d in decisions) and decisions[0] in (
                "FLAG",
                "BLOCK",
            ):
                consistency_bonus += 0.1

        base = sum(per_msg_scores) / n if n else 0.0
        raw = base + consistency_bonus - fp_pen - miss_pen
        return _clamp(raw)
    except Exception:
        return _MIN
