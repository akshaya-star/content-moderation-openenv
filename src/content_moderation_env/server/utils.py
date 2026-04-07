"""Load tasks, normalize labels, rewards, and consistency helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root: .../content-moderation-openenv
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


def load_json_samples(task: str) -> List[Dict[str, Any]]:
    """Load benchmark samples for easy | medium | hard."""
    name = {"easy": "easy", "medium": "medium", "hard": "hard"}[task]
    path = DATA_DIR / f"{name}_samples.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def pick_episode(
    samples: List[Dict[str, Any]], seed: Optional[int], episode_id: Optional[str]
) -> Dict[str, Any]:
    """Deterministically pick one episode/sample."""
    if episode_id:
        for s in samples:
            if s.get("id") == episode_id or s.get("episode_id") == episode_id:
                return s
    rng = random.Random(seed if seed is not None else 42)
    return rng.choice(samples)


def normalize_decision(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    u = raw.strip().upper()
    if u in ("APPROVE", "FLAG", "BLOCK"):
        return u
    return None


def normalize_category(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    c = raw.strip().lower()
    allowed = {"spam", "harassment", "hate", "threat", "none"}
    if c in allowed:
        return c
    return None


def normalize_severity(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("low", "medium", "high"):
        return s
    return None


def step_reward_shaping(
    *,
    task: str,
    pred_decision: Optional[str],
    gt_decision: str,
    is_safe_content: bool,
    pred_category: Optional[str] = None,
    gt_category: Optional[str] = None,
    pred_severity: Optional[str] = None,
    gt_severity: Optional[str] = None,
) -> float:
    """Dense per-step reward contribution (can be negative)."""
    r = 0.0
    if pred_decision is None:
        return -0.05
    if pred_decision == gt_decision:
        r += 0.2
    else:
        if is_safe_content and pred_decision in ("FLAG", "BLOCK"):
            r -= 0.15
        if not is_safe_content and pred_decision == "APPROVE":
            r -= 0.25
    if task == "medium" and gt_category and gt_severity:
        if pred_category == gt_category:
            r += 0.2
        if pred_severity == gt_severity:
            r += 0.2
    return max(-1.0, min(1.0, r))


def messages_public_view(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Strip labels from message for observation."""
    return {"id": msg.get("id"), "text": msg.get("text")}


def hard_queue_context(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Short queue listing for observation (no labels)."""
    return [
        {"id": m.get("id"), "preview": (m.get("text") or "")[:80]}
        for m in messages
    ]
