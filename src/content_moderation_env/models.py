"""Typed Pydantic models for the content moderation OpenEnv."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ModerationAction(Action):
    """Agent action: moderation decision and optional labels."""

    decision: Optional[str] = Field(
        default=None,
        description="APPROVE, FLAG, or BLOCK",
    )
    category: Optional[str] = Field(
        default=None,
        description="For medium task: spam, harassment, hate, threat, or none",
    )
    severity: Optional[str] = Field(
        default=None,
        description="For medium task: low, medium, high",
    )


class ModerationObservation(Observation):
    """What the agent sees after reset or step."""

    task: str = Field(..., description="easy | medium | hard")
    instruction: str = ""
    current_message: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Single message under review {id, text, ...}",
    )
    message_index: int = 0
    total_messages: int = 1
    queue_context: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="For hard task: lightweight view of the queue",
    )
    last_action_error: Optional[str] = None
    episode_score: Optional[float] = Field(
        default=None,
        description="Final normalized score in [0,1] when episode ends",
    )


class ModerationState(State):
    """Server-side episode state (State allows extra fields)."""

    task: str = "easy"
    sample_id: Optional[str] = None
    episode_score: float = 0.0


class ModerationInfo(BaseModel):
    """Optional structured episode metadata for clients (not sent by default)."""

    episode_id: Optional[str] = None
    task: str = "easy"
    sample_id: Optional[str] = None
