"""Typed OpenEnv WebSocket client for the content moderation environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from content_moderation_env.models import ModerationAction, ModerationObservation, ModerationState


class ContentModerationEnv(
    EnvClient[ModerationAction, ModerationObservation, ModerationState]
):
    """Client for connecting to a running content moderation server (local or HF Space)."""

    def _step_payload(self, action: ModerationAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ModerationObservation]:
        obs_data = payload.get("observation", {})
        observation = ModerationObservation(
            task=obs_data.get("task", "easy"),
            instruction=obs_data.get("instruction", ""),
            current_message=obs_data.get("current_message"),
            message_index=obs_data.get("message_index", 0),
            total_messages=obs_data.get("total_messages", 1),
            queue_context=obs_data.get("queue_context") or [],
            last_action_error=obs_data.get("last_action_error"),
            episode_score=obs_data.get("episode_score"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata={},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ModerationState:
        return ModerationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "easy"),
            sample_id=payload.get("sample_id"),
            episode_score=float(payload.get("episode_score", 0.0) or 0.0),
        )
