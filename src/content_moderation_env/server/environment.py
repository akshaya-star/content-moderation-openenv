"""Content moderation environment: reset(), step(), state()."""

from __future__ import annotations

import sys
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from content_moderation_env.models import ModerationAction, ModerationObservation, ModerationState
from content_moderation_env.server import utils

# Graders live at repo root / graders
_ROOT = utils.PROJECT_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graders.easy_grader import grade as grade_easy  # noqa: E402
from graders.hard_grader import grade as grade_hard  # noqa: E402
from graders.medium_grader import grade as grade_medium  # noqa: E402


class ContentModerationEnvironment(
    Environment[ModerationAction, ModerationObservation, ModerationState]
):
    """Simulates moderation workflows with three task difficulties."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = ModerationState(episode_id=str(uuid.uuid4()), step_count=0)
        self._task: str = "easy"
        self._episode: Dict[str, Any] = {}
        self._messages: List[Dict[str, Any]] = []
        self._cursor: int = 0
        self._hard_preds: List[Dict[str, Any]] = []
        self._last_action_error: Optional[str] = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="content-moderation-openenv",
            description="Train and evaluate agents on realistic content moderation decisions.",
            version="1.0.0",
            author="OpenEnv submission",
            documentation_url="https://meta-pytorch.org/OpenEnv/",
        )

    @property
    def state(self) -> ModerationState:
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        try:
            self._last_action_error = None
            task = kwargs.get("task", "easy")
            if task not in ("easy", "medium", "hard"):
                task = "easy"
            self._task = task
    
            samples = utils.load_json_samples(task)
            self._episode = utils.pick_episode(samples, seed, episode_id)
            self._cursor = 0
            self._hard_preds = []
    
            if task == "hard":
                self._messages = list(self._episode.get("messages") or [])
            else:
                self._messages = [self._episode]
    
            eid = self._episode.get("episode_id") or self._episode.get("id")
            self._state = ModerationState(
                episode_id=str(uuid.uuid4()),
                step_count=0,
                task=task,
                sample_id=str(eid) if eid else None,
                episode_score=0.0,
            )
    
            return self._build_obs(
                done=False,
                reward=None,
                episode_score=None,
            )
        except Exception as e:
            self._last_action_error = f"reset error: {e}"
            self._task = kwargs.get("task", "easy")
            return ModerationObservation(
                done=True,
                reward=0.0,
                task=self._task,
                instruction="Error",
                current_message=None,
                message_index=0,
                total_messages=1,
                queue_context=[],
                last_action_error=self._last_action_error,
                episode_score=0.0,
            )

    def step(
        self,
        action: ModerationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        try:
            self._state.step_count += 1
            self._last_action_error = None
    
            err = self._validate_action(action)
            if err:
                self._last_action_error = err
                if self._task in ("easy", "medium"):
                    self._state.episode_score = 0.0
                    return self._build_obs(
                        done=True,
                        reward=-0.05,
                        episode_score=0.0,
                    )
                return self._build_obs(
                    done=False,
                    reward=-0.05,
                    episode_score=None,
                )
    
            d = utils.normalize_decision(action.decision)
            cat = utils.normalize_category(action.category)
            sev = utils.normalize_severity(action.severity)
    
            if self._task in ("easy", "medium"):
                return self._step_single(d, cat, sev)
    
            return self._step_hard(d, cat, sev)
        except Exception as e:
            self._last_action_error = f"step error: {e}"
            return self._build_obs(done=True, reward=0.0, episode_score=0.0)

    def _validate_action(self, action: ModerationAction) -> Optional[str]:
        d = utils.normalize_decision(action.decision)
        if d is None:
            return "invalid or missing decision"
        if self._task == "medium":
            if utils.normalize_category(action.category) is None:
                return "invalid or missing category"
            if utils.normalize_severity(action.severity) is None:
                return "invalid or missing severity"
        return None

    def _step_single(
        self,
        decision: Optional[str],
        category: Optional[str],
        severity: Optional[str],
    ) -> ModerationObservation:
        msg = self._messages[0]
        gt_dec = (msg.get("expected_decision") or "").strip().upper()
        is_safe = gt_dec == "APPROVE"
        labels = msg.get("labels") or []

        pred = {
            "decision": decision,
            "category": category,
            "severity": severity,
        }
        if self._task == "easy":
            gt = {"expected_decision": gt_dec}
            score = grade_easy(pred, gt)
        else:
            gt = {
                "expected_decision": gt_dec,
                "labels": labels,
                "severity": msg.get("severity") or "",
            }
            score = grade_medium(pred, gt)

        primary_cat = (
            utils.normalize_category(labels[0])
            if labels
            else "none"
        )
        shaped = utils.step_reward_shaping(
            task=self._task,
            pred_decision=decision,
            gt_decision=gt_dec,
            is_safe_content=is_safe,
            pred_category=category,
            gt_category=primary_cat,
            pred_severity=severity,
            gt_severity=utils.normalize_severity(msg.get("severity")),
        )
        shaped_clamped = max(0.0, shaped)
        reward = float(max(0.0, min(1.0, 0.5 * score + 0.5 * shaped_clamped)))

        self._state.episode_score = score
        return self._build_obs(
            done=True,
            reward=reward,
            episode_score=score,
        )

    def _step_hard(
        self,
        decision: Optional[str],
        category: Optional[str],
        severity: Optional[str],
    ) -> ModerationObservation:
        if self._cursor >= len(self._messages):
            self._last_action_error = "episode already finished"
            return self._build_obs(done=True, reward=-0.05, episode_score=0.0)

        msg = self._messages[self._cursor]
        mid = msg["id"]
        gt_dec = (msg.get("expected_decision") or "").strip().upper()
        is_safe = bool(msg.get("is_safe", gt_dec == "APPROVE"))

        self._hard_preds.append(
            {
                "id": mid,
                "decision": decision,
                "category": category,
                "severity": severity,
            }
        )

        shaped = utils.step_reward_shaping(
            task="hard",
            pred_decision=decision,
            gt_decision=gt_dec,
            is_safe_content=is_safe,
        )

        self._cursor += 1
        done = self._cursor >= len(self._messages)

        episode_score: Optional[float] = None
        if done:
            pred_full = {"decisions": self._hard_preds}
            gt_full = {"messages": self._messages}
            episode_score = grade_hard(pred_full, gt_full)
            self._state.episode_score = episode_score
            reward = float(max(-1.0, min(1.0, shaped * 0.4 + episode_score * 0.6)))
        else:
            reward = float(max(-1.0, min(1.0, shaped)))

        return self._build_obs(
            done=done,
            reward=reward,
            episode_score=episode_score,
        )

    def _build_obs(
        self,
        *,
        done: bool,
        reward: Optional[float],
        episode_score: Optional[float],
    ) -> ModerationObservation:
        if self._task == "hard" and self._messages:
            if self._cursor < len(self._messages):
                cur = self._messages[self._cursor]
                cm = utils.messages_public_view(cur)
            else:
                cm = None
            qc = utils.hard_queue_context(self._messages)
        elif self._messages:
            cm = utils.messages_public_view(self._messages[0])
            qc = []
        else:
            cm = None
            qc = []

        instr = {
            "easy": "Return one moderation decision: APPROVE, FLAG, or BLOCK for the message.",
            "medium": "Return decision, category (spam|harassment|hate|threat|none), and severity (low|medium|high).",
            "hard": "Moderate each message in order. Be consistent on similar harmful content; avoid blocking safe text.",
        }[self._task]

        return ModerationObservation(
            done=done,
            reward=reward,
            task=self._task,
            instruction=instr,
            current_message=cm,
            message_index=(
                min(self._cursor, len(self._messages) - 1)
                if self._messages
                else 0
            ),
            total_messages=max(len(self._messages), 1),
            queue_context=qc,
            last_action_error=self._last_action_error,
            episode_score=episode_score,
        )

    def close(self) -> None:
        return None
