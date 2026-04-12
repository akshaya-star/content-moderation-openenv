"""
Inference Script Example
===================================
MANDATORY
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN / OPENAI_API_KEY for the LLM.
- Reads OPENENV_BASE_URL for the content moderation environment server (default http://127.0.0.1:8000).
- Uses OpenAI client for chat completions.
- Emits [START], [STEP], [END] lines exactly as specified.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# Ensure local src package is importable when run from repo root
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from content_moderation_env.client import ContentModerationEnv  # noqa: E402
from content_moderation_env.models import ModerationAction  # noqa: E402

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")
BENCHMARK = "content-moderation-openenv"


def _fmt_reward(r: float | None) -> str:
    v = float(r if r is not None else 0.0)
    return f"{v:.2f}"


def _clamp_strict(v: float) -> float:
    """Clamp to strictly (0.0, 1.0) — 0.01 min, 0.99 max."""
    return max(0.01, min(0.99, float(v)))


def _fmt_err(msg: str | None) -> str:
    if msg is None or msg == "":
        return "null"
    return msg.replace("\n", " ").strip()


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _llm_moderate(
    client: OpenAI,
    task: str,
    observation_text: str,
) -> ModerationAction:
    sys_prompt = (
        "You are a content moderation assistant. Reply with a single JSON object only, no markdown. "
        'Keys: "decision" (APPROVE|FLAG|BLOCK), "category" (spam|harassment|hate|threat|none), '
        '"severity" (low|medium|high). For easy task, only "decision" is strictly required; '
        'use category "none" and severity "low" if unsure. For medium, fill all fields. '
        "For hard, fill fields for the current message only."
    )
    user_prompt = f"Task: {task}\n{observation_text}\nOutput JSON:"
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = _extract_json(raw)
    return ModerationAction(
        decision=data.get("decision"),
        category=data.get("category"),
        severity=data.get("severity"),
    )


def _observation_prompt(task: str, obs: Any) -> str:
    parts = [f"instruction: {obs.instruction}"]
    if obs.current_message:
        parts.append(f"message_id: {obs.current_message.get('id')}")
        parts.append(f"text: {obs.current_message.get('text')}")
    if task == "hard" and obs.queue_context:
        parts.append("queue_preview: " + json.dumps(obs.queue_context))
    parts.append(f"message_index: {obs.message_index + 1} / {obs.total_messages}")
    return "\n".join(parts)


def run_task(task: str, client: OpenAI) -> Tuple[bool, int, float, List[float]]:
    """Runs one episode; prints [START] and [STEP] lines only."""
    rewards: List[float] = []
    steps = 0
    score = 0.0

    env_client = ContentModerationEnv(base_url=OPENENV_BASE_URL)
    with env_client.sync() as env:
        res = env.reset(seed=42, task=task)
        obs = res.observation
        while not obs.done:
            prompt = _observation_prompt(task, obs)
            action = _llm_moderate(client, task, prompt)
            action_str = json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))
            sr = env.step(action)
            obs = sr.observation
            steps += 1
            r = sr.reward
            if r is None and obs.reward is not None:
                r = float(obs.reward)
            elif r is None:
                r = 0.0
            rewards.append(float(r))
            print(
                f"[STEP] step={steps} action={action_str} reward={_fmt_reward(r)} "
                f"done={str(obs.done).lower()} error={_fmt_err(obs.last_action_error)}",
                flush=True,
            )
        if obs.episode_score is not None:
            score = _clamp_strict(float(obs.episode_score))
        elif rewards:
            score = _clamp_strict(float(rewards[-1]))
        else:
            score = 0.01
    return True, steps, score, rewards


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        print(
            f"[START] task={task} env={BENCHMARK} model={MODEL_NAME.lower()}",
            flush=True,
        )
        success = False
        steps = 0
        score = 0.01
        rewards: List[float] = []
        try:
            success, steps, score, rewards = run_task(task, client)
        except Exception as e:
            success = False
        score = _clamp_strict(score)
        rew_csv = ",".join(_fmt_reward(_clamp_strict(x)) for x in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rew_csv}",
            flush=True,
        )


if __name__ == "__main__":
    main()
