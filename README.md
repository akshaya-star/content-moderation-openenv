---
title: Content Moderation Openenv
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Content Moderation OpenEnv

Train and evaluate LLM agents on **realistic content moderation**: triage user-generated text, assign harm categories, and stay consistent across a queue—tasks human moderators perform daily on social platforms, marketplaces, and support systems.

## Motivation

Harmful content spreads quickly; automated first-pass moderation reduces risk and human load. This environment turns that workflow into a reproducible benchmark with deterministic graders and shaped rewards, so researchers can compare models on the same tasks.

## Why this is useful in the real world

- Mirrors production moderation pipelines (decision → labels → queue consistency).
- Supports partial credit and trajectory-level scoring, not only a single yes/no at the end.
- Ships with JSON benchmarks and three explicit difficulty levels for curriculum evaluation.

## Environment overview

The server exposes the standard OpenEnv HTTP/WebSocket API (`reset`, `step`, `state`). Each episode samples from JSON data under `data/`. Rewards combine grader scores with step-level shaping (correct decisions, missed harm, false positives, consistency).

## Action space (`ModerationAction`)

| Field | Description |
| --- | --- |
| `decision` | `APPROVE`, `FLAG`, or `BLOCK` |
| `category` | `spam`, `harassment`, `hate`, `threat`, or `none` (required for **medium** and recommended for **hard**) |
| `severity` | `low`, `medium`, or `high` (required for **medium**) |

Invalid or missing required fields yield a penalty and, for easy/medium, end the episode with score `0.0`.

## Observation space (`ModerationObservation`)

| Field | Description |
| --- | --- |
| `task` | `easy`, `medium`, or `hard` |
| `instruction` | Task-specific brief |
| `current_message` | `{ id, text }` (labels removed from the view) |
| `message_index` / `total_messages` | Progress within the episode |
| `queue_context` | Short queue preview for the hard task |
| `done` | Episode finished |
| `reward` | Last step reward |
| `episode_score` | Final grader score in `[0.0, 1.0]` when the episode ends |
| `last_action_error` | Validation error message, if any |

## State structure (`ModerationState`)

Extends OpenEnv `State` with `task`, `sample_id`, and `episode_score` for introspection.

## Reward design

- **Correct decision**: positive contribution; **wrong on safe content**: negative; **missed harm**: stronger negative.
- **Medium**: combines grader score with label/severity shaping.
- **Hard**: per-step shaping plus a **terminal** score from the hard grader (per-message accuracy, consistency groups, false-positive and missed-harm penalties).

All grader outputs are clamped to **0.0–1.0**.

## Task descriptions

### Easy — Single-message moderation

One message per episode. The agent outputs **APPROVE**, **FLAG**, or **BLOCK**.  
**Grader**: exact match on `expected_decision` (score 0 or 1).

### Medium — Multi-label moderation

One message per episode. The agent outputs **decision**, **category**, and **severity**.  
**Grader**: `0.5` decision + `0.3` category + `0.2` severity (clamped).

### Hard — Moderation queue with consistency

Several messages per episode; one **step** per message. Similar harmful messages share a `consistency_group` in the data; aligned decisions earn a bonus.  
**Grader**: combines per-message correctness, consistency, false-positive penalty, and missed-harm penalty.

## How to run locally

```bash
cd content-moderation-openenv
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Optional: `uv lock` (recommended after changing dependencies) to refresh `uv.lock`.

```bash
openenv validate --verbose
openenv validate --url http://127.0.0.1:8000
```

## Docker

```bash
docker build -t content-moderation-openenv .
docker run --rm -p 8000:8000 -e PORT=8000 content-moderation-openenv
```

HF Spaces often set `PORT=7860`; the image respects `PORT`.

## Baseline inference (`inference.py`)

Set environment variables (see `.env.example`):

- `API_BASE_URL` — OpenAI-compatible chat completions endpoint  
- `MODEL_NAME` — Model id  
- `HF_TOKEN` or `OPENAI_API_KEY` — API key  
- `OPENENV_BASE_URL` — Base URL of **this** environment (default `http://127.0.0.1:8000`)

```bash
pip install -r requirements.txt
pip install -e .
python inference.py
```

The script prints `[START]`, one `[STEP]` per `step()`, and `[END]` per task with `score` in `[0, 1]` and comma-separated step rewards (two decimal places), as required by the hackathon spec.

## Baseline scores (example)

Recorded with a fixed seed (`42`) and a given model endpoint. Replace with your own runs:

| Task | Example score | Notes |
| --- | --- | --- |
| easy | 0.65–0.95 | Depends on model and sample draw |
| medium | 0.45–0.85 | Stricter (three fields) |
| hard | 0.35–0.75 | Multi-step + consistency |

## Hugging Face Space

Deploy this repo as a Docker Space tagged for OpenEnv. After deployment, set your Space URL here:

**Space:** `https://huggingface.co/spaces/<YOUR_ORG>/content-moderation-openenv`

Ping `GET /health` and run `openenv validate --url https://<your-space>.hf.space` (or your custom domain).

## Repository layout

- `src/content_moderation_env/` — `models.py`, `client.py`, `server/` (environment + FastAPI)
- `server/` — thin entrypoint for `openenv.yaml` (`app: server.app:app`)
- `graders/` — deterministic `grade(prediction, ground_truth) -> float`
- `data/` — `easy_samples.json`, `medium_samples.json`, `hard_samples.json`
- `inference.py` — baseline LLM runner
- `openenv.yaml`, `Dockerfile`, `pyproject.toml`, `requirements.txt`

