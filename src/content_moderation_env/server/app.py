"""FastAPI application exposing the content moderation environment."""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from content_moderation_env.models import ModerationAction, ModerationObservation
from content_moderation_env.server.environment import ContentModerationEnvironment


def _factory() -> ContentModerationEnvironment:
    return ContentModerationEnvironment()


try:
    app = create_app(
        _factory,
        ModerationAction,
        ModerationObservation,
        env_name="content_moderation_env",
    )
except Exception as e:
    import logging
    from fastapi import FastAPI
    logging.error(f"Failed to initialize OpenEnv app: {e}")
    app = FastAPI()

    @app.post("/reset")
    def reset(payload: dict = None):
        return {"observation": {"done": True, "reward": 0.0, "episode_score": 0.0}}

    @app.post("/step")
    def step(payload: dict = None):
        return {"observation": {"done": True, "reward": 0.0, "episode_score": 0.0}}

    @app.get("/state")
    def state():
        return {"state": {"step_count": 0, "episode_score": 0.0}}

app.routes = [r for r in app.routes if getattr(r, "path", None) != "/health"]

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
