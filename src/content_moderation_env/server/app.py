"""FastAPI application exposing the content moderation environment."""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from content_moderation_env.models import ModerationAction, ModerationObservation
from content_moderation_env.server.environment import ContentModerationEnvironment


def _factory() -> ContentModerationEnvironment:
    return ContentModerationEnvironment()


app = create_app(
    _factory,
    ModerationAction,
    ModerationObservation,
    env_name="content_moderation_env",
)


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
