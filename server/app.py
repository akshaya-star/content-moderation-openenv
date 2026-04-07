"""
OpenEnv entrypoint expected by `openenv.yaml` (app: server.app:app).

Delegates to the implementation under `src/content_moderation_env`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from content_moderation_env.server.app import app as _app  # noqa: E402
from content_moderation_env.server.app import main as _impl_main  # noqa: E402

app = _app


def main() -> None:
    """Uvicorn entrypoint for `python -m openenv` / setuptools scripts."""
    _impl_main()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
