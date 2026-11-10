"""
Loads API keys from the project's env files.

key.env is read first, then .env. Variables that are already set in the
environment are never overwritten. Both files are git-ignored; see
.env.example for the variable names.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILES = ("key.env", ".env")

_loaded = False


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not set."""


def load_env() -> None:
    """Loads key.env and .env from the project root (once)."""
    global _loaded
    if _loaded:
        return
    for name in ENV_FILES:
        path = PROJECT_ROOT / name
        if path.is_file():
            load_dotenv(path, override=False)
    _loaded = True


def get_key(name: str) -> Optional[str]:
    """Returns the value of an env variable (loading the env files first), or None."""
    load_env()
    value = os.environ.get(name, "").strip()
    return value or None


def require_key(name: str) -> str:
    """Like get_key, but raises MissingAPIKeyError if the variable is not set."""
    value = get_key(name)
    if value is None:
        raise MissingAPIKeyError(
            f"{name} is not set. Add it to key.env or .env in the project root (see .env.example)."
        )
    return value
