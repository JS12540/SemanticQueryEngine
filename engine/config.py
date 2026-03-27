"""
engine/config.py
================
Central configuration object – reads all settings from environment variables
(loaded from .env by python-dotenv).

Usage
-----
    from engine.config import Settings

    cfg = Settings()          # reads from environment / .env
    print(cfg.openai_model)   # "gpt-4o"

All fields have sensible defaults so the application starts with only
OPENAI_API_KEY set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """Return the repository root (parent of engine/)."""
    return Path(__file__).parent.parent


@dataclass
class Settings:
    """
    Immutable configuration bag populated from environment variables.

    Attributes
    ----------
    openai_api_key    : OpenAI API key (required)
    openai_model      : Model ID to use (default: gpt-4o)
    db_path           : Path to cashflo_sample.db
    semantic_yaml     : Path to semantic_layer.yaml
    cache_db_path     : Path to query_cache.db
    max_retries       : LLM self-correction retries on SQL error
    use_cache         : Whether to use the query cache
    streamlit_port    : Streamlit server port
    """

    openai_api_key: str = field(default="")
    openai_model: str = field(default="gpt-4.1-mini")
    db_path: Path = field(default_factory=lambda: _project_root() / "cashflo_sample.db")
    semantic_yaml: Path = field(default_factory=lambda: _project_root() / "semantic_layer.yaml")
    cache_db_path: Path = field(default_factory=lambda: _project_root() / "query_cache.db")
    max_retries: int = field(default=1)
    use_cache: bool = field(default=True)
    streamlit_port: int = field(default=8501)

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        """
        Build a Settings instance from environment variables.

        Parameters
        ----------
        env_file : optional path to a .env file.  Defaults to the project-root .env.
        """
        if env_file is None:
            env_file = _project_root() / ".env"
        load_dotenv(env_file, override=False)  # don't override already-set vars

        root = _project_root()

        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_model="gpt-4.1-mini",
            db_path=Path(os.environ.get("CASHFLO_DB", str(root / "cashflo_sample.db"))),
            semantic_yaml=Path(os.environ.get("CASHFLO_YAML", str(root / "semantic_layer.yaml"))),
            cache_db_path=Path(os.environ.get("CASHFLO_CACHE_DB", str(root / "query_cache.db"))),
            max_retries=int(os.environ.get("CASHFLO_MAX_RETRIES", "1")),
            use_cache=os.environ.get("CASHFLO_NO_CACHE", "false").lower() != "true",
            streamlit_port=int(os.environ.get("STREAMLIT_SERVER_PORT", "8501")),
        )

    def validate(self) -> None:
        """Raise ValueError if any required setting is missing or invalid."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set.\n"
                "Add it to your .env file or export it:  export OPENAI_API_KEY=sk-proj-..."
            )
        if not self.db_path.exists():
            raise ValueError(
                f"Database not found: {self.db_path}\n"
                "Run the setup script to create it from the SQL dump."
            )
        if not self.semantic_yaml.exists():
            raise ValueError(f"Semantic layer YAML not found: {self.semantic_yaml}")

    def __repr__(self) -> str:  # pragma: no cover
        key_preview = (
            f"{self.openai_api_key[:8]}..." if self.openai_api_key else "<not set>"
        )
        return (
            f"Settings(model={self.openai_model!r}, "
            f"db={self.db_path.name!r}, "
            f"cache={self.use_cache}, "
            f"key={key_preview})"
        )
