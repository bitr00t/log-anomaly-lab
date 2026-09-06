"""Central configuration, loaded from environment variables / .env.

No API keys required anywhere in this project: embeddings and all detectors
run locally. Only Qdrant needs to be reachable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Vector database -------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection: str = "logs"

    # --- Embeddings -------------------------------------------------------
    # MiniLM is a good default for short log lines: fast on CPU, and log
    # messages rarely exceed its context window.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Paths ------------------------------------------------------------
    logs_file: Path = Path("data/logs.jsonl")
    results_dir: Path = Path("results")


settings = Settings()
