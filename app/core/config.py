from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import FileExtension


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "Mini-RAG"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # MongoDB
    mongodb_url: SecretStr = SecretStr("mongodb://localhost:27017")
    mongodb_database: str = "mini_rag"

    # File storage
    upload_dir: Path = Path("./data/uploads")
    max_upload_size_mb: int = 10
    allowed_extensions: list[FileExtension] = list(FileExtension)

    # Chunking
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200

    # Embeddings
    embedding_provider: Literal["hash", "voyage"] = "hash"
    embedding_dimension: int = 512  # used by the local hash provider
    embedding_batch_size: int = 64
    voyage_api_key: SecretStr | None = None
    voyage_model: str = "voyage-3.5"

    # LLM (Claude)
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 16000
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Server-side refusal fallbacks. Supported on the Claude API; disable on Bedrock/Vertex/Foundry.
    llm_enable_fallbacks: bool = True

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
