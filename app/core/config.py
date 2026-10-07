from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import EmbeddingProviderName, FileExtension, VectorStoreName


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Mini-RAG"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    mongodb_url: SecretStr = SecretStr("mongodb://localhost:27017")
    mongodb_database: str = "mini_rag"

    vector_store: VectorStoreName = VectorStoreName.QDRANT
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "chunks"

    upload_dir: Path = Path("./data/uploads")
    max_upload_size_mb: int = 10
    allowed_extensions: list[FileExtension] = list(FileExtension)

    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200

    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.FASTEMBED
    embedding_dimension: int = 512
    embedding_batch_size: int = 64
    fastembed_model: str = "google/embeddinggemma-300m"
    fastembed_threads: int | None = None
    models_dir: Path = Path("./data/models")
    voyage_api_key: SecretStr | None = None
    voyage_model: str = "voyage-3.5"
    voyage_dimension: int = 1024

    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 16000
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_enable_fallbacks: bool = True

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
