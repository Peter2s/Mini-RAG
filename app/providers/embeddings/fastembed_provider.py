"""Free, local embeddings through FastEmbed (ONNX Runtime, no PyTorch needed).

The default model, google/embeddinggemma-300m, covers 100+ languages including Arabic and
matches English questions to Arabic passages. Models are downloaded once into `models_dir`.
"""

import asyncio
import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.concurrency import run_in_threadpool

from app.core.exceptions import EmbeddingError
from app.providers.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

_PREFIXES: dict[str, tuple[str, str]] = {
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "Qwen/Qwen3-Embedding-0.6B": (
        "Instruct: Given a question, retrieve passages that answer it\nQuery:",
        "",
    ),
    "Qwen/Qwen3-Embedding-0.6B-Q": (
        "Instruct: Given a question, retrieve passages that answer it\nQuery:",
        "",
    ),
}
_SUPPORT_FILES = ["*.json", "*.txt", "*.model"]


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str, models_dir: Path, threads: int | None = None) -> None:
        self._model_name = model_name
        self._models_dir = models_dir
        self._threads = threads
        self._query_prefix, self._document_prefix = _PREFIXES.get(model_name, ("", ""))
        self._model: TextEmbedding | None = None
        self._lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return f"fastembed:{self._model_name}"

    @cached_property
    def dimension(self) -> int:
        dimension: int = self._description["dim"]
        return dimension

    async def warmup(self) -> None:
        """Download and load the model ahead of the first request."""
        await self._get_model()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([self._document_prefix + text for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self._embed([self._query_prefix + text])
        return vector

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        model = await self._get_model()
        return await run_in_threadpool(_embed, model, texts)

    async def aclose(self) -> None:
        self._model = None

    async def _get_model(self) -> "TextEmbedding":
        async with self._lock:
            if self._model is None:
                try:
                    self._model = await run_in_threadpool(self._load)
                except Exception as error:
                    raise EmbeddingError(
                        f"Could not load embedding model '{self._model_name}': {error}"
                    ) from error
            return self._model

    def _load(self) -> "TextEmbedding":
        from fastembed import TextEmbedding
        from huggingface_hub import snapshot_download

        description = self._description
        repository = description["sources"]["hf"]
        local_dir = self._models_dir / self._model_name.replace("/", "--")
        weights = [description["model_file"], *description.get("additional_files", [])]
        if not all((local_dir / weight).is_file() for weight in weights):
            logger.info("Downloading embedding model %s into %s", self._model_name, local_dir)
            snapshot_download(
                repo_id=repository,
                local_dir=local_dir,
                allow_patterns=[*weights, *_SUPPORT_FILES],
            )
        return TextEmbedding(
            self._model_name, specific_model_path=str(local_dir), threads=self._threads
        )

    @cached_property
    def _description(self) -> dict[str, Any]:
        """The model's entry in FastEmbed's registry: weight files, source, and dimension."""
        from fastembed import TextEmbedding

        for description in TextEmbedding.list_supported_models():
            if description["model"] == self._model_name:
                return description
        raise EmbeddingError(
            f"FastEmbed doesn't support '{self._model_name}'. "
            "See TextEmbedding.list_supported_models() for the available models."
        )


def _embed(model: "TextEmbedding", texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in model.embed(texts)]
