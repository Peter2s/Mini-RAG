from typing import Literal

from app.core.exceptions import EmbeddingError
from app.providers.embeddings.base import EmbeddingProvider


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Semantic embeddings from Voyage AI. Requires `pip install voyageai` and VOYAGE_API_KEY."""

    def __init__(self, api_key: str | None, model: str, dimension: int) -> None:
        try:
            import voyageai
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "EMBEDDING_PROVIDER=voyage requires the `voyageai` package: pip install voyageai"
            ) from error
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._model = model
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"voyage:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "document")

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self._embed([text], "query")
        return vector

    async def _embed(
        self, texts: list[str], input_type: Literal["document", "query"]
    ) -> list[list[float]]:
        try:
            result = await self._client.embed(texts, model=self._model, input_type=input_type)
        except Exception as error:
            raise EmbeddingError(f"Voyage embedding request failed: {error}") from error
        embeddings: list[list[float]] = result.embeddings
        return embeddings

    async def warmup(self) -> None:
        return None

    async def aclose(self) -> None:
        return None
