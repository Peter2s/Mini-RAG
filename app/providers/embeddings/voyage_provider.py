from app.core.exceptions import EmbeddingError
from app.providers.embeddings.base import InputType


class VoyageEmbeddingProvider:
    """Semantic embeddings from Voyage AI. Requires `pip install voyageai` and VOYAGE_API_KEY."""

    name = "voyage"

    def __init__(self, api_key: str | None, model: str) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "EMBEDDING_PROVIDER=voyage requires the `voyageai` package: pip install voyageai"
            ) from exc
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        try:
            result = await self._client.embed(texts, model=self._model, input_type=input_type)
        except Exception as exc:  # voyageai raises several unrelated error types
            raise EmbeddingError(f"Voyage embedding request failed: {exc}") from exc
        return result.embeddings

    async def aclose(self) -> None:
        return None
