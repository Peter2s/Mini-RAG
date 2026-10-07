from abc import ABC, abstractmethod

from app.schemas.vector import EmbeddingSpace


class EmbeddingProvider(ABC):
    """Interface every embedding model implements.

    One instance is built at startup and shared by every request, since loading a model is
    expensive. Implementations subclass it: FastEmbedProvider (default), VoyageEmbeddingProvider,
    and HashEmbeddingProvider. To add one, subclass this and register it in the factory.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Names the model, e.g. "fastembed:google/embeddinggemma-300m".

        Vectors from different model ids never mix: each gets its own space in the vector store.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Size of every vector the model returns. Known without loading the model."""

    @property
    def space(self) -> EmbeddingSpace:
        return EmbeddingSpace(model_id=self.model_id, dimension=self.dimension)

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages to store, one vector per text, in order."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Retrieval models encode queries and passages differently."""

    @abstractmethod
    async def warmup(self) -> None:
        """Prepare the model (e.g. download it) before the first request."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release resources at shutdown."""
