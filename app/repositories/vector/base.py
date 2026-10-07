import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.schemas.vector import EmbeddingSpace, VectorHit, VectorPoint


class VectorRepository(ABC):
    """Interface every vector database implements.

    Stores chunk embeddings and finds the closest ones, scoped to one collection.

    Chunk text and metadata stay in PostgreSQL, the source of truth. A vector store only keeps
    what similarity search needs: the vector, and the ids to filter and look chunks up by.
    Implementations subclass it: QdrantVectorRepository (default) and PgVectorRepository.
    To add a store, subclass this, implement every method, and register it in the factory.

    Each instance serves one embedding space (model + dimension), given to the constructor,
    and never mixes it with vectors from another model.
    """

    @abstractmethod
    async def init(self) -> None:
        """Connect and prepare storage for the embedding space. Called once at startup.

        Raises VectorStoreError if existing storage doesn't fit the space.
        """

    @abstractmethod
    async def upsert(self, collection_id: str, points: Sequence[VectorPoint]) -> None:
        """Insert or replace vectors. Writing the same chunk twice keeps one copy.

        Raises ValueError if a vector's size isn't the space's dimension.
        """

    @abstractmethod
    async def search(
        self, collection_id: str, query_vector: list[float], top_k: int
    ) -> list[VectorHit]:
        """Return the `top_k` closest chunks, best first. Raises ValueError on a size mismatch."""

    @abstractmethod
    async def delete_document(self, collection_id: str, document_id: str) -> None:
        """Remove the vectors of one document's chunks."""

    @abstractmethod
    async def delete_collection(self, collection_id: str) -> None:
        """Remove every vector in a collection."""

    @abstractmethod
    async def count(self, collection_id: str) -> int:
        """How many vectors a collection has."""

    @abstractmethod
    async def ping(self) -> None:
        """Raise if the store can't be reached."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections. Called once at shutdown."""


def check_dimension(space: EmbeddingSpace, vectors: Sequence[list[float]]) -> None:
    """Raise ValueError unless every vector has the space's dimension."""
    for vector in vectors:
        if len(vector) != space.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: '{space.model_id}' should produce "
                f"{space.dimension}-dimension vectors, got {len(vector)}."
            )


def model_slug(model_id: str, separator: str = "-") -> str:
    """A model id made safe for names, e.g. "fastembed:google/x-1" -> "fastembed-google-x-1"."""
    return re.sub(r"[^a-z0-9]+", separator, model_id.lower()).strip(separator)
