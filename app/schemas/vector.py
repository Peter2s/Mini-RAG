from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """One chunk's embedding, ready to be written to a vector store."""

    chunk_id: str
    document_id: str
    vector: list[float]


@dataclass(frozen=True, slots=True)
class VectorHit:
    """A search result from a vector store: which chunk matched, and how closely."""

    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """Which model made a set of vectors, and their size.

    Vectors are only comparable within one space, so vector stores keep spaces apart.
    """

    model_id: str
    dimension: int
