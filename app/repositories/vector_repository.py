from typing import Protocol

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.schemas.rag import VectorMatch


class VectorRepository(Protocol):
    """Similarity search over indexed chunks. Swap in Qdrant/pgvector by implementing this."""

    async def search(
        self, collection_id: int, query_vector: list[float], top_k: int
    ) -> list[VectorMatch]: ...


class SQLVectorRepository:
    """Brute-force cosine similarity over embeddings stored in the chunks table.

    Fine for thousands of chunks per collection. Replace with a real vector DB beyond that.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search(
        self, collection_id: int, query_vector: list[float], top_k: int
    ) -> list[VectorMatch]:
        result = await self.session.execute(
            select(
                Chunk.id,
                Chunk.document_id,
                Document.filename,
                Chunk.chunk_index,
                Chunk.text,
                Chunk.embedding,
                Chunk.chunk_metadata,
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.collection_id == collection_id, Chunk.embedding.is_not(None))
        )
        rows = result.all()
        if not rows:
            return []

        matrix = np.asarray([row.embedding for row in rows], dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: index has {matrix.shape[1]}, query has "
                f"{query.shape[0]}. Re-index the collection with reset=true."
            )

        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        scores = np.divide(matrix @ query, norms, out=np.zeros(len(rows)), where=norms > 0)
        top = np.argsort(-scores)[:top_k]

        return [
            VectorMatch(
                chunk_id=rows[i].id,
                document_id=rows[i].document_id,
                filename=rows[i].filename,
                chunk_index=rows[i].chunk_index,
                text=rows[i].text,
                score=float(scores[i]),
                metadata=rows[i].chunk_metadata or {},
            )
            for i in top
        ]
