from typing import Protocol

import numpy as np
from pymongo.asynchronous.database import AsyncDatabase

from app.core.enums import MongoCollection
from app.db.types import MongoDocument
from app.models import to_object_id
from app.schemas.rag import VectorMatch


class VectorRepository(Protocol):
    """Similarity search over indexed chunks. Swap in Atlas Vector Search or Qdrant here."""

    async def search(
        self, collection_id: str, query_vector: list[float], top_k: int
    ) -> list[VectorMatch]: ...


class MongoVectorRepository:
    """Brute-force cosine similarity over embeddings stored in the chunks collection.

    Fine for thousands of chunks per collection. Beyond that, use a vector index such as
    MongoDB Atlas Vector Search.
    """

    def __init__(self, db: AsyncDatabase[MongoDocument]) -> None:
        self.chunks = db[MongoCollection.CHUNKS]
        self.documents = db[MongoCollection.DOCUMENTS]

    async def search(
        self, collection_id: str, query_vector: list[float], top_k: int
    ) -> list[VectorMatch]:
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return []
        chunk_rows = await self.chunks.find(
            {"collection_id": collection_object_id, "embedding": {"$ne": None}},
            {"document_id": 1, "chunk_index": 1, "text": 1, "metadata": 1, "embedding": 1},
        ).to_list()
        if not chunk_rows:
            return []

        matrix = np.asarray([chunk_row["embedding"] for chunk_row in chunk_rows], dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: index has {matrix.shape[1]}, query has "
                f"{query.shape[0]}. Re-index the collection with reset=true."
            )

        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        scores = np.divide(matrix @ query, norms, out=np.zeros(len(chunk_rows)), where=norms > 0)
        top_chunk_rows = [
            chunk_rows[position] | {"score": float(scores[position])}
            for position in np.argsort(-scores)[:top_k]
        ]

        # Look up filenames only for the documents that made the cut.
        document_ids = list({chunk_row["document_id"] for chunk_row in top_chunk_rows})
        filenames = {
            document["_id"]: document["filename"]
            async for document in self.documents.find(
                {"_id": {"$in": document_ids}}, {"filename": 1}
            )
        }
        return [
            VectorMatch(
                chunk_id=str(chunk_row["_id"]),
                document_id=str(chunk_row["document_id"]),
                filename=filenames.get(chunk_row["document_id"], "unknown"),
                chunk_index=chunk_row["chunk_index"],
                text=chunk_row["text"],
                score=chunk_row["score"],
                metadata=chunk_row.get("metadata") or {},
            )
            for chunk_row in top_chunk_rows
        ]
