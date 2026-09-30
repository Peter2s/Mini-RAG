from typing import Protocol

import numpy as np
from pymongo.asynchronous.database import AsyncDatabase

from app.models import Chunk, Document, to_object_id
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

    def __init__(self, db: AsyncDatabase) -> None:
        self.chunks = db[Chunk.collection_name]
        self.documents = db[Document.collection_name]

    async def search(
        self, collection_id: str, query_vector: list[float], top_k: int
    ) -> list[VectorMatch]:
        cid = to_object_id(collection_id)
        if cid is None:
            return []
        rows = await self.chunks.find(
            {"collection_id": cid, "embedding": {"$ne": None}},
            {"document_id": 1, "chunk_index": 1, "text": 1, "metadata": 1, "embedding": 1},
        ).to_list()
        if not rows:
            return []

        matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: index has {matrix.shape[1]}, query has "
                f"{query.shape[0]}. Re-index the collection with reset=true."
            )

        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        scores = np.divide(matrix @ query, norms, out=np.zeros(len(rows)), where=norms > 0)
        top = [rows[i] | {"score": float(scores[i])} for i in np.argsort(-scores)[:top_k]]

        # Look up filenames only for the documents that made the cut.
        document_ids = list({row["document_id"] for row in top})
        filenames = {
            doc["_id"]: doc["filename"]
            async for doc in self.documents.find({"_id": {"$in": document_ids}}, {"filename": 1})
        }
        return [
            VectorMatch(
                chunk_id=str(row["_id"]),
                document_id=str(row["document_id"]),
                filename=filenames.get(row["document_id"], "unknown"),
                chunk_index=row["chunk_index"],
                text=row["text"],
                score=row["score"],
                metadata=row.get("metadata") or {},
            )
            for row in top
        ]
