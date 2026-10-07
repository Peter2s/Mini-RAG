"""Brute-force vector search over embeddings stored on MongoDB chunk documents.

No extra service is needed, which suits tests and small datasets (thousands of chunks per
collection). Every search reads all of a collection's vectors, so use Qdrant beyond that.

Each chunk keeps one vector, tagged with the model that made it (`embedding_model`). Search
only reads vectors of the current model, so a model switch never mixes vector spaces.
"""

from collections.abc import Sequence

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne
from pymongo.asynchronous.database import AsyncDatabase

from app.core.enums import MongoCollection
from app.db.types import MongoDocument
from app.models import to_object_id
from app.repositories.vector.base import VectorRepository, check_dimension
from app.schemas.vector import EmbeddingSpace, VectorHit, VectorPoint

_NO_VECTOR: MongoDocument = {"embedding": None, "embedding_model": None}


class MongoVectorRepository(VectorRepository):
    def __init__(self, db: AsyncDatabase[MongoDocument], space: EmbeddingSpace) -> None:
        self._db = db
        self._space = space
        self._chunks = db[MongoCollection.CHUNKS]

    async def init(self) -> None:
        return None

    async def upsert(self, collection_id: str, points: Sequence[VectorPoint]) -> None:
        if points:
            check_dimension(self._space, [point.vector for point in points])
            await self._chunks.bulk_write(
                [
                    UpdateOne(
                        {"_id": ObjectId(point.chunk_id)},
                        {
                            "$set": {
                                "embedding": point.vector,
                                "embedding_model": self._space.model_id,
                            }
                        },
                    )
                    for point in points
                ],
                ordered=False,
            )

    async def search(
        self, collection_id: str, query_vector: list[float], top_k: int
    ) -> list[VectorHit]:
        check_dimension(self._space, [query_vector])
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return []
        chunk_rows = await self._chunks.find(
            self._vectors_of(collection_object_id), {"embedding": 1}
        ).to_list()
        if not chunk_rows:
            return []

        matrix = np.asarray([chunk_row["embedding"] for chunk_row in chunk_rows], dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        scores = np.divide(matrix @ query, norms, out=np.zeros(len(chunk_rows)), where=norms > 0)
        return [
            VectorHit(chunk_id=str(chunk_rows[position]["_id"]), score=float(scores[position]))
            for position in np.argsort(-scores)[:top_k]
        ]

    async def delete_document(self, collection_id: str, document_id: str) -> None:
        document_object_id = to_object_id(document_id)
        if document_object_id is not None:
            await self._chunks.update_many(
                {"document_id": document_object_id}, {"$set": _NO_VECTOR}
            )

    async def delete_collection(self, collection_id: str) -> None:
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is not None:
            await self._chunks.update_many(
                {"collection_id": collection_object_id}, {"$set": _NO_VECTOR}
            )

    async def count(self, collection_id: str) -> int:
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return 0
        return await self._chunks.count_documents(self._vectors_of(collection_object_id))

    async def ping(self) -> None:
        await self._db.command("ping")

    async def close(self) -> None:
        return None

    def _vectors_of(self, collection_object_id: ObjectId) -> MongoDocument:
        """Chunks of a collection holding a vector from the current model."""
        return {"collection_id": collection_object_id, "embedding_model": self._space.model_id}
