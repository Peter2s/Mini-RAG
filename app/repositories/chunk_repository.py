from collections.abc import Iterable, Sequence

from app.db.types import MongoDocument
from app.models import Chunk, to_object_id, utcnow
from app.repositories.base import MongoRepository
from app.schemas.chunk import TextChunk


class ChunkRepository(MongoRepository[Chunk]):
    model = Chunk

    async def replace_for_document(
        self, collection_id: str, document_id: str, chunks: Sequence[TextChunk]
    ) -> int:
        """Delete a document's existing chunks and insert new ones.

        A standalone MongoDB server has no multi-document transactions, so this runs as two
        writes. If the insert fails, the caller marks the document as failed.
        """
        await self.delete_by_document(document_id)
        if chunks:
            await self.collection.insert_many(
                [
                    Chunk(
                        collection_id=collection_id,
                        document_id=document_id,
                        chunk_index=index,
                        text=chunk.text,
                        metadata=chunk.metadata,
                    ).to_mongo()
                    for index, chunk in enumerate(chunks)
                ]
            )
        return len(chunks)

    async def list_for_indexing(self, collection_id: str, embedding_model: str) -> list[Chunk]:
        """Chunks not yet indexed with `embedding_model`: new ones, or indexed by another model."""
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return []
        query: MongoDocument = {
            "collection_id": collection_object_id,
            "embedding_model": {"$ne": embedding_model},
        }
        return await self._find(query, projection={"embedding": 0})

    async def mark_indexed(self, chunk_ids: Sequence[str], embedding_model: str) -> None:
        object_ids = [object_id for chunk_id in chunk_ids if (object_id := to_object_id(chunk_id))]
        if object_ids:
            await self.collection.update_many(
                {"_id": {"$in": object_ids}},
                {"$set": {"embedding_model": embedding_model, "indexed_at": utcnow()}},
            )

    async def mark_unindexed(self, collection_id: str) -> None:
        """Flag every chunk of a collection for re-embedding."""
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is not None:
            await self.collection.update_many(
                {"collection_id": collection_object_id},
                {"$set": {"embedding_model": None, "indexed_at": None}},
            )

    async def get_many(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        """Load chunks by id, e.g. to turn vector search hits into text. Missing ids are skipped."""
        object_ids = [object_id for chunk_id in chunk_ids if (object_id := to_object_id(chunk_id))]
        if not object_ids:
            return []
        return await self._find({"_id": {"$in": object_ids}}, projection={"embedding": 0})

    async def count_by_collection(self, collection_id: str) -> int:
        collection_object_id = to_object_id(collection_id)
        return (
            0
            if collection_object_id is None
            else await self.count({"collection_id": collection_object_id})
        )

    async def delete_by_document(self, document_id: str) -> int:
        document_object_id = to_object_id(document_id)
        if document_object_id is None:
            return 0
        return (
            await self.collection.delete_many({"document_id": document_object_id})
        ).deleted_count

    async def delete_by_collection(self, collection_id: str) -> int:
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return 0
        return (
            await self.collection.delete_many({"collection_id": collection_object_id})
        ).deleted_count
