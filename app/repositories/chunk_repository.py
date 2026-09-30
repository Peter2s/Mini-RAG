from collections.abc import Sequence

from bson import ObjectId
from pymongo import UpdateOne

from app.models import Chunk, to_object_id
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

    async def list_for_indexing(self, collection_id: str, only_missing: bool) -> list[Chunk]:
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return []
        query = {"collection_id": collection_object_id}
        if only_missing:
            query["embedding"] = None
        # Existing embeddings aren't needed to re-embed, so don't transfer them.
        return await self._find(query, projection={"embedding": 0})

    async def set_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        if embeddings:
            await self.collection.bulk_write(
                [
                    UpdateOne({"_id": ObjectId(chunk_id)}, {"$set": {"embedding": vector}})
                    for chunk_id, vector in embeddings.items()
                ],
                ordered=False,
            )

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
