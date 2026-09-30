from app.models import Document, to_object_id
from app.repositories.base import MongoRepository


class DocumentRepository(MongoRepository[Document]):
    model = Document

    async def get_for_collection(self, collection_id: str, document_id: str) -> Document | None:
        collection_object_id, document_object_id = (
            to_object_id(collection_id),
            to_object_id(document_id),
        )
        if collection_object_id is None or document_object_id is None:
            return None
        return await self._find_one(
            {"_id": document_object_id, "collection_id": collection_object_id}
        )

    async def list_by_collection(
        self, collection_id: str, offset: int = 0, limit: int = 50
    ) -> list[Document]:
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return []
        return await self._find({"collection_id": collection_object_id}, skip=offset, limit=limit)

    async def list_all_by_collection(self, collection_id: str) -> list[Document]:
        return await self.list_by_collection(collection_id, limit=0)

    async def count_by_collection(self, collection_id: str) -> int:
        collection_object_id = to_object_id(collection_id)
        return (
            0
            if collection_object_id is None
            else await self.count({"collection_id": collection_object_id})
        )

    async def delete_by_collection(self, collection_id: str) -> int:
        collection_object_id = to_object_id(collection_id)
        if collection_object_id is None:
            return 0
        result = await self.collection.delete_many({"collection_id": collection_object_id})
        return result.deleted_count
