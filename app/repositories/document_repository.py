from app.models import Document, to_object_id
from app.repositories.base import MongoRepository


class DocumentRepository(MongoRepository[Document]):
    model = Document

    async def get_for_collection(self, collection_id: str, document_id: str) -> Document | None:
        cid, did = to_object_id(collection_id), to_object_id(document_id)
        if cid is None or did is None:
            return None
        return await self._find_one({"_id": did, "collection_id": cid})

    async def list_by_collection(
        self, collection_id: str, offset: int = 0, limit: int = 50
    ) -> list[Document]:
        cid = to_object_id(collection_id)
        if cid is None:
            return []
        return await self._find({"collection_id": cid}, skip=offset, limit=limit)

    async def list_all_by_collection(self, collection_id: str) -> list[Document]:
        return await self.list_by_collection(collection_id, limit=0)

    async def count_by_collection(self, collection_id: str) -> int:
        cid = to_object_id(collection_id)
        return 0 if cid is None else await self.count({"collection_id": cid})

    async def delete_by_collection(self, collection_id: str) -> int:
        cid = to_object_id(collection_id)
        if cid is None:
            return 0
        result = await self.collection.delete_many({"collection_id": cid})
        return result.deleted_count
