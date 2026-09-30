from typing import Any

from bson import ObjectId
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.models import MongoModel, to_object_id, utcnow


class MongoRepository[ModelT: MongoModel]:
    """Generic CRUD over one MongoDB collection. Invalid ids behave like missing documents."""

    model: type[ModelT]

    def __init__(self, db: AsyncDatabase) -> None:
        self.collection: AsyncCollection = db[self.model.collection_name]

    async def get(self, obj_id: str) -> ModelT | None:
        oid = to_object_id(obj_id)
        if oid is None:
            return None
        return await self._find_one({"_id": oid})

    async def add(self, obj: ModelT) -> ModelT:
        result = await self.collection.insert_one(obj.to_mongo())
        obj.id = str(result.inserted_id)
        return obj

    async def save(self, obj: ModelT) -> ModelT:
        """Replace the stored document with `obj`, refreshing `updated_at` when it has one."""
        if "updated_at" in type(obj).model_fields:
            obj.updated_at = utcnow()
        await self.collection.replace_one({"_id": ObjectId(obj.id)}, obj.to_mongo())
        return obj

    async def delete(self, obj: ModelT) -> None:
        await self.collection.delete_one({"_id": ObjectId(obj.id)})

    async def count(self, query: dict[str, Any] | None = None) -> int:
        return await self.collection.count_documents(query or {})

    async def _find_one(self, query: dict[str, Any]) -> ModelT | None:
        data = await self.collection.find_one(query)
        return self.model.from_mongo(data) if data else None

    async def _find(
        self,
        query: dict[str, Any],
        *,
        skip: int = 0,
        limit: int = 0,
        projection: dict[str, Any] | None = None,
    ) -> list[ModelT]:
        cursor = self.collection.find(query, projection).sort("_id", 1).skip(skip).limit(limit)
        return [self.model.from_mongo(data) async for data in cursor]
