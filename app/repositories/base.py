from typing import Any

from bson import ObjectId
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.db.types import MongoDocument
from app.models import MongoModel, to_object_id, utcnow
from app.models.base import TimestampedModel


class MongoRepository[ModelT: MongoModel]:
    """Generic CRUD over one MongoDB collection. Invalid ids behave like missing documents."""

    model: type[ModelT]

    def __init__(self, db: AsyncDatabase[MongoDocument]) -> None:
        self.collection: AsyncCollection[MongoDocument] = db[self.model.collection_name]

    async def get(self, entity_id: str) -> ModelT | None:
        object_id = to_object_id(entity_id)
        if object_id is None:
            return None
        return await self._find_one({"_id": object_id})

    async def add(self, entity: ModelT) -> ModelT:
        await self.collection.insert_one(entity.to_mongo())
        return entity

    async def save(self, entity: ModelT) -> ModelT:
        """Replace the stored document with `obj`, refreshing `updated_at` when it has one."""
        if isinstance(entity, TimestampedModel):
            entity.updated_at = utcnow()
        await self.collection.replace_one({"_id": ObjectId(entity.id)}, entity.to_mongo())
        return entity

    async def delete(self, entity: ModelT) -> None:
        await self.collection.delete_one({"_id": ObjectId(entity.id)})

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
