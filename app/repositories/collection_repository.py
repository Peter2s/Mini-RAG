from pymongo.errors import DuplicateKeyError

from app.core.exceptions import ConflictError
from app.models import Collection
from app.repositories.base import MongoRepository


class CollectionRepository(MongoRepository[Collection]):
    model = Collection

    async def get_by_name(self, name: str) -> Collection | None:
        return await self._find_one({"name": name})

    async def add(self, entity: Collection) -> Collection:
        try:
            return await super().add(entity)
        except DuplicateKeyError as error:  # unique index on name
            raise ConflictError(f"Collection '{entity.name}' already exists") from error

    async def list(self, offset: int = 0, limit: int = 50) -> list[Collection]:
        return await self._find({}, skip=offset, limit=limit)
