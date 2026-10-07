from collections.abc import Sequence

from app.models import Collection
from app.repositories.base import SqlRepository


class CollectionRepository(SqlRepository[Collection]):
    model = Collection

    async def get_by_name(self, name: str) -> Collection | None:
        return await self._find_one(Collection.name == name)

    async def list(self, offset: int = 0, limit: int = 50) -> Sequence[Collection]:
        return await self._find(offset=offset, limit=limit)

    def conflict_message(self, entity: Collection) -> str:
        return f"Collection '{entity.name}' already exists"
