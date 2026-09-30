from collections.abc import Sequence

from sqlalchemy import select

from app.models import Collection
from app.repositories.base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    model = Collection

    async def get_by_name(self, name: str) -> Collection | None:
        result = await self.session.execute(select(Collection).where(Collection.name == name))
        return result.scalar_one_or_none()

    async def list(self, offset: int = 0, limit: int = 50) -> Sequence[Collection]:
        result = await self.session.execute(
            select(Collection).order_by(Collection.id).offset(offset).limit(limit)
        )
        return result.scalars().all()
