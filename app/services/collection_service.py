from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.models import Collection
from app.repositories import CollectionRepository, FileRepository, VectorRepository
from app.schemas.collection import CollectionCreate, CollectionRead
from app.schemas.common import Page


class CollectionService:
    def __init__(
        self,
        session: AsyncSession,
        collections: CollectionRepository,
        files: FileRepository,
        vectors: VectorRepository,
    ) -> None:
        self.session = session
        self.collections = collections
        self.files = files
        self.vectors = vectors

    async def create(self, data: CollectionCreate) -> Collection:
        name = data.name.strip()
        if await self.collections.get_by_name(name):
            raise ConflictError(f"Collection '{name}' already exists")
        return await self.collections.add(Collection(name=name, description=data.description))

    async def get(self, collection_id: str) -> Collection:
        collection = await self.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"Collection {collection_id} not found")
        return collection

    async def list(self, offset: int, limit: int) -> Page[CollectionRead]:
        items = await self.collections.list(offset=offset, limit=limit)
        return Page[CollectionRead](
            items=[CollectionRead.model_validate(collection) for collection in items],
            total=await self.collections.count(),
            offset=offset,
            limit=limit,
        )

    async def delete(self, collection_id: str) -> None:
        """Delete a collection. Its documents and chunks go with it (ON DELETE CASCADE)."""
        collection = await self.get(collection_id)
        key = str(collection.id)
        await self.vectors.delete_collection(key)
        await self.collections.delete(collection)
        await self.session.commit()
        await self.files.delete_collection(key)
