from app.core.exceptions import ConflictError, NotFoundError
from app.models import Collection
from app.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    FileRepository,
    VectorRepository,
)
from app.schemas.collection import CollectionCreate, CollectionRead
from app.schemas.common import Page


class CollectionService:
    def __init__(
        self,
        collections: CollectionRepository,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        files: FileRepository,
        vectors: VectorRepository,
    ) -> None:
        self.collections = collections
        self.documents = documents
        self.chunks = chunks
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
        collection = await self.get(collection_id)
        await self.vectors.delete_collection(collection_id)
        await self.chunks.delete_by_collection(collection_id)
        await self.documents.delete_by_collection(collection_id)
        await self.collections.delete(collection)
        await self.files.delete_collection(collection_id)
