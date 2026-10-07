import uuid
from collections.abc import Iterable, Sequence

from sqlalchemy import select

from app.models import Document, EntityId, parse_id
from app.repositories.base import SqlRepository


class DocumentRepository(SqlRepository[Document]):
    model = Document

    def conflict_message(self, entity: Document) -> str:
        return "A file with the same content is already in this collection"

    async def filenames(self, document_ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, str]:
        """Map document ids to their original filenames."""
        ids = list(document_ids)
        if not ids:
            return {}
        rows = await self.session.execute(
            select(Document.id, Document.filename).where(Document.id.in_(ids))
        )
        return dict(rows.tuples().all())

    async def get_by_content_hash(
        self, collection_id: EntityId, content_hash: str
    ) -> Document | None:
        parsed = parse_id(collection_id)
        if parsed is None:
            return None
        return await self._find_one(
            Document.collection_id == parsed, Document.content_hash == content_hash
        )

    async def get_for_collection(
        self, collection_id: EntityId, document_id: EntityId
    ) -> Document | None:
        parsed_collection, parsed_document = parse_id(collection_id), parse_id(document_id)
        if parsed_collection is None or parsed_document is None:
            return None
        return await self._find_one(
            Document.id == parsed_document, Document.collection_id == parsed_collection
        )

    async def list_by_collection(
        self, collection_id: EntityId, offset: int = 0, limit: int = 50
    ) -> Sequence[Document]:
        parsed = parse_id(collection_id)
        if parsed is None:
            return []
        return await self._find(Document.collection_id == parsed, offset=offset, limit=limit)

    async def list_all_by_collection(self, collection_id: EntityId) -> Sequence[Document]:
        return await self.list_by_collection(collection_id, limit=0)

    async def count_by_collection(self, collection_id: EntityId) -> int:
        parsed = parse_id(collection_id)
        return 0 if parsed is None else await self.count(Document.collection_id == parsed)
