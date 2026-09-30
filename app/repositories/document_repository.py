from collections.abc import Sequence

from sqlalchemy import func, select

from app.models import Document
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    async def get_for_collection(self, collection_id: int, document_id: int) -> Document | None:
        result = await self.session.execute(
            select(Document).where(
                Document.id == document_id, Document.collection_id == collection_id
            )
        )
        return result.scalar_one_or_none()

    async def list_by_collection(
        self, collection_id: int, offset: int = 0, limit: int = 50
    ) -> Sequence[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.collection_id == collection_id)
            .order_by(Document.id)
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()

    async def list_all_by_collection(self, collection_id: int) -> Sequence[Document]:
        result = await self.session.execute(
            select(Document).where(Document.collection_id == collection_id).order_by(Document.id)
        )
        return result.scalars().all()

    async def count_by_collection(self, collection_id: int) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.collection_id == collection_id)
        )
        return result.scalar_one()
