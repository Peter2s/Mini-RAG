from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, select, update

from app.models import Chunk
from app.repositories.base import BaseRepository


@dataclass(frozen=True, slots=True)
class NewChunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkRepository(BaseRepository[Chunk]):
    model = Chunk

    async def replace_for_document(
        self, collection_id: int, document_id: int, chunks: Sequence[NewChunk]
    ) -> int:
        """Delete a document's existing chunks and insert new ones in one transaction."""
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self.session.add_all(
            Chunk(
                collection_id=collection_id,
                document_id=document_id,
                chunk_index=i,
                text=chunk.text,
                chunk_metadata=chunk.metadata,
            )
            for i, chunk in enumerate(chunks)
        )
        await self.session.commit()
        return len(chunks)

    async def list_for_indexing(self, collection_id: int, only_missing: bool) -> Sequence[Chunk]:
        stmt = select(Chunk).where(Chunk.collection_id == collection_id).order_by(Chunk.id)
        if only_missing:
            stmt = stmt.where(Chunk.embedding.is_(None))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def set_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        for chunk_id, vector in embeddings.items():
            await self.session.execute(
                update(Chunk).where(Chunk.id == chunk_id).values(embedding=vector)
            )
        await self.session.commit()

    async def count_by_collection(self, collection_id: int) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.collection_id == collection_id)
        )
        return result.scalar_one()
