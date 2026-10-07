import math
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import delete, func, insert, update

from app.models import Chunk, EntityId, parse_id
from app.repositories.base import SqlRepository
from app.schemas.chunk import TextChunk


class ChunkRepository(SqlRepository[Chunk]):
    model = Chunk

    async def replace_for_document(
        self, collection_id: uuid.UUID, document_id: uuid.UUID, chunks: Sequence[TextChunk]
    ) -> int:
        """Swap a document's chunks for new ones, all or nothing.

        A savepoint keeps the session usable if this fails, so the caller can still record
        the failure on the document.
        """
        async with self.session.begin_nested():
            await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            if chunks:
                await self.session.execute(
                    insert(Chunk),
                    [
                        {
                            "collection_id": collection_id,
                            "document_id": document_id,
                            "chunk_index": index,
                            "text": _storable(chunk.text),
                            "meta": _storable(chunk.metadata),
                        }
                        for index, chunk in enumerate(chunks)
                    ],
                )
        return len(chunks)

    async def list_for_indexing(
        self, collection_id: EntityId, embedding_model: str
    ) -> Sequence[Chunk]:
        """Chunks not yet indexed with `embedding_model`: new ones, or indexed by another model."""
        parsed = parse_id(collection_id)
        if parsed is None:
            return []
        return await self._find(
            Chunk.collection_id == parsed, Chunk.embedding_model.is_distinct_from(embedding_model)
        )

    async def mark_indexed(self, chunk_ids: Iterable[uuid.UUID], embedding_model: str) -> None:
        ids = list(chunk_ids)
        if ids:
            await self.session.execute(
                update(Chunk)
                .where(Chunk.id.in_(ids))
                .values(embedding_model=embedding_model, indexed_at=func.now()),
                execution_options={"synchronize_session": False},
            )

    async def mark_unindexed(self, collection_id: uuid.UUID) -> None:
        """Flag every chunk of a collection for re-embedding."""
        await self.session.execute(
            update(Chunk)
            .where(Chunk.collection_id == collection_id)
            .values(embedding_model=None, indexed_at=None),
            execution_options={"synchronize_session": False},
        )

    async def get_many(self, chunk_ids: Iterable[EntityId]) -> Sequence[Chunk]:
        """Load chunks by id, e.g. to turn vector search hits into text. Missing ids are skipped."""
        ids = [parsed for chunk_id in chunk_ids if (parsed := parse_id(chunk_id))]
        if not ids:
            return []
        return await self._find(Chunk.id.in_(ids))

    async def count_by_collection(self, collection_id: EntityId) -> int:
        parsed = parse_id(collection_id)
        return 0 if parsed is None else await self.count(Chunk.collection_id == parsed)


def _storable(value: Any) -> Any:
    """Make loader output storable: PostgreSQL rejects NUL characters and non-finite JSON."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {_storable(str(key)): _storable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_storable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, bool | int | float):
        return value
    return _storable(str(value))
