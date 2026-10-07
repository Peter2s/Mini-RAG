import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Entity


class Chunk(Entity):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index(None, "collection_id", "id"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_not_negative"),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata")
    embedding_model: Mapped[str | None] = mapped_column(Text)
    indexed_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
