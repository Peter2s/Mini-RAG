import uuid

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import DocumentStatus
from app.db.base import Entity, TimestampMixin


class Document(TimestampMixin, Entity):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("collection_id", "content_hash"),
        Index(None, "collection_id", "id"),
        CheckConstraint("size_bytes >= 0", name="size_bytes_not_negative"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="content_hash_is_sha256"),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE")
    )
    filename: Mapped[str] = mapped_column(Text)
    stored_path: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="status",
            native_enum=False,
            create_constraint=True,
            length=20,
            values_callable=lambda statuses: [status.value for status in statuses],
        ),
        server_default=DocumentStatus.UPLOADED.value,
    )
    error: Mapped[str | None] = mapped_column(Text)
