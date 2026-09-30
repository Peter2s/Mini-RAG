from typing import ClassVar

from pydantic import Field

from app.core.enums import DocumentStatus, MongoCollection
from app.models.base import ObjectIdStr, TimestampedModel


class Document(TimestampedModel):
    collection_name: ClassVar[MongoCollection] = MongoCollection.DOCUMENTS
    object_id_fields: ClassVar[tuple[str, ...]] = ("collection_id",)

    collection_id: ObjectIdStr
    filename: str
    stored_path: str
    content_type: str | None = None
    size_bytes: int
    # SHA-256 of the file's bytes. Unique per collection, so the same content can't be
    # uploaded twice, whatever the file is called.
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: DocumentStatus = DocumentStatus.UPLOADED
    error: str | None = None
