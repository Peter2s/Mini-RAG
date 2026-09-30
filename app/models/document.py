from typing import ClassVar

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
    status: DocumentStatus = DocumentStatus.UPLOADED
    error: str | None = None
