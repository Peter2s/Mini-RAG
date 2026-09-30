from datetime import datetime
from typing import Any, ClassVar

from pydantic import Field

from app.core.enums import MongoCollection
from app.models.base import MongoModel, ObjectIdStr, utcnow


class Chunk(MongoModel):
    collection_name: ClassVar[MongoCollection] = MongoCollection.CHUNKS
    object_id_fields: ClassVar[tuple[str, ...]] = ("collection_id", "document_id")

    collection_id: ObjectIdStr
    document_id: ObjectIdStr
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None  
    created_at: datetime = Field(default_factory=utcnow)
