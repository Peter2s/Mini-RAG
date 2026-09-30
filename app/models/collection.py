from typing import ClassVar

from app.models.base import TimestampedModel


class Collection(TimestampedModel):
    collection_name: ClassVar[str] = "collections"

    name: str
    description: str | None = None
