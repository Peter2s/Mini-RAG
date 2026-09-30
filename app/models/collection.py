from typing import ClassVar

from app.core.enums import MongoCollection
from app.models.base import TimestampedModel


class Collection(TimestampedModel):
    collection_name: ClassVar[MongoCollection] = MongoCollection.COLLECTIONS

    name: str
    description: str | None = None
