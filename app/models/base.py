from datetime import UTC, datetime
from typing import Annotated, Any, ClassVar, Self

from bson import ObjectId
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from app.core.enums import MongoCollection


def utcnow() -> datetime:
    return datetime.now(UTC)


def _object_id_to_str(value: Any) -> Any:
    return str(value) if isinstance(value, ObjectId) else value


ObjectIdStr = Annotated[str, BeforeValidator(_object_id_to_str)]


def to_object_id(value: str) -> ObjectId | None:
    """Parse a 24-character hex id. Anything else returns None, which callers treat as not found."""
    if isinstance(value, str) and len(value) == 24 and ObjectId.is_valid(value):
        return ObjectId(value)
    return None


class MongoModel(BaseModel):
    """A document stored in MongoDB. `id` maps to `_id`; id fields are stored as ObjectIds."""

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

    collection_name: ClassVar[MongoCollection]
    object_id_fields: ClassVar[tuple[str, ...]] = ()

    id: ObjectIdStr = Field(default_factory=lambda: str(ObjectId()))

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> Self:
        data = dict(data)
        data["id"] = data.pop("_id")
        return cls.model_validate(data)

    def to_mongo(self) -> dict[str, Any]:
        data = self.model_dump(exclude={"id"})
        for name in self.object_id_fields:
            data[name] = ObjectId(data[name])
        data["_id"] = ObjectId(self.id)
        return data


class TimestampedModel(MongoModel):
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
