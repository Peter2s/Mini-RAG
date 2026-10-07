import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9 _.-]+$")
    description: str | None = Field(default=None, max_length=2000)


class CollectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
