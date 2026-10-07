from sqlalchemy import CheckConstraint, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Entity, TimestampMixin


class Collection(TimestampMixin, Entity):
    __tablename__ = "collections"
    __table_args__ = (CheckConstraint("char_length(name) >= 1", name="name_not_empty"),)

    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
