from collections.abc import Sequence

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
from app.db.base import Entity
from app.models import EntityId, parse_id

_UNIQUE_VIOLATION = "23505"


def is_unique_violation(error: IntegrityError) -> bool:
    return getattr(error.orig, "sqlstate", None) == _UNIQUE_VIOLATION


class SqlRepository[ModelT: Entity]:
    """Generic CRUD over one table. Invalid ids behave like missing rows.

    Repositories flush but never commit: the request's session commits, or a service commits
    where progress has to survive a later failure.
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, entity_id: EntityId) -> ModelT | None:
        parsed = parse_id(entity_id)
        return None if parsed is None else await self.session.get(self.model, parsed)

    async def add(self, entity: ModelT) -> ModelT:
        """Insert in a savepoint, so a duplicate leaves the session usable."""
        try:
            async with self.session.begin_nested():
                self.session.add(entity)
        except IntegrityError as error:
            if is_unique_violation(error):
                raise ConflictError(self.conflict_message(entity)) from error
            raise
        return entity

    async def save(self, entity: ModelT) -> ModelT:
        await self.session.flush()
        return entity

    async def delete(self, entity: ModelT) -> None:
        await self.session.delete(entity)
        await self.session.flush()

    async def count(self, *where: ColumnElement[bool]) -> int:
        statement = select(func.count()).select_from(self.model).where(*where)
        return (await self.session.execute(statement)).scalar_one()

    def conflict_message(self, entity: ModelT) -> str:
        return f"This {self.model.__tablename__.removesuffix('s')} already exists"

    async def _find_one(self, *where: ColumnElement[bool]) -> ModelT | None:
        return (await self.session.scalars(select(self.model).where(*where))).first()

    async def _find(
        self, *where: ColumnElement[bool], offset: int = 0, limit: int = 0
    ) -> Sequence[ModelT]:
        statement = select(self.model).where(*where).order_by(self.model.id).offset(offset)
        if limit:
            statement = statement.limit(limit)
        return (await self.session.scalars(statement)).all()
