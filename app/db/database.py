from pathlib import Path

from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


class Database:
    """Owns the SQLAlchemy engine and session factory for the app's lifetime."""

    def __init__(self, url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self.sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def init(self) -> None:
        """Refuse to start on a database whose schema is behind the migrations."""
        expected = ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()
        async with self.engine.connect() as connection:
            current = await connection.run_sync(_current_revision)
        if current != expected:
            raise RuntimeError(
                f"Database schema is at revision {current}, the app needs {expected}. "
                "Run: uv run alembic upgrade head"
            )

    async def ping(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def close(self) -> None:
        await self.engine.dispose()


def _current_revision(connection: Connection) -> str | None:
    return MigrationContext.configure(connection).get_current_revision()
