from pathlib import Path

from sqlalchemy import event, make_url, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base


class Database:
    """Owns the async engine and session factory for the app's lifetime."""

    def __init__(self, url: str, echo: bool = False) -> None:
        parsed = make_url(url)
        self._is_sqlite = parsed.get_backend_name() == "sqlite"
        if self._is_sqlite and parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_async_engine(url, echo=echo)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

        if self._is_sqlite:
            # SQLite ignores ON DELETE CASCADE unless foreign keys are enabled per connection.
            @event.listens_for(self.engine.sync_engine, "connect")
            def _enable_foreign_keys(dbapi_connection, _record) -> None:  # noqa: ANN001
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    async def create_all(self) -> None:
        # Importing models registers them on Base.metadata.
        import app.models  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def ping(self) -> bool:
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True

    async def dispose(self) -> None:
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        return self.sessionmaker()
