"""Alembic environment: runs migrations against DATABASE_URL with the async engine."""

import asyncio
import logging.config

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

import app.models  # noqa: F401
from app.core.config import get_settings
from app.db.base import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    logging.config.fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    override = config.get_main_option("sqlalchemy.url")
    return override or get_settings().database_url.get_secret_value()


def include_name(name: str | None, type_: str, parent_names: object) -> bool:
    """Only compare tables the models define, not pgvector's per-model tables."""
    if type_ == "table":
        return name in target_metadata.tables
    return True


def _configure(connection: Connection | None = None) -> None:
    context.configure(
        connection=connection,
        url=None if connection else _database_url(),
        target_metadata=target_metadata,
        include_name=include_name,
        compare_type=True,
        literal_binds=connection is None,
        dialect_opts={"paramstyle": "named"} if connection is None else {},
    )


def run_migrations_offline() -> None:
    _configure()
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = async_engine_from_config(
        {"sqlalchemy.url": _database_url()}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
