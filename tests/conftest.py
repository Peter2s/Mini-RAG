import os
import uuid
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import QdrantClient
from sqlalchemy import Connection, Engine, MetaData, Table, create_engine, make_url, text
from sqlalchemy.exc import OperationalError

from app.core.config import Settings
from app.core.enums import VectorStoreName
from app.dependencies import get_llm_provider
from app.main import create_app
from app.schemas.llm import LLMResponse

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("collections", "documents", "chunks")


class FakeLLM:
    """Records prompts and returns a canned answer, so tests never call the real API."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def generate(self, *, system: str, prompt: str) -> LLMResponse:
        self.calls.append({"system": system, "prompt": prompt})
        return LLMResponse(
            text="Paris is the capital of France [1].", model="fake-model", stop_reason="end_turn"
        )

    async def aclose(self) -> None:
        return None


def alembic_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    config.attributes["configure_logger"] = False
    return config


def sync_url(database_url: str) -> str:
    """The same database through psycopg, for plain synchronous test code."""
    url = make_url(database_url).set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def postgres_admin() -> Iterator[tuple[Engine, str]]:
    """The test server: DATABASE_TEST_URL, else DATABASE_URL from the environment or `.env`."""
    url = os.environ.get("DATABASE_TEST_URL") or Settings().database_url.get_secret_value()
    engine = create_engine(
        sync_url(url), isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 2}
    )
    try:
        with engine.connect():
            pass
    except OperationalError as error:
        engine.dispose()
        pytest.skip(
            f"PostgreSQL is not reachable ({error.__class__.__name__}). Run: docker compose up -d"
        )
    yield engine, url
    engine.dispose()


@pytest.fixture(scope="session")
def database_url(postgres_admin: tuple[Engine, str]) -> Iterator[str]:
    """A throwaway database for the test session, migrated to the latest revision."""
    engine, server_url = postgres_admin
    name = f"mini_rag_test_{uuid.uuid4().hex[:12]}"
    with engine.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(server_url).set(database=name).render_as_string(hide_password=False)
    command.upgrade(alembic_config(url), "head")
    yield url
    with engine.connect() as connection:
        connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))


@pytest.fixture(scope="session")
def sync_engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(sync_url(database_url), isolation_level="AUTOCOMMIT")
    yield engine
    engine.dispose()


@pytest.fixture
def db(sync_engine: Engine) -> Iterator[Connection]:
    """Empty tables for each test, plus a connection for direct assertions."""
    with sync_engine.connect() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
        vector_tables: list[str] = list(
            connection.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename LIKE 'chunk\\_vectors\\_\\_%'"
                )
            ).scalars()
        )
        for name in vector_tables:
            Table(name, MetaData()).drop(connection)
        yield connection


@pytest.fixture
def settings(tmp_path, database_url: str, db: Connection) -> Settings:
    return Settings(
        _env_file=None,
        database_url=database_url,
        upload_dir=tmp_path / "uploads",
        max_upload_size_mb=1,
        embedding_provider="hash",
        vector_store=VectorStoreName.PGVECTOR,
        anthropic_api_key="test-key",
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def collection(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/v1/collections", json={"name": "demo", "description": "test"})
    assert response.status_code == 201
    created: dict[str, Any] = response.json()
    return created


@pytest.fixture(scope="session")
def qdrant_connection() -> tuple[str, str | None]:
    """Qdrant for tests: QDRANT_TEST_URL / QDRANT_TEST_API_KEY, else the app settings."""
    settings = Settings()
    url = os.environ.get("QDRANT_TEST_URL") or settings.qdrant_url
    api_key = os.environ.get("QDRANT_TEST_API_KEY") or (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    )
    try:
        with closing(QdrantClient(url=url, api_key=api_key, timeout=3)) as client:
            client.get_collections()
    except Exception as error:
        pytest.skip(
            f"Qdrant is not reachable ({error.__class__.__name__}). Run: docker compose up -d"
        )
    return url, api_key


@pytest.fixture
def qdrant_collection(qdrant_connection: tuple[str, str | None]) -> Iterator[str]:
    """A throwaway Qdrant collection prefix per test. Its collections are deleted afterwards."""
    prefix = f"mini_rag_test_{uuid.uuid4().hex[:12]}"
    yield prefix
    url, api_key = qdrant_connection
    with closing(QdrantClient(url=url, api_key=api_key)) as client:
        for existing in client.get_collections().collections:
            if existing.name.startswith(f"{prefix}__"):
                client.delete_collection(existing.name)


@pytest.fixture
def qdrant_settings(
    settings: Settings, qdrant_connection: tuple[str, str | None], qdrant_collection: str
) -> Settings:
    url, api_key = qdrant_connection
    return settings.model_copy(
        update={
            "vector_store": VectorStoreName.QDRANT,
            "qdrant_url": url,
            "qdrant_api_key": SecretStr(api_key) if api_key else None,
            "qdrant_collection": qdrant_collection,
        }
    )


@pytest.fixture
def qdrant_client_app(qdrant_settings: Settings, fake_llm: FakeLLM) -> Iterator[TestClient]:
    """The API backed by PostgreSQL for documents and Qdrant for vectors."""
    app = create_app(qdrant_settings)
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    with TestClient(app) as test_client:
        yield test_client
