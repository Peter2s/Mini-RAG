import os
import uuid
from collections.abc import Iterator
from contextlib import closing
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.core.enums import VectorStoreName
from app.db.types import MongoDocument
from app.dependencies import get_llm_provider
from app.main import create_app
from app.schemas.llm import LLMResponse


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


@pytest.fixture(scope="session")
def mongo_url() -> str:
    """MongoDB for tests: MONGODB_TEST_URL, else MONGODB_URL from the environment or `.env`."""
    url = os.environ.get("MONGODB_TEST_URL") or Settings().mongodb_url.get_secret_value()
    try:
        client: MongoClient[MongoDocument]
        with MongoClient(url, serverSelectionTimeoutMS=2000) as client:
            client.admin.command("ping")
    except PyMongoError as error:
        pytest.skip(
            f"MongoDB is not reachable ({error.__class__.__name__}). Run: docker compose up -d"
        )
    return url


@pytest.fixture
def mongo_db(mongo_url: str) -> Iterator[Database[MongoDocument]]:
    """A throwaway database per test, dropped afterwards. Used for direct assertions."""
    name = f"mini_rag_test_{uuid.uuid4().hex[:12]}"
    client: MongoClient[MongoDocument]
    with MongoClient(mongo_url, tz_aware=True) as client:
        yield client[name]
        client.drop_database(name)


@pytest.fixture
def settings(tmp_path, mongo_url: str, mongo_db: Database[MongoDocument]) -> Settings:
    return Settings(
        _env_file=None,
        mongodb_url=mongo_url,
        mongodb_database=mongo_db.name,
        upload_dir=tmp_path / "uploads",
        max_upload_size_mb=1,
        embedding_provider="hash",
        vector_store=VectorStoreName.MONGODB,
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
    """The API backed by MongoDB for documents and Qdrant for vectors."""
    app = create_app(qdrant_settings)
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    with TestClient(app) as test_client:
        yield test_client
