import os
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.core.config import Settings
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
        with MongoClient(url, serverSelectionTimeoutMS=2000) as client:
            client.admin.command("ping")
    except PyMongoError as error:
        pytest.skip(
            f"MongoDB is not reachable ({error.__class__.__name__}). Run: docker compose up -d"
        )
    return url


@pytest.fixture
def mongo_db(mongo_url: str) -> Iterator[Database]:
    """A throwaway database per test, dropped afterwards. Used for direct assertions."""
    name = f"mini_rag_test_{uuid.uuid4().hex[:12]}"
    with MongoClient(mongo_url, tz_aware=True) as client:
        yield client[name]
        client.drop_database(name)


@pytest.fixture
def settings(tmp_path, mongo_url: str, mongo_db: Database) -> Settings:
    return Settings(
        _env_file=None,
        mongodb_url=mongo_url,
        mongodb_database=mongo_db.name,
        upload_dir=tmp_path / "uploads",
        max_upload_size_mb=1,
        embedding_provider="hash",
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
def collection(client: TestClient) -> dict:
    response = client.post("/api/v1/collections", json={"name": "demo", "description": "test"})
    assert response.status_code == 201
    return response.json()
