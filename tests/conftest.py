from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
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
