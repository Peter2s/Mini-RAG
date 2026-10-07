import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from app.core.config import Settings
from app.core.enums import EmbeddingProviderName, VectorStoreName
from app.core.exceptions import EmbeddingError
from app.providers.embeddings import build_embedding_provider
from app.providers.embeddings.fastembed_provider import FastEmbedProvider

GEMMA = "google/embeddinggemma-300m"
DESCRIPTION = {
    "model": GEMMA,
    "sources": {"hf": "onnx-community/embeddinggemma-300m-ONNX"},
    "model_file": "onnx/model.onnx",
    "additional_files": ["onnx/model.onnx_data"],
    "dim": 768,
}


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding: records inputs, returns fixed vectors."""

    instances: ClassVar[list["FakeTextEmbedding"]] = []

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.seen: list[str] = []
        FakeTextEmbedding.instances.append(self)

    @staticmethod
    def list_supported_models() -> list[dict[str, Any]]:
        return [DESCRIPTION]

    def embed(self, texts: list[str]):
        self.seen.extend(texts)
        return (np.array([1.0, 2.0, 3.0], dtype=np.float32) for _ in texts)


@pytest.fixture
def fake_fastembed(monkeypatch):
    """Swap in fake `fastembed` and `huggingface_hub` modules.

    The provider imports them lazily, so the real ones (and ONNX Runtime) never load in tests.
    """
    FakeTextEmbedding.instances = []
    downloads: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: downloads.append(kwargs)),
    )
    return downloads


def test_adds_query_and_document_prefixes(fake_fastembed, tmp_path):
    provider = FastEmbedProvider(GEMMA, tmp_path)

    vector = asyncio.run(provider.embed_query("ما هي الإجازة؟"))
    vectors = asyncio.run(provider.embed_documents(["نص المستند"]))

    assert vector == [1.0, 2.0, 3.0]
    assert vectors == [[1.0, 2.0, 3.0]]
    [model] = FakeTextEmbedding.instances
    assert model.seen == [
        "task: search result | query: ما هي الإجازة؟",
        "title: none | text: نص المستند",
    ]


def test_unknown_prefix_model_gets_text_unchanged(fake_fastembed, tmp_path, monkeypatch):
    name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    monkeypatch.setitem(DESCRIPTION, "model", name)
    provider = FastEmbedProvider(name, tmp_path)
    asyncio.run(provider.embed_query("سؤال"))
    assert FakeTextEmbedding.instances[0].seen == ["سؤال"]


def test_model_is_downloaded_to_a_plain_folder_and_loaded_once(fake_fastembed, tmp_path):
    provider = FastEmbedProvider(GEMMA, tmp_path, threads=2)

    async def many_concurrent_requests() -> None:
        await asyncio.gather(provider.warmup(), *(provider.embed_query("x") for _ in range(5)))

    asyncio.run(many_concurrent_requests())

    assert len(FakeTextEmbedding.instances) == 1
    [download] = fake_fastembed
    local_dir = tmp_path / "google--embeddinggemma-300m"
    assert download["local_dir"] == local_dir
    assert {"onnx/model.onnx", "onnx/model.onnx_data"} <= set(download["allow_patterns"])
    assert FakeTextEmbedding.instances[0].kwargs == {
        "specific_model_path": str(local_dir),
        "threads": 2,
    }


def test_skips_download_when_weights_are_already_on_disk(fake_fastembed, tmp_path):
    local_dir = tmp_path / "google--embeddinggemma-300m" / "onnx"
    local_dir.mkdir(parents=True)
    (local_dir / "model.onnx").write_bytes(b"weights")
    (local_dir / "model.onnx_data").write_bytes(b"weights")

    asyncio.run(FastEmbedProvider(GEMMA, tmp_path).warmup())

    assert fake_fastembed == []
    assert len(FakeTextEmbedding.instances) == 1


def test_unsupported_model_raises_embedding_error(fake_fastembed, tmp_path):
    provider = FastEmbedProvider("someone/not-a-model", tmp_path)
    with pytest.raises(EmbeddingError, match="doesn't support 'someone/not-a-model'"):
        _ = provider.dimension
    with pytest.raises(EmbeddingError, match="doesn't support 'someone/not-a-model'"):
        asyncio.run(provider.embed_query("x"))


def test_identity_and_dimension_are_known_without_loading_the_model(fake_fastembed, tmp_path):
    provider = FastEmbedProvider(GEMMA, tmp_path)
    assert provider.model_id == "fastembed:google/embeddinggemma-300m"
    assert provider.dimension == 768
    assert FakeTextEmbedding.instances == []
    assert fake_fastembed == []


def test_factory_builds_fastembed_provider(tmp_path):
    settings = Settings(
        _env_file=None,
        embedding_provider=EmbeddingProviderName.FASTEMBED,
        fastembed_model=GEMMA,
        models_dir=tmp_path,
    )
    assert isinstance(build_embedding_provider(settings), FastEmbedProvider)


@pytest.mark.skipif(
    not os.environ.get("RUN_MODEL_TESTS"),
    reason="downloads a real model; set RUN_MODEL_TESTS=1 to run",
)
def test_arabic_documents_end_to_end_with_a_real_model(tmp_path, mongo_url, mongo_db, fake_llm):
    from fastapi.testclient import TestClient

    from app.dependencies import get_llm_provider
    from app.main import create_app

    settings = Settings(
        _env_file=None,
        mongodb_url=mongo_url,
        mongodb_database=mongo_db.name,
        upload_dir=tmp_path / "uploads",
        embedding_provider=EmbeddingProviderName.FASTEMBED,
        vector_store=VectorStoreName.MONGODB,
        fastembed_model=os.environ.get("FASTEMBED_TEST_MODEL", GEMMA),
        models_dir=Path(os.environ.get("MODELS_DIR", tmp_path / "models")),
    )
    app = create_app(settings)
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    documents = {
        "leave.txt": "يحصل الموظفون على واحد وعشرين يوماً من الإجازة السنوية المدفوعة الأجر.",
        "salary.txt": "يتم صرف الرواتب في اليوم الأخير من كل شهر ميلادي.",
        "insurance.txt": "تغطي التأمينات الصحية الموظف وزوجته وأطفاله.",
    }
    with TestClient(app) as client:
        collection = client.post("/api/v1/collections", json={"name": "arabic"}).json()
        base = f"/api/v1/collections/{collection['id']}"
        for name, text in documents.items():
            upload = client.post(
                f"{base}/documents", files={"file": (name, text.encode(), "text/plain")}
            )
            assert upload.status_code == 201
        assert client.post(f"{base}/documents/process").json()["documents_failed"] == 0
        assert client.post(f"{base}/index").json()["chunks_indexed"] == 3

        def top_file(query: str) -> str:
            results = client.post(f"{base}/search", json={"query": query, "top_k": 1}).json()
            filename: str = results["results"][0]["filename"]
            return filename

        assert top_file("في أي يوم أستلم مرتبي؟") == "salary.txt"
        assert top_file("هل يشمل التأمين الطبي عائلتي؟") == "insurance.txt"
        assert top_file("How many vacation days do I get?") == "leave.txt"
