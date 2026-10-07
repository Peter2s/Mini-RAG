from fastapi.testclient import TestClient

from app.core.exceptions import VectorStoreError
from app.main import create_app
from tests.pdf_factory import make_pdf

DOCS = {
    "france.txt": b"Paris is the capital of France. France is famous for wine and cheese.",
    "japan.txt": b"Tokyo is the capital of Japan. Japan is known for sushi and bullet trains.",
    "brazil.txt": b"Brasilia is the capital of Brazil. Brazil is famous for football and coffee.",
}


def _ingest(client, collection_id):
    for name, content in DOCS.items():
        response = client.post(
            f"/api/v1/collections/{collection_id}/documents",
            files={"file": (name, content, "text/plain")},
        )
        assert response.status_code == 201

    processed = client.post(f"/api/v1/collections/{collection_id}/documents/process").json()
    assert processed["documents_processed"] == 3

    indexed = client.post(f"/api/v1/collections/{collection_id}/index").json()
    assert indexed["chunks_indexed"] == indexed["total_chunks"] == 3


def test_search_ranks_relevant_chunk_first(client, collection):
    _ingest(client, collection["id"])

    response = client.post(
        f"/api/v1/collections/{collection['id']}/search",
        json={"query": "What is the capital of Japan?", "top_k": 2},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["filename"] == "japan.txt"
    assert results[0]["score"] >= results[1]["score"]


def test_index_is_incremental_unless_do_reset(client, collection):
    _ingest(client, collection["id"])
    base = f"/api/v1/collections/{collection['id']}/index"
    assert client.post(base).json()["chunks_indexed"] == 0
    assert client.post(base, json={"do_reset": True}).json()["chunks_indexed"] == 3


def test_switching_embedding_model_reindexes_without_a_reset(client, collection, settings):
    _ingest(client, collection["id"])
    base = f"/api/v1/collections/{collection['id']}"

    other_model = settings.model_copy(update={"embedding_dimension": 256})
    with TestClient(create_app(other_model)) as other:
        assert other.post(f"{base}/index").json()["chunks_indexed"] == 3
        assert other.post(f"{base}/index").json()["chunks_indexed"] == 0
        hits = other.post(f"{base}/search", json={"query": "capital of Japan", "top_k": 1})
        assert hits.json()["results"][0]["filename"] == "japan.txt"


def test_interrupted_reset_leaves_chunks_pending(client, collection, monkeypatch):
    _ingest(client, collection["id"])
    base = f"/api/v1/collections/{collection['id']}/index"

    async def unreachable(collection_id: str) -> None:
        raise VectorStoreError("vector store down")

    monkeypatch.setattr(client.app.state.vectors, "delete_collection", unreachable)
    assert client.post(base, json={"do_reset": True}).status_code == 502
    monkeypatch.undo()

    assert client.post(base).json()["chunks_indexed"] == 3


def test_answer_uses_retrieved_context(client, collection, fake_llm):
    _ingest(client, collection["id"])

    response = client.post(
        f"/api/v1/collections/{collection['id']}/answer",
        json={"question": "What is the capital of France?", "top_k": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Paris")
    assert body["model"] == "fake-model"
    assert body["sources"][0]["filename"] == "france.txt"

    [call] = fake_llm.calls
    assert "Paris is the capital of France" in call["prompt"]
    assert "Question: What is the capital of France?" in call["prompt"]


def test_answer_without_index_is_bad_request(client, collection, fake_llm):
    response = client.post(
        f"/api/v1/collections/{collection['id']}/answer", json={"question": "Anything?"}
    )
    assert response.status_code == 400
    assert fake_llm.calls == []


def test_deleting_collection_removes_chunks(client, collection):
    _ingest(client, collection["id"])
    assert client.delete(f"/api/v1/collections/{collection['id']}").status_code == 204
    response = client.post(
        f"/api/v1/collections/{collection['id']}/search", json={"query": "capital"}
    )
    assert response.status_code == 404


def test_pdf_and_markdown_end_to_end(client, collection, fake_llm):
    base = f"/api/v1/collections/{collection['id']}"
    pdf = make_pdf("Welcome to the handbook", "Employees get 21 vacation days per year")
    uploads = {
        "guide.pdf": (pdf, "application/pdf"),
        "cafe.md": (b"# Cafeteria\n\nThe cafeteria opens at 8am.", "text/markdown"),
    }
    for name, (content, mime) in uploads.items():
        response = client.post(f"{base}/documents", files={"file": (name, content, mime)})
        assert response.status_code == 201

    processed = client.post(f"{base}/documents/process").json()
    assert processed["documents_processed"] == 2, processed
    assert [result["pages_loaded"] for result in processed["results"]] == [2, 1]
    client.post(f"{base}/index")

    hits = client.post(f"{base}/search", json={"query": "vacation days", "top_k": 1}).json()
    [hit] = hits["results"]
    assert hit["filename"] == "guide.pdf"
    assert hit["text"] == "Employees get 21 vacation days per year"
    assert hit["metadata"]["page_label"] == "2"
    assert hit["metadata"]["total_pages"] == 2
    assert "source" not in hit["metadata"]

    answer = client.post(f"{base}/answer", json={"question": "vacation days?", "top_k": 1})
    assert answer.json()["sources"][0]["metadata"]["page"] == 1
    assert 'source="guide.pdf" chunk="1" page="2"' in fake_llm.calls[0]["prompt"]
