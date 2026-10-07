import hashlib

import pytest

from app.core.enums import FileExtension
from app.repositories import DocumentRepository


def _upload(client, collection_id, name="notes.txt", content=b"hello world", mime="text/plain"):
    return client.post(
        f"/api/v1/collections/{collection_id}/documents", files={"file": (name, content, mime)}
    )


def test_upload_list_get_delete_document(client, collection, settings):
    response = _upload(client, collection["id"])
    assert response.status_code == 201
    document = response.json()
    assert document["status"] == "uploaded"
    assert document["size_bytes"] == len(b"hello world")

    base = f"/api/v1/collections/{collection['id']}/documents"
    assert client.get(base).json()["total"] == 1
    assert client.get(f"{base}/{document['id']}").json()["filename"] == "notes.txt"

    stored = list((settings.upload_dir / str(collection["id"])).iterdir())
    assert len(stored) == 1

    assert client.delete(f"{base}/{document['id']}").status_code == 204
    assert client.get(f"{base}/{document['id']}").status_code == 404
    assert not stored[0].exists()


def test_rejects_disallowed_extension(client, collection):
    response = _upload(client, collection["id"], name="evil.exe")
    assert response.status_code == 400


def test_rejects_empty_and_oversized_files(client, collection):
    assert _upload(client, collection["id"], content=b"").status_code == 400
    too_big = b"x" * (1024 * 1024 + 1)
    assert _upload(client, collection["id"], content=too_big).status_code == 413


@pytest.mark.parametrize("collection_id", ["999", "not-an-id", "65f0c0ffee0000000000abcd"])
def test_upload_to_missing_collection_404(client, collection_id):
    assert _upload(client, collection_id).status_code == 404


def test_missing_document_404(client, collection):
    base = f"/api/v1/collections/{collection['id']}/documents"
    assert client.get(f"{base}/65f0c0ffee0000000000abcd").status_code == 404
    assert client.get(f"{base}/bad-id").status_code == 404


def test_process_creates_chunks(client, collection):
    text = ("Sentence number {}. " * 200).format(*range(200)).encode()
    document = _upload(client, collection["id"], content=text).json()

    response = client.post(
        f"/api/v1/collections/{collection['id']}/documents/{document['id']}/process",
        json={"chunk_size": 500, "chunk_overlap": 50},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processed"
    assert body["chunks_created"] > 1


def test_process_rejects_overlap_not_smaller_than_size(client, collection):
    document = _upload(client, collection["id"]).json()
    response = client.post(
        f"/api/v1/collections/{collection['id']}/documents/{document['id']}/process",
        json={"chunk_size": 200, "chunk_overlap": 200},
    )
    assert response.status_code == 422


def test_extension_check_is_case_insensitive(client, collection):
    response = _upload(client, collection["id"], name="README.MD", content=b"# hi")
    assert response.status_code == 201


def test_respects_allowed_extensions_setting(client, collection, settings):
    settings.allowed_extensions = [FileExtension.TXT]
    response = _upload(client, collection["id"], name="notes.md", content=b"# hi")
    assert response.status_code == 400
    assert "Allowed: .txt" in response.json()["error"]["message"]


def test_same_content_under_another_name_is_rejected(client, collection):
    first = _upload(client, collection["id"], name="report.txt", content=b"quarterly numbers")
    assert first.status_code == 201

    duplicate = _upload(
        client, collection["id"], name="copy-of-report.md", content=b"quarterly numbers"
    )
    assert duplicate.status_code == 409
    message = duplicate.json()["error"]["message"]
    assert "'report.txt'" in message
    assert first.json()["id"] in message


def test_same_name_with_different_content_is_accepted(client, collection):
    assert (
        _upload(client, collection["id"], name="notes.txt", content=b"version one").status_code
        == 201
    )
    assert (
        _upload(client, collection["id"], name="notes.txt", content=b"version two").status_code
        == 201
    )
    assert client.get(f"/api/v1/collections/{collection['id']}/documents").json()["total"] == 2


def test_same_content_is_allowed_in_another_collection(client, collection):
    other = client.post("/api/v1/collections", json={"name": "other"}).json()
    assert _upload(client, collection["id"], content=b"shared file").status_code == 201
    assert _upload(client, other["id"], content=b"shared file").status_code == 201


def test_file_is_stored_under_its_content_hash(client, collection, settings):
    content = b"# Title\nbody"
    document = _upload(client, collection["id"], name="My Notes (final).MD", content=content).json()

    expected_hash = hashlib.sha256(content).hexdigest()
    assert document["content_hash"] == expected_hash
    assert document["filename"] == "My Notes (final).MD"
    stored = [path.name for path in (settings.upload_dir / collection["id"]).iterdir()]
    assert stored == [f"{expected_hash}.md"]


def test_content_can_be_uploaded_again_after_deleting_it(client, collection):
    first = _upload(client, collection["id"], content=b"temporary").json()
    base = f"/api/v1/collections/{collection['id']}/documents"
    assert client.delete(f"{base}/{first['id']}").status_code == 204
    assert _upload(client, collection["id"], content=b"temporary").status_code == 201


def test_concurrent_duplicate_is_rejected_by_index_and_keeps_the_file(
    client, collection, settings, monkeypatch
):
    first = _upload(client, collection["id"], content=b"raced content").json()

    async def not_found(self, collection_id, content_hash):
        return None

    monkeypatch.setattr(DocumentRepository, "get_by_content_hash", not_found)
    duplicate = _upload(client, collection["id"], content=b"raced content")

    assert duplicate.status_code == 409
    stored = settings.upload_dir / collection["id"] / f"{first['content_hash']}.txt"
    assert stored.read_bytes() == b"raced content"
