from app.core.enums import FileExtension


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


def test_upload_to_missing_collection_404(client):
    assert _upload(client, 999).status_code == 404


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
