def test_create_get_list_delete_collection(client, collection):
    collection_id = collection["id"]

    assert client.get(f"/api/v1/collections/{collection_id}").json()["name"] == "demo"

    listing = client.get("/api/v1/collections").json()
    assert listing["total"] == 1
    assert listing["items"][0]["id"] == collection_id

    assert client.delete(f"/api/v1/collections/{collection_id}").status_code == 204
    missing = client.get(f"/api/v1/collections/{collection_id}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_duplicate_collection_name_conflicts(client, collection):
    response = client.post("/api/v1/collections", json={"name": "demo"})
    assert response.status_code == 409


def test_invalid_collection_name_rejected(client):
    response = client.post("/api/v1/collections", json={"name": "../etc"})
    assert response.status_code == 422
