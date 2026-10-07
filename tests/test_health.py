from app.core.enums import DatabaseStatus, HealthStatus
from app.dependencies import get_mongo, get_vector_repository


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == HealthStatus.OK
    assert body["database"] == DatabaseStatus.OK
    assert body["vector_store"] == DatabaseStatus.OK


def test_health_reports_degraded_when_database_is_unreachable(client):
    class UnreachableMongo:
        async def ping(self) -> bool:
            raise ConnectionError("down")

    client.app.dependency_overrides[get_mongo] = lambda: UnreachableMongo()
    body = client.get("/api/v1/health").json()
    assert body["status"] == HealthStatus.DEGRADED
    assert body["database"] == DatabaseStatus.UNAVAILABLE


def test_health_reports_degraded_when_vector_store_is_unreachable(client):
    class UnreachableVectorStore:
        async def ping(self) -> None:
            raise ConnectionError("down")

    client.app.dependency_overrides[get_vector_repository] = lambda: UnreachableVectorStore()
    body = client.get("/api/v1/health").json()
    assert body["status"] == HealthStatus.DEGRADED
    assert body["database"] == DatabaseStatus.OK
    assert body["vector_store"] == DatabaseStatus.UNAVAILABLE
