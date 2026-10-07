from collections.abc import Awaitable, Callable

from fastapi import APIRouter

from app.core.enums import DatabaseStatus, HealthStatus
from app.dependencies import DatabaseDep, SettingsDep, VectorsDep
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


async def _check(ping: Callable[[], Awaitable[object]]) -> DatabaseStatus:
    try:
        await ping()
    except Exception:
        return DatabaseStatus.UNAVAILABLE
    return DatabaseStatus.OK


@router.get("/health")
async def health(
    settings: SettingsDep,
    database: DatabaseDep,
    vectors: VectorsDep,
) -> HealthResponse:
    database_status = await _check(database.ping)
    vector_store = await _check(vectors.ping)
    healthy = database_status == vector_store == DatabaseStatus.OK
    return HealthResponse(
        status=HealthStatus.OK if healthy else HealthStatus.DEGRADED,
        app=settings.app_name,
        version=settings.app_version,
        database=database_status,
        vector_store=vector_store,
    )
