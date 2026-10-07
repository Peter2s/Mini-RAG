from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.enums import DatabaseStatus, HealthStatus
from app.db import MongoDatabase
from app.dependencies import SettingsDep, VectorsDep, get_mongo
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
    mongo: Annotated[MongoDatabase, Depends(get_mongo)],
    vectors: VectorsDep,
) -> HealthResponse:
    database = await _check(mongo.ping)
    vector_store = await _check(vectors.ping)
    healthy = database == vector_store == DatabaseStatus.OK
    return HealthResponse(
        status=HealthStatus.OK if healthy else HealthStatus.DEGRADED,
        app=settings.app_name,
        version=settings.app_version,
        database=database,
        vector_store=vector_store,
    )
