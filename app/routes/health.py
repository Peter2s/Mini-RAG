from typing import Annotated

from fastapi import APIRouter, Depends

from app.db import MongoDatabase
from app.dependencies import SettingsDep, get_mongo
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep, mongo: Annotated[MongoDatabase, Depends(get_mongo)]):
    try:
        await mongo.ping()
        database = "ok"
    except Exception:
        database = "unavailable"
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        database=database,
    )
