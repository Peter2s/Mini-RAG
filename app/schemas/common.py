from pydantic import BaseModel

from app.core.enums import DatabaseStatus, HealthStatus


class HealthResponse(BaseModel):
    status: HealthStatus
    app: str
    version: str
    database: DatabaseStatus
    vector_store: DatabaseStatus


class Page[T](BaseModel):
    items: list[T]
    total: int
    offset: int
    limit: int
