from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    database: str


class Page[T](BaseModel):
    items: list[T]
    total: int
    offset: int
    limit: int
