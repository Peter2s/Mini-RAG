from fastapi import APIRouter

from app.routes import collections, documents, health, rag

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(collections.router)
api_router.include_router(documents.router)
api_router.include_router(rag.router)

__all__ = ["api_router"]
