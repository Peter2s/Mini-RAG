from fastapi import APIRouter, Query, status

from app.dependencies import CollectionServiceDep
from app.schemas.collection import CollectionCreate, CollectionRead
from app.schemas.common import Page

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionRead, status_code=status.HTTP_201_CREATED)
async def create_collection(data: CollectionCreate, service: CollectionServiceDep):
    return await service.create(data)


@router.get("", response_model=Page[CollectionRead])
async def list_collections(
    service: CollectionServiceDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    return await service.list(offset=offset, limit=limit)


@router.get("/{collection_id}", response_model=CollectionRead)
async def get_collection(collection_id: str, service: CollectionServiceDep):
    return await service.get(collection_id)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_id: str, service: CollectionServiceDep) -> None:
    await service.delete(collection_id)
