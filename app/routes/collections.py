from typing import Annotated

from fastapi import APIRouter, Query, status

from app.dependencies import CollectionServiceDep
from app.models import Collection
from app.schemas.collection import CollectionCreate, CollectionRead
from app.schemas.common import Page

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionRead, status_code=status.HTTP_201_CREATED)
async def create_collection(data: CollectionCreate, service: CollectionServiceDep) -> Collection:
    return await service.create(data)


@router.get("")
async def list_collections(
    service: CollectionServiceDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[CollectionRead]:
    return await service.list(offset=offset, limit=limit)


@router.get("/{collection_id}", response_model=CollectionRead)
async def get_collection(collection_id: str, service: CollectionServiceDep) -> Collection:
    return await service.get(collection_id)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_id: str, service: CollectionServiceDep) -> None:
    await service.delete(collection_id)
