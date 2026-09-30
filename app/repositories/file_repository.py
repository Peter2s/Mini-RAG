import re
import shutil
import uuid
from pathlib import Path

from starlette.concurrency import run_in_threadpool

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class FileRepository:
    """Stores uploaded files on local disk under `<upload_dir>/<collection_id>/`."""

    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir

    def collection_dir(self, collection_id: int) -> Path:
        return self.upload_dir / str(collection_id)

    async def save(self, collection_id: int, filename: str, data: bytes) -> Path:
        safe_name = _UNSAFE_CHARS.sub("_", Path(filename).name).strip("._") or "file"
        target = self.collection_dir(collection_id) / f"{uuid.uuid4().hex}_{safe_name}"

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await run_in_threadpool(_write)
        return target

    async def delete(self, path: str | Path) -> None:
        await run_in_threadpool(Path(path).unlink, missing_ok=True)

    async def delete_collection(self, collection_id: int) -> None:
        await run_in_threadpool(
            shutil.rmtree, self.collection_dir(collection_id), ignore_errors=True
        )
