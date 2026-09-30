import shutil
from pathlib import Path

from starlette.concurrency import run_in_threadpool


class FileRepository:
    """Stores uploaded files on local disk under `<upload_dir>/<collection_id>/<stored_name>`.

    Callers name files by their content hash, so identical content maps to the same file.
    """

    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir

    def collection_dir(self, collection_id: str) -> Path:
        return self.upload_dir / str(collection_id)

    async def save(self, collection_id: str, stored_name: str, data: bytes) -> Path:
        if not stored_name or Path(stored_name).name != stored_name:
            raise ValueError(f"stored_name must be a plain file name, got {stored_name!r}")
        target = self.collection_dir(collection_id) / stored_name

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await run_in_threadpool(_write)
        return target

    async def delete(self, path: str | Path) -> None:
        await run_in_threadpool(Path(path).unlink, missing_ok=True)

    async def delete_collection(self, collection_id: str) -> None:
        await run_in_threadpool(
            shutil.rmtree, self.collection_dir(collection_id), ignore_errors=True
        )
