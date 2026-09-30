from enum import StrEnum
from pathlib import Path


class FileExtension(StrEnum):
    """File types the app can extract text from."""

    TXT = ".txt"
    MD = ".md"
    PDF = ".pdf"

    @classmethod
    def from_path(cls, path: str | Path) -> "FileExtension | None":
        """Return the extension of `path` (case-insensitive), or None if it isn't supported."""
        try:
            return cls(Path(path).suffix.lower())
        except ValueError:
            return None


class MongoCollection(StrEnum):
    """Names of the MongoDB collections the app stores data in."""

    COLLECTIONS = "collections"
    DOCUMENTS = "documents"
    CHUNKS = "chunks"
