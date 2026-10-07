"""Text extraction built on LangChain document loaders.

All LangChain usage lives here, so switching loaders never touches the services.
"""

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader

from app.core.enums import FileExtension
from app.schemas.chunk import LoadedPage

_HIDDEN_METADATA = frozenset({"source", "file_path"})


class UnsupportedFileTypeError(ValueError):
    pass


class DocumentLoadError(ValueError):
    pass


def get_loader(path: Path) -> BaseLoader:
    """Return the LangChain loader for a file, chosen by its extension."""
    match FileExtension.from_path(path):
        case FileExtension.TXT | FileExtension.MD:
            return TextLoader(path, encoding="utf-8")
        case FileExtension.PDF:
            return PyPDFLoader(path, mode="page")
        case None:
            raise UnsupportedFileTypeError(f"Unsupported file type: {path.suffix or 'none'}")


def load_pages(path: Path) -> list[LoadedPage]:
    """Load a file into pages of content and metadata. Blocking: call it from a thread pool."""
    loader = get_loader(path)
    try:
        documents = loader.load()
    except Exception as error:
        raise DocumentLoadError(
            f"Could not extract text from this {path.suffix.lower()} file"
        ) from error
    return [
        LoadedPage(content=document.page_content, metadata=_clean_metadata(document.metadata))
        for document in documents
    ]


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop private keys and make values JSON-safe so they can be stored and returned."""
    return {
        key: value if isinstance(value, str | int | float | bool) or value is None else str(value)
        for key, value in metadata.items()
        if key not in _HIDDEN_METADATA
    }
