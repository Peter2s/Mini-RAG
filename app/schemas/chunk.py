from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LoadedPage:
    """One unit of content from a loader: a PDF page, or a whole text/markdown file."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of a page, ready to be stored, with a copy of its page's metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
