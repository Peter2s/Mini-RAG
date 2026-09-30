import re

# Preferred break points, strongest first.
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\f\v]+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters with `chunk_overlap` overlap.

    Chunks end on the strongest natural boundary (paragraph, line, sentence, word) found in
    the second half of the window, so they rarely cut words or sentences in half.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    text = normalize_text(text)
    chunks: list[str] = []
    start, length = 0, len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _find_break(text, start, end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = _snap_to_word_start(text, next_start, end)

    return chunks


def _find_break(text: str, start: int, end: int) -> int:
    min_end = start + (end - start) // 2
    for separator in _SEPARATORS:
        index = text.rfind(separator, min_end, end)
        if index != -1:
            return index + len(separator)
    return end


def _snap_to_word_start(text: str, pos: int, limit: int) -> int:
    """Move `pos` forward to the start of the next word so overlaps don't begin mid-word."""
    if pos == 0 or text[pos - 1].isspace():
        return pos
    for i in range(pos, limit):
        if text[i].isspace():
            return i + 1
    return pos
