import uuid

EntityId = str | uuid.UUID


def parse_id(value: EntityId) -> uuid.UUID | None:
    """Parse an id from a URL or another table. Anything else returns None: not found."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(value)
    except (TypeError, ValueError):
        return None
