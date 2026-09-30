import hashlib


def sha256_hex(data: bytes) -> str:
    """Content fingerprint: identical bytes always give the same 64-character hex string."""
    return hashlib.sha256(data).hexdigest()
