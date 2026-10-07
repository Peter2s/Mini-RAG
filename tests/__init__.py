"""Test package. Runs before conftest.py.

qdrant-client imports FastEmbed whenever it's installed, and FastEmbed loads ONNX Runtime,
whose native threads can abort the interpreter at exit on macOS ("recursive_mutex lock
failed"). No test needs a real model except the opt-in one, so FastEmbed is made unimportable
here; qdrant-client then skips it, and that test removes FASTEMBED_BLOCKER.
"""

import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType


class _BlockFastEmbed(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname == "fastembed":
            raise ModuleNotFoundError("FastEmbed is blocked in tests", name=fullname)
        return None


FASTEMBED_BLOCKER = _BlockFastEmbed()
sys.meta_path.insert(0, FASTEMBED_BLOCKER)
