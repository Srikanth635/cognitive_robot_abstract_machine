"""
Run the ordinary CLI while rejecting any optional SDK import.
"""

from __future__ import annotations

import importlib.abc
import runpy
import sys
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec
    from types import ModuleType

# %% optional dependency absence


@dataclass
class AbsentTransport(importlib.abc.MetaPathFinder):
    """
    Emulate an installation without the optional MCP package.
    """

    def find_spec(
        self, fullname: str, path: object = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname == "mcp" or fullname.startswith("mcp."):
            raise ModuleNotFoundError(fullname)
        return None


if __name__ == "__main__":
    sys.meta_path.insert(0, AbsentTransport())
    runpy.run_module("repository_knowledge", run_name="__main__")
