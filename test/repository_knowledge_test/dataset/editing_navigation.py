"""
Edit source after navigation to exercise the final freshness check.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from repository_knowledge.model import QueryPage
from repository_knowledge.operations.queries import Navigation

# %% query interleaving


@dataclass
class EditingNavigation(Navigation):
    """
    Introduce an edit immediately after source evidence is selected.
    """

    replacement_path: Path | None = None
    """
    The file containing the new source bytes.
    """

    changed_path: Path | None = None
    """
    The working file changed during query execution.
    """

    def describe_symbol(
        self, symbol_identifier: str, limit: int = 20, offset: int = 0
    ) -> QueryPage:
        """
        Edit source after :meth:`Navigation.describe_symbol` selects evidence.
        """
        result = super().describe_symbol(symbol_identifier, limit, offset)
        if self.replacement_path is not None and self.changed_path is not None:
            shutil.copyfile(self.replacement_path, self.changed_path)
        return result
