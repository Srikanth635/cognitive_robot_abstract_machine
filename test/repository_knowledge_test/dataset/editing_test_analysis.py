"""
Introduce a source edit during static test discovery.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.assistance import TestCatalog
from repository_knowledge.contracts import TestAnalyzer
from repository_knowledge.model import IndexData

# %% capture interleaving


@dataclass
class EditingTestAnalysis(TestAnalyzer):
    """
    Change captured source after discovering its static tests.
    """

    replacement_path: Path
    """The replacement source dataset."""
    changed_path: Path
    """
    The working source edited during the query.
    """

    def discover(self, index: IndexData) -> TestCatalog:
        result = PytestSource().discover(index)
        shutil.copyfile(self.replacement_path, self.changed_path)
        return result
