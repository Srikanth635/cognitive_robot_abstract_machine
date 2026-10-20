"""
A source/index replacement interleaved with canonical verification.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.contracts import CanonicalReader, IndexWriter
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.publication import ConsumptionResult

from ..support import CommittedProject

# %% moving source mimic


@dataclass
class AdvancingConsumption(CanonicalReader):
    """
    Advance source and cache while the old publication is being admitted.
    """

    result: ConsumptionResult
    """The verdict for the initially selected source."""
    project: CommittedProject
    """
    The disposable source history changed during verification.
    """

    index_writer: IndexWriter
    """The local cache replaced with the newer source graph."""

    def read(self, revision: str) -> ConsumptionResult:
        self.project.git(
            "commit", "--quiet", "--allow-empty", "-m", "Advance during canonical read"
        )
        self.index_writer.replace(
            IndexBuilder(GitSource(self.project.root), PythonSource()).build("HEAD")
        )
        return self.result
