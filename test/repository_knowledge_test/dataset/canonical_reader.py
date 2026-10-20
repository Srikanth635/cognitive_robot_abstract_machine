"""
Authenticated-consumer results isolated from the client composition tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from repository_knowledge.contracts import CanonicalReader
from repository_knowledge.publication import ConsumptionResult

# %% consumer boundary mimic


@dataclass
class CapturedConsumption(CanonicalReader):
    """
    Supply an already evaluated trust verdict to a context client.
    """

    result: ConsumptionResult
    """The independently evaluated canonical result."""
    requested_revisions: list[str] = field(default_factory=list)
    """
    The exact source identities requested by the context client.
    """

    def read(self, revision: str) -> ConsumptionResult:
        self.requested_revisions.append(revision)
        return self.result
