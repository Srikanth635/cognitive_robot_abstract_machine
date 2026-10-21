"""
Deterministic fault boundaries for transactional publication tests.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.model import SymbolRecord

if TYPE_CHECKING:
    from typing_extensions import Callable

# %% coordinated publication


@dataclass
class PausedCodec(RecordCodec):
    """
    Hold an unfinished replacement while a reader inspects the old snapshot.
    """

    publication_started: Event = field(default_factory=Event)
    """
    The signal that replacement reached the declaration stage.
    """

    reader_finished: Event = field(default_factory=Event)
    """
    The signal allowing the publication transaction to complete.
    """

    deadline_seconds: float = 10.0
    """
    The maximum wait before an abandoned coordination fails.
    """

    def encode(self, record: object) -> str:
        if isinstance(record, SymbolRecord):
            self.publication_started.set()
            if not self.reader_finished.wait(self.deadline_seconds):
                raise TimeoutError("The snapshot reader did not finish.")
        return super().encode(record)


@dataclass
class CapacityConnection:
    """
    Limit new pages while retaining SQLite's actual write-failure behavior.
    """

    connection_factory: Callable[..., sqlite3.Connection]
    """
    The SQLite connection constructor without fault injection.
    """

    def __call__(self, database: Path, timeout: float) -> sqlite3.Connection:
        """
        Open a database whose maximum page count cannot grow.
        """
        connection = self.connection_factory(database, timeout=timeout)
        connection.executescript(
            (Path(__file__).with_name("full_disk.sql")).read_text(encoding="utf-8")
        )
        return connection
