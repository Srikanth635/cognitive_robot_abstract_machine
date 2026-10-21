"""
Interrupt a real SQLite publication before its transaction commits.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.model import SymbolRecord

# %% interrupted transaction


class ProcessExit(IntEnum):
    """
    The intentional termination after manifest replacement.
    """

    INTERRUPTED = 29


@dataclass
class InterruptedCodec(RecordCodec):
    """
    Terminate when publication reaches declarations after clearing old rows.
    """

    def encode(self, record: object) -> str:
        if isinstance(record, SymbolRecord):
            os._exit(ProcessExit.INTERRUPTED)
        return super().encode(record)


def main() -> None:
    """
    Start a replacement that terminates before publication completes.
    """
    storage = SQLiteIndex(Path(sys.argv[1]))
    original = storage.read()
    replacement = replace(
        original, manifest=replace(original.manifest, snapshot_identifier="interrupted")
    )
    SQLiteIndex(storage.database_path, InterruptedCodec()).replace(replacement)


if __name__ == "__main__":
    main()
