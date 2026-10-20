"""
Optional integrations cannot become requirements of ordinary source queries.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.model import QueryPage

from .support import CommittedProject, FixtureSymbol

# %% source-only installation


def test_cli_queries_work_without_the_optional_mcp_sdk(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    An SDK outage cannot prevent the existing CLI from navigating source.
    """
    script = Path(__file__).parent / "dataset/without_mcp.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "find-symbols",
            FixtureSymbol.CHILD,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    page = RecordCodec().decode(result.stdout, QueryPage)
    assert [symbol.symbol_identifier for symbol in page.symbols] == [
        FixtureSymbol.CHILD
    ]
