"""
The versioned session report survives its actual executable boundary.
"""

from __future__ import annotations

import subprocess
import sys

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.clients import SessionReport, SessionState
from repository_knowledge.interfaces.session import SessionPath

from .support import CommittedProject

# %% report process serialization


def test_session_subprocess_report_decodes_in_the_parent_interpreter(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Executable-local enums cannot leak into the portable report vocabulary.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge.interfaces.session",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = RecordCodec().decode(result.stdout, SessionReport)
    assert report.state == SessionState.CURRENT
    assert report.wiki_entrypoint == SessionPath.WIKI.value
