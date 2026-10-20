"""
Read-only startup inspection through the configured source adapters.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.clients import SessionReport, SessionState
from repository_knowledge.interfaces.cli import Command, CommandArguments, CommandParser
from repository_knowledge.model import SourceScope
from repository_knowledge.operations.freshness import FreshnessCheck

# %% inexpensive startup inspection


class SessionPath(StrEnum):
    """
    The local curated knowledge entry point.
    """

    WIKI = "repository_knowledge/wiki/index.md"


@dataclass
class SessionApplication:
    """
    Inspect an existing cache without indexing or installing anything.
    """

    def inspect(self, arguments: CommandArguments) -> SessionReport:
        """
        Report current, stale or missing evidence and source-only guidance.
        """
        if not arguments.database_path.is_file():
            return SessionReport(
                SessionState.MISSING,
                "",
                "",
                SessionPath.WIKI.value,
                "Knowledge index missing; use source navigation. Wiki: "
                + SessionPath.WIKI,
            )
        source_reader = (
            WorkingTreeSource(
                arguments.repository_root, arguments.included_untracked_paths
            )
            if arguments.source_scope == SourceScope.WORKING_TREE
            else GitSource(arguments.repository_root)
        )
        freshness = FreshnessCheck(
            SQLiteIndex(arguments.database_path), source_reader
        ).check(arguments.revision)
        state = (
            SessionState.CURRENT
            if freshness.matches_requested_revision
            and freshness.matches_requested_source
            else SessionState.STALE
        )
        guidance = (
            "Use bounded queries; local wiki concepts are drafts."
            if state == SessionState.CURRENT
            else "Use source navigation until the index is refreshed."
        )
        return SessionReport(
            state,
            freshness.requested_commit,
            freshness.snapshot_identifier,
            SessionPath.WIKI.value,
            f"Knowledge index {state} for {freshness.requested_commit}. {guidance} Wiki: {SessionPath.WIKI}",
        )


# %% versioned process entry


def main() -> int:
    """
    Print a bounded report for an external startup client.
    """
    arguments = CommandParser().parse([*sys.argv[1:], Command.CHECK_FRESHNESS])
    print(RecordCodec().encode(SessionApplication().inspect(arguments)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
