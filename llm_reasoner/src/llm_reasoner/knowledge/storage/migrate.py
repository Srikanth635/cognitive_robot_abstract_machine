"""
``python -m llm_reasoner.knowledge.storage.migrate store <directory> [--backup
<label>]`` migrates a store of hypotheses through the shipped migrations it has not been
migrated through; ``python -m llm_reasoner.knowledge.storage.migrate files <migration>
<path>...`` migrates recorded answers and Turtle fixtures through one shipped migration.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path

from typing_extensions import Optional, Sequence

from llm_reasoner.knowledge.storage.migration import (
    RecordedFileMigration,
    StoreMigration,
    shipped_migrations,
)


class MigrationCommand(StrEnum):
    """
    What the command line migrates.
    """

    STORE = "store"
    """
    A store of hypotheses.
    """

    FILES = "files"
    """
    Recorded answers and Turtle fixtures.
    """


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """
    Migrate what the command line names, and print what changed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    store = commands.add_parser(MigrationCommand.STORE)
    store.add_argument("directory", type=Path)
    store.add_argument("--backup")
    files = commands.add_parser(MigrationCommand.FILES)
    files.add_argument("migration")
    files.add_argument("paths", type=Path, nargs="+")
    parsed = parser.parse_args(arguments)
    if parsed.command == MigrationCommand.STORE:
        migration = StoreMigration(directory=parsed.directory)
        if parsed.backup:
            print(f"backed up to {migration.back_up(parsed.backup)}")
        report = migration.run()
        print(f"migrated: {report.migrated}\nstale: {report.stale}")
        return
    (chosen,) = [
        migration
        for migration in shipped_migrations()
        if migration.name == parsed.migration
    ]
    for path in RecordedFileMigration(migration=chosen).run(parsed.paths):
        print(f"migrated {path}")


if __name__ == "__main__":
    main()
