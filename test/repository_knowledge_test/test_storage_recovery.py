"""
Publication failures preserve a complete recoverable source snapshot.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.interfaces.cli import CommandParser
from repository_knowledge.interfaces.session import SessionApplication
from repository_knowledge.model import IndexData

from .dataset.interrupted_publication import ProcessExit
from .dataset.publication_failures import CapacityConnection, PausedCodec
from .support import CommittedProject

# %% abrupt exit and reader isolation


def test_restart_recovers_an_interrupted_publication(
    source_index: IndexData, indexed_storage: SQLiteIndex
) -> None:
    """
    An abrupt process exit cannot expose its uncommitted replacement.
    """
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "dataset/interrupted_publication.py"),
            str(indexed_storage.database_path),
        ],
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == ProcessExit.INTERRUPTED
    restarted = SQLiteIndex(indexed_storage.database_path)
    assert restarted.read() == source_index
    replacement = replace(source_index, symbols=[], references=[])
    restarted.replace(replacement)
    assert restarted.read() == replacement


def test_concurrent_reader_observes_only_complete_publications(
    source_index: IndexData, indexed_storage: SQLiteIndex
) -> None:
    """
    Readers see the prior snapshot until the replacement commits.
    """
    codec = PausedCodec()
    writer = SQLiteIndex(indexed_storage.database_path, codec)
    replacement = replace(
        source_index,
        manifest=replace(source_index.manifest, snapshot_identifier="next-publication"),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(writer.replace, replacement)
        started = codec.publication_started.wait(codec.deadline_seconds)
        observed = indexed_storage.read()
        codec.reader_finished.set()
        publication.result(timeout=codec.deadline_seconds)
    assert started is True
    assert observed == source_index
    assert indexed_storage.read() == replacement


# %% capacity and corruption


def test_full_disk_preserves_the_prior_snapshot(
    source_index: IndexData,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    SQLite's page-cap failure rolls back rows already replaced in the transaction.
    """
    connect = sqlite3.connect
    monkeypatch.setattr(
        "repository_knowledge.adapters.sqlite_index.sqlite3.connect",
        CapacityConnection(connect),
    )
    replacement = replace(
        source_index,
        sources=[
            replace(source, source_text=source.source_text * 65536)
            for source in source_index.sources
        ],
    )
    with pytest.raises(sqlite3.OperationalError) as failure:
        indexed_storage.replace(replacement)
    assert failure.value.sqlite_errorcode == sqlite3.SQLITE_FULL
    assert indexed_storage.read() == source_index


def test_corrupt_cache_never_reports_current_source(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Startup fails visibly instead of granting freshness to invalid cache bytes.
    """
    database = tmp_path / "corrupt.sqlite3"
    shutil.copyfile(Path(__file__).parent / "dataset/corrupt.sqlite3", database)
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(database),
            "check-freshness",
        ]
    )
    with pytest.raises(sqlite3.DatabaseError) as failure:
        SessionApplication().inspect(arguments)
    assert failure.value.sqlite_errorcode == sqlite3.SQLITE_NOTADB


def test_quarantined_corruption_can_be_rebuilt(
    source_index: IndexData, tmp_path: Path
) -> None:
    """
    Retaining corrupt bytes separately permits an explicit fresh publication.
    """
    database = tmp_path / "index.sqlite3"
    corrupt = Path(__file__).parent / "dataset/corrupt.sqlite3"
    shutil.copyfile(corrupt, database)
    quarantine = database.with_suffix(".quarantine")
    database.rename(quarantine)
    storage = SQLiteIndex(database)
    storage.replace(source_index)
    assert storage.read() == source_index
    assert quarantine.read_bytes() == corrupt.read_bytes()


def test_reader_lock_deadline_does_not_replace_evidence(
    source_index: IndexData, indexed_storage: SQLiteIndex
) -> None:
    """
    A competing exclusive transaction yields a visible lock failure.
    """
    with closing(sqlite3.connect(indexed_storage.database_path)) as connection:
        connection.executescript(
            (Path(__file__).parent / "dataset/locked.sql").read_text(encoding="utf-8")
        )
        with pytest.raises(sqlite3.OperationalError) as failure:
            SQLiteIndex(indexed_storage.database_path, timeout_seconds=0.0).read()
        assert failure.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        connection.rollback()
    assert indexed_storage.read() == source_index
