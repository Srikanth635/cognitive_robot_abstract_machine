"""
Persistence publishes complete snapshots and preserves typed evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.exceptions import InvalidSerializedType
from repository_knowledge.model import IndexData

# %% persistence evidence


def test_sqlite_round_trip_preserves_source_index(
    source_index: IndexData, indexed_storage: SQLiteIndex
) -> None:
    """
    Persisted declarations and relationships retain their exact source evidence.
    """
    assert indexed_storage.read() == source_index


def test_index_publication_replaces_previous_snapshot(
    source_index: IndexData, indexed_storage: SQLiteIndex
) -> None:
    """
    Publishing a new snapshot does not retain removed declarations.
    """
    replacement = IndexData(
        source_index.manifest, source_index.sources, [], [], source_index.diagnostics
    )
    indexed_storage.replace(replacement)
    assert indexed_storage.read() == replacement


def test_unknown_serialized_type_is_rejected_before_import(tmp_path: Path) -> None:
    """
    An index payload cannot select arbitrary importable classes.
    """
    payload = (Path(__file__).parent / "dataset/unknown_type.json").read_text(
        encoding="utf-8"
    )
    with pytest.raises(InvalidSerializedType):
        RecordCodec().decode(payload, IndexData)
