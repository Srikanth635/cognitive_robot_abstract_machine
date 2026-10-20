"""
Persist coherent source graphs using transactional SQLite publication.
"""

from __future__ import annotations

import sqlite3
import json
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.contracts import IndexReader, IndexWriter
from repository_knowledge.exceptions import (
    IndexUnavailable,
    SourceScopeMismatch,
    UnsupportedIndexFormat,
)
from repository_knowledge.model import (
    Diagnostic,
    IndexData,
    IndexManifest,
    IndexSelection,
    IndexVersion,
    ReferenceRecord,
    SelectionKind,
    SourceRecord,
    SourceAnalysis,
    SymbolRecord,
    TraversalDirection,
)

if TYPE_CHECKING:
    from typing_extensions import Any

# %% packaged SQL statements


class SQLFile(StrEnum):
    """
    Statements defining persistence and publication for the source index.
    """

    SCHEMA = "schema.sql"
    CLEAR = "clear.sql"
    MANIFEST = "manifest.sql"
    SOURCES = "sources.sql"
    SYMBOLS = "symbols.sql"
    RELATIONSHIPS = "relationships.sql"
    DIAGNOSTICS = "diagnostics.sql"
    INSERT_MANIFEST = "insert_manifest.sql"
    INSERT_SOURCE = "insert_source.sql"
    INSERT_SYMBOL = "insert_symbol.sql"
    INSERT_RELATIONSHIP = "insert_relationship.sql"
    INSERT_DIAGNOSTIC = "insert_diagnostic.sql"
    SYMBOL_DETAILS = "symbol_details.sql"
    EXACT_SYMBOLS = "exact_symbols.sql"
    SEARCH_SYMBOLS = "search_symbols.sql"
    SOURCE_SELECTION = "source_selection.sql"
    INCOMING_SELECTION = "incoming_selection.sql"
    OUTGOING_SELECTION = "outgoing_selection.sql"
    ANALYSES = "analyses.sql"
    INSERT_ANALYSIS = "insert_analysis.sql"
    PUBLICATION = "publication.sql"

    def read(self) -> str:
        """
        Read the packaged statement associated with this persistence operation.
        """
        return (Path(__file__).with_name("sql") / self.value).read_text(
            encoding="utf-8"
        )


class StorageColumn(StrEnum):
    """
    Named columns passed across the SQLite boundary.
    """

    PAYLOAD = "payload"
    PATH = "relative_path"
    SYMBOL = "symbol_identifier"
    LINE = "start_line"
    ORDINAL = "ordinal"
    SOURCE = "source_identifier"
    TARGET = "target_identifier"
    QUERY = "query"
    IDENTIFIERS = "identifiers"
    PATHS = "paths"


class SQLFunction(StrEnum):
    """
    The text comparison supplied to source-evidence queries.
    """

    CASEFOLD = "casefold"


# %% transactional source graph


@dataclass
class SQLiteIndex(IndexReader, IndexWriter):
    """
    Read and atomically publish a complete structural source graph.
    """

    database_path: Path
    """
    The rebuildable SQLite index file.
    """

    codec: RecordCodec = field(default_factory=RecordCodec)
    """
    The owned-evidence serialization boundary.
    """

    timeout_seconds: float = 5.0
    """
    How long a client waits for a conflicting database transaction.
    """

    def replace(self, index: IndexData) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(
            sqlite3.connect(self.database_path, timeout=self.timeout_seconds)
        ) as connection:
            connection.row_factory = sqlite3.Row
            connection.executescript(SQLFile.SCHEMA.read())
            with connection:
                connection.execute(SQLFile.PUBLICATION.read())
                manifests = self._records(connection, SQLFile.MANIFEST, IndexManifest)
                if (
                    manifests
                    and manifests[0].source_scope != index.manifest.source_scope
                ):
                    raise SourceScopeMismatch(
                        manifests[0].source_scope, index.manifest.source_scope
                    )
                for statement in SQLFile.CLEAR.read().split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute(
                    SQLFile.INSERT_MANIFEST.read(),
                    {StorageColumn.PAYLOAD: self.codec.encode(index.manifest)},
                )
                connection.executemany(
                    SQLFile.INSERT_ANALYSIS.read(),
                    (
                        {
                            StorageColumn.PATH: analysis.identity.relative_path,
                            StorageColumn.PAYLOAD: zlib.compress(
                                self.codec.encode(analysis).encode("utf-8")
                            ),
                        }
                        for analysis in index.analyses
                    ),
                )
                connection.executemany(
                    SQLFile.INSERT_SOURCE.read(),
                    [
                        {
                            StorageColumn.PATH: source.relative_path,
                            StorageColumn.PAYLOAD: self.codec.encode(source),
                        }
                        for source in index.sources
                    ],
                )
                connection.executemany(
                    SQLFile.INSERT_SYMBOL.read(),
                    [
                        {
                            StorageColumn.SYMBOL: symbol.symbol_identifier,
                            StorageColumn.PATH: symbol.location.relative_path,
                            StorageColumn.LINE: symbol.location.start_line,
                            StorageColumn.PAYLOAD: self.codec.encode(symbol),
                        }
                        for symbol in index.symbols
                    ],
                )
                connection.executemany(
                    SQLFile.INSERT_RELATIONSHIP.read(),
                    [
                        {
                            StorageColumn.ORDINAL: ordinal,
                            StorageColumn.SOURCE: reference.source_identifier,
                            StorageColumn.TARGET: reference.target_identifier,
                            StorageColumn.PAYLOAD: self.codec.encode(reference),
                        }
                        for ordinal, reference in enumerate(index.references)
                    ],
                )
                connection.executemany(
                    SQLFile.INSERT_DIAGNOSTIC.read(),
                    [
                        {
                            StorageColumn.ORDINAL: ordinal,
                            StorageColumn.PAYLOAD: self.codec.encode(diagnostic),
                        }
                        for ordinal, diagnostic in enumerate(index.diagnostics)
                    ],
                )

    def read(self, selection: IndexSelection | None = None) -> IndexData:
        if not self.database_path.is_file():
            raise IndexUnavailable(str(self.database_path))
        with closing(
            sqlite3.connect(self.database_path, timeout=self.timeout_seconds)
        ) as connection:
            connection.row_factory = sqlite3.Row
            connection.create_function(
                SQLFunction.CASEFOLD, 1, str.casefold, deterministic=True
            )
            with connection:
                connection.execute("BEGIN")
                manifests = self._records(connection, SQLFile.MANIFEST, IndexManifest)
                if len(manifests) != 1:
                    raise IndexUnavailable(str(self.database_path))
                manifest = manifests[0]
                if manifest.format_version != IndexVersion.SOURCE_ANALYSIS:
                    raise UnsupportedIndexFormat(manifest.format_version)
                if selection is not None:
                    return self._select(connection, manifest, selection)
                return IndexData(
                    manifest,
                    self._records(connection, SQLFile.SOURCES, SourceRecord),
                    self._records(connection, SQLFile.SYMBOLS, SymbolRecord),
                    self._records(connection, SQLFile.RELATIONSHIPS, ReferenceRecord),
                    self._records(connection, SQLFile.DIAGNOSTICS, Diagnostic),
                    self.read_analyses(connection),
                )

    def read_analyses(self, connection: sqlite3.Connection) -> list[SourceAnalysis]:
        """
        Decode compressed original evidence from the active read transaction.
        """
        return [
            self.codec.decode(
                zlib.decompress(row[StorageColumn.PAYLOAD]).decode("utf-8"),
                SourceAnalysis,
            )
            for row in connection.execute(SQLFile.ANALYSES.read())
        ]

    def _select(
        self,
        connection: sqlite3.Connection,
        manifest: IndexManifest,
        selection: IndexSelection,
    ) -> IndexData:
        """
        Read the source records needed by one coherent evidence selection.
        """
        if selection.selection_kind == SelectionKind.MANIFEST:
            return IndexData(manifest, [], [], [], [])
        if selection.selection_kind == SelectionKind.TASK_CONTEXT:
            return IndexData(
                manifest,
                self._records(connection, SQLFile.SOURCES, SourceRecord),
                self._records(connection, SQLFile.SYMBOLS, SymbolRecord),
                self._records(connection, SQLFile.RELATIONSHIPS, ReferenceRecord),
                self._records(connection, SQLFile.DIAGNOSTICS, Diagnostic),
            )
        if selection.selection_kind == SelectionKind.KNOWLEDGE:
            return IndexData(
                manifest,
                self._records(connection, SQLFile.SOURCES, SourceRecord),
                self._records(connection, SQLFile.SYMBOLS, SymbolRecord),
                [],
                self._records(connection, SQLFile.DIAGNOSTICS, Diagnostic),
            )
        if selection.selection_kind == SelectionKind.ANALYSES:
            return IndexData(
                manifest,
                self._records(connection, SQLFile.SOURCES, SourceRecord),
                [],
                [],
                [],
                self.read_analyses(connection),
            )
        symbols: list[SymbolRecord] = []
        references: list[ReferenceRecord] = []
        if selection.selection_kind == SelectionKind.SYMBOL_DETAILS:
            symbols = self._records(
                connection,
                SQLFile.SYMBOL_DETAILS,
                SymbolRecord,
                {StorageColumn.SYMBOL: selection.symbol_identifier},
            )
        elif selection.selection_kind == SelectionKind.SYMBOL_SEARCH:
            parameters = {StorageColumn.QUERY: selection.query}
            symbols = self._records(
                connection, SQLFile.EXACT_SYMBOLS, SymbolRecord, parameters
            )
            if not symbols:
                symbols = self._records(
                    connection, SQLFile.SEARCH_SYMBOLS, SymbolRecord, parameters
                )
        elif selection.selection_kind == SelectionKind.REFERENCES:
            references = self._neighbourhood(
                connection,
                IndexSelection(
                    SelectionKind.RELATIONSHIPS,
                    symbol_identifier=selection.symbol_identifier,
                    direction=TraversalDirection.INCOMING,
                ),
            )
        else:
            references = self._neighbourhood(connection, selection)
        paths = sorted(
            {record.location.relative_path for record in [*symbols, *references]}
        )
        sources = self._records(
            connection,
            SQLFile.SOURCE_SELECTION,
            SourceRecord,
            {StorageColumn.PATHS: json.dumps(paths)},
        )
        diagnostics = self._records(connection, SQLFile.DIAGNOSTICS, Diagnostic)
        return IndexData(manifest, sources, symbols, references, diagnostics)

    def _neighbourhood(
        self, connection: sqlite3.Connection, selection: IndexSelection
    ) -> list[ReferenceRecord]:
        """
        Read indexed edges at the requested relationship distance.
        """
        statement = (
            SQLFile.OUTGOING_SELECTION
            if selection.direction == TraversalDirection.OUTGOING
            else SQLFile.INCOMING_SELECTION
        )
        frontier = {selection.symbol_identifier}
        visited: set[str] = set()
        references_by_ordinal: dict[int, ReferenceRecord] = {}
        for distance in range(selection.depth):
            current = frontier - visited
            if not current:
                break
            visited.update(current)
            edges: list[ReferenceRecord] = []
            for row in connection.execute(
                statement.read(),
                {StorageColumn.IDENTIFIERS: json.dumps(sorted(current))},
            ):
                ordinal = row[StorageColumn.ORDINAL]
                if ordinal not in references_by_ordinal:
                    references_by_ordinal[ordinal] = self.codec.decode(
                        row[StorageColumn.PAYLOAD], ReferenceRecord
                    )
                edges.append(references_by_ordinal[ordinal])
            frontier = {
                (
                    reference.target_identifier
                    if selection.direction == TraversalDirection.OUTGOING
                    else reference.source_identifier
                )
                for reference in edges
                if reference.target_identifier
            }
        return [
            references_by_ordinal[ordinal] for ordinal in sorted(references_by_ordinal)
        ]

    def _records(
        self,
        connection: sqlite3.Connection,
        statement: SQLFile,
        record_class: type,
        parameters: dict[str, object] | None = None,
    ) -> list[Any]:
        """
        Decode ordered rows through the fixed evidence vocabulary.
        """
        return [
            self.codec.decode(row[StorageColumn.PAYLOAD], record_class)
            for row in connection.execute(statement.read(), parameters or {})
        ]
