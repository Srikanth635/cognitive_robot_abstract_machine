"""
Bounded navigation over evidence from a single indexed revision.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.contracts import IndexReader
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.model import (
    IndexData,
    IndexSelection,
    QueryPage,
    ReferenceRecord,
    Resolution,
    SelectionKind,
    SourceExcerpt,
    SourceLocation,
    SymbolRecord,
    TraversalDirection,
)

# %% bounded source navigation


@dataclass
class IndexContents(IndexReader):
    """
    Provide one completed source graph throughout an operation.
    """

    index: IndexData
    """The graph captured before the operation starts."""

    def read(self, selection: IndexSelection | None = None) -> IndexData:
        return self.index


@dataclass
class Navigation:
    """
    Answer source-navigation questions without editing repository state.
    """

    index_reader: IndexReader
    """
    The completed source graph being queried.
    """

    maximum_results: int = 100
    """
    The maximum evidence items returned by one query page.
    """

    maximum_depth: int = 8
    """
    The maximum relationship distance accepted by a traversal.
    """

    maximum_excerpt_lines: int = 80
    """
    The maximum source lines returned for each evidence location.
    """

    def __post_init__(self) -> None:
        """
        Reject query budgets that cannot contain source evidence.
        """
        if (
            min(self.maximum_results, self.maximum_depth, self.maximum_excerpt_lines)
            < 1
        ):
            raise InvalidQuery("Query budgets must be positive.")

    def _bounds(self, limit: int, offset: int) -> None:
        """
        Reject result bounds that exceed the configured query budget.
        """
        if limit < 1 or limit > self.maximum_results or offset < 0:
            raise InvalidQuery(
                "The requested result page is outside its configured bounds."
            )

    def find_symbols(self, query: str, limit: int = 20, offset: int = 0) -> QueryPage:
        """
        Find qualified, exact-name and lexical declaration matches.
        """
        self._bounds(limit, offset)
        index = self.index_reader.read(
            IndexSelection(SelectionKind.SYMBOL_SEARCH, query=query)
        )
        needle = query.casefold()
        exact_matches = [
            symbol
            for symbol in index.symbols
            if needle
            in (symbol.symbol_identifier.casefold(), symbol.symbol_name.casefold())
        ]
        matches = exact_matches or [
            symbol
            for symbol in index.symbols
            if needle in symbol.symbol_identifier.casefold()
            or needle in symbol.documentation.casefold()
        ]
        matches.sort(
            key=lambda symbol: (
                symbol.symbol_identifier.casefold() != needle,
                symbol.symbol_name.casefold() != needle,
                symbol.symbol_identifier,
                symbol.location.start_line,
            )
        )
        selected = matches[offset : offset + limit]
        return self._page(index, selected, [], len(matches), limit, offset)

    def describe_symbol(
        self, symbol_identifier: str, limit: int = 20, offset: int = 0
    ) -> QueryPage:
        """
        Describe exact declaration identities and their committed source.
        """
        self._bounds(limit, offset)
        index = self.index_reader.read(
            IndexSelection(
                SelectionKind.SYMBOL_DETAILS, symbol_identifier=symbol_identifier
            )
        )
        matches = [
            symbol
            for symbol in index.symbols
            if symbol.symbol_identifier == symbol_identifier
        ]
        return self._page(
            index, matches[offset : offset + limit], [], len(matches), limit, offset
        )

    def find_references(
        self, symbol_identifier: str, limit: int = 20, offset: int = 0
    ) -> QueryPage:
        """
        Return incoming resolved relationships with their source evidence.
        """
        self._bounds(limit, offset)
        index = self.index_reader.read(
            IndexSelection(
                SelectionKind.REFERENCES, symbol_identifier=symbol_identifier
            )
        )
        matches = [
            reference
            for reference in index.references
            if reference.target_identifier == symbol_identifier
        ]
        return self._page(
            index, [], matches[offset : offset + limit], len(matches), limit, offset
        )

    def find_relationships(
        self,
        symbol_identifier: str,
        direction: TraversalDirection = TraversalDirection.OUTGOING,
        depth: int = 1,
        limit: int = 20,
        offset: int = 0,
    ) -> QueryPage:
        """
        Traverse a bounded incoming or outgoing relationship neighbourhood.
        """
        self._bounds(limit, offset)
        if (
            depth < 1
            or depth > self.maximum_depth
            or direction not in tuple(TraversalDirection)
        ):
            raise InvalidQuery(
                "The requested relationship traversal is outside its configured bounds."
            )
        index = self.index_reader.read(
            IndexSelection(
                SelectionKind.RELATIONSHIPS,
                symbol_identifier=symbol_identifier,
                direction=direction,
                depth=depth,
                maximum_depth=self.maximum_depth,
            )
        )
        frontier = {symbol_identifier}
        visited: set[str] = set()
        matches: list[ReferenceRecord] = []
        for distance in range(depth):
            current = frontier - visited
            if not current:
                break
            visited.update(current)
            edges = [
                reference
                for reference in index.references
                if (
                    reference.source_identifier
                    if direction == TraversalDirection.OUTGOING
                    else reference.target_identifier
                )
                in current
            ]
            matches.extend(reference for reference in edges if reference not in matches)
            frontier = {
                (
                    reference.target_identifier
                    if direction == TraversalDirection.OUTGOING
                    else reference.source_identifier
                )
                for reference in edges
                if reference.target_identifier
            }
        return self._page(
            index, [], matches[offset : offset + limit], len(matches), limit, offset
        )

    def _page(
        self,
        index: IndexData,
        symbols: list[SymbolRecord],
        references: list[ReferenceRecord],
        total_matches: int,
        limit: int,
        offset: int,
    ) -> QueryPage:
        """
        Attach revision, diagnostics and exact source slices to bounded evidence.
        """
        sources = {source.relative_path: source for source in index.sources}
        locations = [symbol.location for symbol in symbols] + [
            reference.location for reference in references
        ]
        excerpts: list[SourceExcerpt] = []
        for location in dict.fromkeys(locations):
            source = sources[location.relative_path]
            end_line = min(
                location.end_line, location.start_line + self.maximum_excerpt_lines - 1
            )
            extent = SourceLocation(
                location.relative_path, location.start_line, end_line
            )
            excerpts.append(
                SourceExcerpt(
                    extent,
                    "\n".join(
                        source.source_text.splitlines()[
                            location.start_line - 1 : end_line
                        ]
                    ),
                    location.end_line - end_line,
                )
            )
        paths = {location.relative_path for location in locations}
        diagnostics = [
            diagnostic
            for diagnostic in index.diagnostics
            if not paths or diagnostic.relative_path in paths
        ]
        next_offset = offset + limit if offset + limit < total_matches else None
        return QueryPage(
            index.manifest.snapshot_identifier,
            symbols,
            references,
            excerpts,
            diagnostics,
            total_matches,
            next_offset,
            Resolution.RESOLVED if total_matches else Resolution.UNRESOLVED,
            source_scope=index.manifest.source_scope,
        )
