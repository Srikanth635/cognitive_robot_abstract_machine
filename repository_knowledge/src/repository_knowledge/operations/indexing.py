"""
Build source graphs and resolve only supported lexical relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from repository_knowledge.contracts import SnapshotReader, SourceAnalyzer
from repository_knowledge.model import (
    AnalysisIdentity,
    ImportBinding,
    IndexData,
    ReferenceRecord,
    RelationshipKind,
    Resolution,
    RepositorySnapshot,
    SourceAnalysis,
    SourceKind,
    SymbolKind,
    SymbolRecord,
)

# %% index construction


@dataclass
class IndexBuilder:
    """
    Assemble one source graph using explicit source and analysis contracts.
    """

    source_reader: SnapshotReader
    """The committed-source provider."""

    source_analyzer: SourceAnalyzer
    """
    The declaration and lexical-evidence provider.
    """

    def build(self, revision: str, previous: IndexData | None = None) -> IndexData:
        """
        Build deterministic evidence for the requested source revision.
        """
        snapshot = self.source_reader.read_snapshot(revision)
        analyses = self.analyze(snapshot, previous)
        return self.assemble(snapshot, analyses)

    def analyze(
        self, snapshot: RepositorySnapshot, previous: IndexData | None = None
    ) -> list[SourceAnalysis]:
        """
        Reuse file evidence only while its source and interpretation agree.
        """
        reusable: dict[AnalysisIdentity, SourceAnalysis] = {}
        if previous is not None and (
            previous.manifest.python_version == snapshot.manifest.python_version
            and previous.manifest.extractor_version
            == snapshot.manifest.extractor_version
            and previous.manifest.format_version == snapshot.manifest.format_version
            and previous.manifest.source_scope == snapshot.manifest.source_scope
            and previous.manifest.worktree_identifier
            == snapshot.manifest.worktree_identifier
        ):
            reusable = {analysis.identity: analysis for analysis in previous.analyses}
        analyses: list[SourceAnalysis] = []
        for source in snapshot.sources:
            if source.source_kind not in (SourceKind.PYTHON, SourceKind.DOCUMENT):
                continue
            identity = AnalysisIdentity.from_source(source)
            analysis = reusable.get(identity)
            if analysis is None:
                analysis = SourceAnalysis(
                    identity, self.source_analyzer.extract(source)
                )
            analyses.append(analysis)
        return analyses

    def assemble(
        self, snapshot: RepositorySnapshot, analyses: list[SourceAnalysis]
    ) -> IndexData:
        """
        Resolve current lexical targets from the complete file-evidence set.
        """
        extractions = [analysis.extraction for analysis in analyses]
        symbols = [
            symbol for extraction in extractions for symbol in extraction.symbols
        ]
        bindings = [
            binding for extraction in extractions for binding in extraction.bindings
        ]
        resolver = ReferenceResolver(symbols, bindings)
        references = [
            resolver.resolve(reference)
            for extraction in extractions
            for reference in extraction.references
        ]
        return IndexData(
            snapshot.manifest,
            snapshot.sources,
            sorted(
                symbols,
                key=lambda symbol: (
                    symbol.symbol_identifier,
                    symbol.location.start_line,
                ),
            ),
            sorted(
                references,
                key=lambda reference: (
                    reference.location.relative_path,
                    reference.location.start_line,
                    reference.source_identifier,
                    reference.relationship_kind,
                    reference.target_expression,
                ),
            ),
            snapshot.diagnostics
            + [
                diagnostic
                for extraction in extractions
                for diagnostic in extraction.diagnostics
            ],
            analyses,
        )


# %% lexical target resolution


@dataclass
class ReferenceResolver:
    """
    Resolve import bindings and lexical declarations without guessing dispatch.
    """

    symbols: list[SymbolRecord]
    """
    The complete available declaration inventory.
    """

    bindings: list[ImportBinding]
    """
    The imported and shadowed names collected from source.
    """

    declarations_by_identifier: dict[str, SymbolRecord] = field(init=False)
    """
    The declaration context indexed once for target resolution.
    """

    bindings_by_scope: dict[BindingIdentity, str] = field(init=False)
    """
    Lexical targets indexed by scope and local name.
    """

    def __post_init__(self) -> None:
        """
        Index declaration and binding contexts before resolving references.
        """
        self.declarations_by_identifier = {
            symbol.symbol_identifier: symbol for symbol in self.symbols
        }
        self.bindings_by_scope = {
            BindingIdentity(
                binding.scope_identifier, binding.local_name
            ): binding.target_name
            for binding in self.bindings
        }

    def resolve(self, reference: ReferenceRecord) -> ReferenceRecord:
        """
        Identify a target only when lexical evidence determines it.
        """
        if reference.resolution == Resolution.RESOLVED:
            return reference
        if reference.relationship_kind == RelationshipKind.DOCUMENTS:
            resource = next(
                (
                    symbol
                    for symbol in self.symbols
                    if symbol.location.relative_path == reference.target_source_path
                    and symbol.symbol_kind in (SymbolKind.MODULE, SymbolKind.DOCUMENT)
                ),
                None,
            )
            if resource is None:
                return reference
            target = resource.symbol_identifier
            if reference.target_fragment:
                target += "." + reference.target_fragment
            if target in self.declarations_by_identifier:
                return replace(
                    reference, target_identifier=target, resolution=Resolution.RESOLVED
                )
            return reference
        expression = reference.target_expression.split("[", 1)[0]
        if not expression or not all(
            part.isidentifier() for part in expression.split(".")
        ):
            return reference
        declarations = self.declarations_by_identifier
        bindings = self.bindings_by_scope
        owner = declarations.get(reference.source_identifier)
        if owner is None:
            return reference
        root_name, separator, suffix = expression.partition(".")
        scope = reference.source_identifier
        visited: set[str] = set()
        while scope:
            scope_declaration = declarations.get(scope)
            class_scope = (
                scope_declaration is not None
                and scope_declaration.symbol_kind in (SymbolKind.CLASS, SymbolKind.ENUM)
            )
            if not class_scope or scope == reference.source_identifier:
                key = BindingIdentity(scope, root_name)
                if key in bindings:
                    target = bindings[key]
                    if not target:
                        return reference
                    expression = target + (separator + suffix if separator else "")
                    break
                candidate = f"{scope}.{expression}"
                if candidate in declarations:
                    expression = candidate
                    break
            if scope == owner.module_name:
                break
            scope = scope.rpartition(".")[0]
        while expression not in visited:
            visited.add(expression)
            if expression in declarations:
                return replace(
                    reference,
                    target_identifier=expression,
                    resolution=Resolution.RESOLVED,
                )
            module_name, separator, local_name = expression.rpartition(".")
            target = bindings.get(BindingIdentity(module_name, local_name))
            if not separator or not target or target == expression:
                return reference
            expression = target
        return reference


@dataclass(frozen=True)
class BindingIdentity:
    """
    Identify a name within its lexical scope.
    """

    scope_identifier: str
    """
    The module or declaration supplying the scope.
    """

    local_name: str
    """
    The name bound inside that scope.
    """
