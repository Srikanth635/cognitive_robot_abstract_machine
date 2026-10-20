"""Source identities and evidence shared by indexing and navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import PurePosixPath, PureWindowsPath

from repository_knowledge.exceptions import (
    InvalidQuery,
    InvalidSourceLocation,
    SourceRevisionMismatch,
    SourceSnapshotMismatch,
)

# %% evidence vocabulary


class SourceKind(StrEnum):
    """Distinguish supported source material from recorded exclusions."""

    PYTHON = "python"
    DOCUMENT = "document"
    CONFIGURATION = "configuration"
    TEMPLATE = "template"
    EXCLUDED = "excluded"
    UNSUPPORTED = "unsupported"


class SymbolKind(StrEnum):
    """Describe the declaration represented by a source symbol."""

    MODULE = "module"
    CLASS = "class"
    ENUM = "enum"
    FUNCTION = "function"
    METHOD = "method"
    FIELD = "field"
    ENUM_MEMBER = "enum_member"
    VARIABLE = "variable"
    TYPE_ALIAS = "type_alias"
    DOCUMENT = "document"
    DOCUMENT_SECTION = "document_section"


class RelationshipKind(StrEnum):
    """Describe evidence connecting a source declaration to another name."""

    DEFINES = "defines"
    IMPORTS = "imports"
    INHERITS = "inherits"
    ANNOTATES = "annotates"
    REFERENCES = "references"
    CALLS = "calls"
    DOCUMENTS = "documents"


class Resolution(StrEnum):
    """Describe how precisely evidence identifies its target."""

    RESOLVED = "resolved"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"


class DiagnosticKind(StrEnum):
    """Describe an extraction limitation or intentional inventory exclusion."""

    GENERATED = "generated"
    BINARY = "binary"
    VENDORED = "vendored"
    UNSUPPORTED = "unsupported"
    SYNTAX = "syntax"
    ENCODING = "encoding"
    SYMBOLIC_LINK = "symbolic_link"
    SUBMODULE = "submodule"
    UNRESOLVED = "unresolved"


class IndexVersion(IntEnum):
    """Identify supported semantic source-index formats."""

    SYMBOL_GRAPH = 1
    SOURCE_ANALYSIS = 2


class SourceScope(StrEnum):
    """The source state represented by returned evidence."""

    COMMITTED = "committed"
    WORKING_TREE = "working-tree"


class TraversalDirection(StrEnum):
    """The direction in which source relationships are followed."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"


# %% source records


class SelectionKind(StrEnum):
    """The evidence required by a committed-source query."""

    MANIFEST = "manifest"
    KNOWLEDGE = "knowledge"
    TASK_CONTEXT = "task_context"
    ANALYSES = "analyses"
    SYMBOL_SEARCH = "symbol_search"
    SYMBOL_DETAILS = "symbol_details"
    REFERENCES = "references"
    RELATIONSHIPS = "relationships"


@dataclass(frozen=True)
class IndexSelection:
    """Select source evidence while retaining one snapshot identity."""

    selection_kind: SelectionKind
    """The evidence category required by the operation."""

    symbol_identifier: str = ""
    """The qualified declaration selected for navigation."""

    query: str = ""
    """The exact-name or lexical declaration query."""

    direction: TraversalDirection = TraversalDirection.OUTGOING
    """The direction followed by relationship traversal."""

    depth: int = 1
    """The number of relationship steps included in the selection."""

    maximum_depth: int = 8
    """The maximum relationship distance allowed by the operation."""

    def __post_init__(self) -> None:
        if self.depth < 1 or self.depth > self.maximum_depth:
            raise InvalidQuery("Relationship selection exceeds its configured depth.")


@dataclass(frozen=True)
class SourceLocation:
    """An inclusive line range within a repository-relative source file."""

    relative_path: str
    """The source path relative to its repository snapshot."""

    start_line: int
    """The first included one-based line."""

    end_line: int
    """The final included one-based line."""

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or PureWindowsPath(self.relative_path).drive
            or ".." in path.parts
            or "\\" in self.relative_path
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise InvalidSourceLocation(
                self.relative_path, self.start_line, self.end_line
            )


@dataclass(frozen=True)
class SourceRecord:
    """Captured source contents and their canonical import context."""

    relative_path: str
    """The file's repository-relative POSIX path."""

    object_identifier: str
    """The committed Git object, or empty for working-file contents."""

    content_digest: str
    """The exact byte digest, or file-mode marker for excluded contents."""

    source_kind: SourceKind
    """The selected analysis category."""

    source_text: str = ""
    """Decoded source text for supported material."""

    module_name: str = ""
    """The canonical module name for Python source."""

    distribution_name: str = ""
    """The project distribution owning the source root."""

    is_package: bool = False
    """Whether the module is a package initializer."""


@dataclass(frozen=True)
class SymbolRecord:
    """A declaration identified by its module and lexical qualification."""

    symbol_identifier: str
    """The canonical qualified declaration identity."""

    symbol_name: str
    """The declaration's unqualified source name."""

    module_name: str
    """The module containing the declaration."""

    symbol_kind: SymbolKind
    """The declaration category."""

    location: SourceLocation
    """The exact declaration extent."""

    signature: str = ""
    """The normalized declaration header or annotated assignment."""

    documentation: str = ""
    """Source documentation attached to the declaration."""

    base_expressions: list[str] = field(default_factory=list)
    """Unmodified semantic base expressions, including generic parameters."""

    decorators: list[str] = field(default_factory=list)
    """The declaration's decorator expressions."""

    import_statements: list[str] = field(default_factory=list)
    """Normalized import statements when the declaration is a module."""


@dataclass(frozen=True)
class ReferenceRecord:
    """Source evidence for a relationship and its target resolution."""

    source_identifier: str
    """The declaration or module containing the reference."""

    target_expression: str
    """The name or expression written in source."""

    relationship_kind: RelationshipKind
    """The relationship supported by the source."""

    location: SourceLocation
    """The reference's source extent."""

    target_identifier: str = ""
    """The resolved declaration or module identity, when available."""

    resolution: Resolution = Resolution.UNRESOLVED
    """Whether the target is exact, a candidate or unknown."""

    is_type_only: bool = False
    """Whether the reference occurs only within type-checking context."""

    target_source_path: str = ""
    """The repository-relative resource named by a document link."""

    target_fragment: str = ""
    """The optional resource fragment written in the document link."""


@dataclass(frozen=True)
class ImportBinding:
    """A lexical name's imported target or explicit local shadow."""

    local_name: str
    """The name introduced in its lexical scope."""

    target_name: str
    """The canonical imported name, or empty for a local shadow."""

    scope_identifier: str
    """The module or declaration owning the binding."""

    is_type_only: bool = False
    """Whether the binding is guarded by type checking."""


@dataclass(frozen=True)
class Diagnostic:
    """An inventory or extraction limitation attached to a source file."""

    relative_path: str
    """The source path affected by the limitation."""

    diagnostic_kind: DiagnosticKind
    """The reason the source was excluded or incompletely analysed."""

    explanation: str
    """A description of the limitation."""


@dataclass(frozen=True)
class IndexManifest:
    """The reproducible source identity and interpretation of an index."""

    repository_identifier: str
    """The repository history identity shared across checkouts."""

    commit_identifier: str
    """The committed revision supplying the snapshot's baseline."""

    tree_identifier: str
    """The Git tree containing the committed source."""

    snapshot_identifier: str
    """The digest binding source, interpretation and configuration."""

    source_digest: str
    """The digest of the sorted selected source identities."""

    python_version: str
    """The interpreter version used to interpret Python source."""

    source_count: int
    """The number of inventory records, including exclusions."""

    format_version: int = IndexVersion.SOURCE_ANALYSIS
    """The persisted source-graph format."""

    extractor_version: int = IndexVersion.SOURCE_ANALYSIS
    """The declaration and relationship extraction contract."""

    source_scope: SourceScope = SourceScope.COMMITTED
    """The committed or working-file state covered by this snapshot."""

    worktree_identifier: str = ""
    """The canonical checkout identity for a working-file snapshot."""

    index_digest: str = ""
    """The captured Git index identity."""

    checkout_commit: str = ""
    """The actual checkout HEAD captured with working files."""

    included_untracked_paths: tuple[str, ...] = ()
    """The explicit repository-relative untracked inclusion policy."""


@dataclass
class RepositorySnapshot:
    """A captured source inventory with explicit extraction exclusions."""

    manifest: IndexManifest
    """The source identity captured by this snapshot."""

    sources: list[SourceRecord]
    """The tracked and explicitly included source inventory."""

    diagnostics: list[Diagnostic]
    """Excluded, unsupported or undecodable inventory entries."""


@dataclass
class FileExtraction:
    """Declarations and lexical evidence extracted from one source file."""

    symbols: list[SymbolRecord] = field(default_factory=list)
    """The declarations belonging to the file."""

    references: list[ReferenceRecord] = field(default_factory=list)
    """The relationships written in the file."""

    bindings: list[ImportBinding] = field(default_factory=list)
    """Imported names and local shadows required for resolution."""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    """The file's analysis limitations."""


@dataclass(frozen=True)
class AnalysisIdentity:
    """The source contents and import context determining file analysis."""

    relative_path: str
    """The source path supplying the analysis."""

    content_digest: str
    """The exact source-byte identity."""

    module_name: str
    """The source's canonical import name."""

    is_package: bool
    """Whether the source declares a package initializer."""

    @classmethod
    def from_source(cls, source: SourceRecord) -> AnalysisIdentity:
        """Capture the source inputs that determine extraction."""
        return cls(
            source.relative_path,
            source.content_digest,
            source.module_name,
            source.is_package,
        )


@dataclass
class SourceAnalysis:
    """Unresolved structural evidence retained for source refresh."""

    identity: AnalysisIdentity
    """The source inputs determining this analysis."""

    extraction: FileExtraction
    """The declarations, lexical bindings and original references."""


@dataclass
class IndexData:
    """Structural source evidence for one captured snapshot."""

    manifest: IndexManifest
    """The captured source identity and interpretation."""

    sources: list[SourceRecord]
    """The source inventory and supported contents."""

    symbols: list[SymbolRecord]
    """The extracted declarations."""

    references: list[ReferenceRecord]
    """The resolved and unresolved relationships."""

    diagnostics: list[Diagnostic]
    """Visible inventory and extraction limitations."""

    analyses: list[SourceAnalysis] = field(default_factory=list)
    """Per-file evidence retained before global target resolution."""


@dataclass(frozen=True)
class SourceExcerpt:
    """An exact source slice returned as query evidence."""

    location: SourceLocation
    """The extent represented by the excerpt."""

    source_text: str
    """The source text in the selected range."""

    omitted_lines: int = 0
    """Declaration lines omitted after this bounded excerpt."""


@dataclass
class QueryPage:
    """Bounded evidence associated with a single source snapshot."""

    snapshot_identifier: str
    """The snapshot against which the query was evaluated."""

    symbols: list[SymbolRecord]
    """Matching source declarations."""

    references: list[ReferenceRecord]
    """Matching relationships and their resolution evidence."""

    excerpts: list[SourceExcerpt]
    """Source slices supporting the returned evidence."""

    diagnostics: list[Diagnostic]
    """Relevant source-analysis limitations."""

    total_matches: int
    """The number of matches before result bounds were applied."""

    next_offset: int | None
    """The next result offset, or none when the result is exhausted."""

    evidence_resolution: Resolution
    """Whether the requested evidence was located in this snapshot."""

    source_scope: SourceScope = SourceScope.COMMITTED
    """The committed source state covered by this result."""


@dataclass(frozen=True)
class FreshnessReport:
    """Compatibility between saved evidence and explicitly selected source."""

    snapshot_identifier: str
    """The indexed source snapshot."""

    indexed_commit: str
    """The exact commit captured by the index."""

    requested_commit: str
    """The exact commit selected by the consumer."""

    matches_requested_revision: bool
    """Whether both identities refer to the same committed revision."""

    source_scope: SourceScope = SourceScope.COMMITTED
    """The source state covered by this compatibility check."""

    matches_requested_source: bool = True
    """Whether the captured working-file identity still matches current source."""

    requested_snapshot_identifier: str = ""
    """The current working-file identity, when captured for comparison."""

    def require_current(self) -> None:
        """Reject evidence that differs from the requested source."""
        if not self.matches_requested_revision:
            raise SourceRevisionMismatch(self.indexed_commit, self.requested_commit)
        if not self.matches_requested_source:
            raise SourceSnapshotMismatch(
                self.snapshot_identifier, self.requested_snapshot_identifier
            )


@dataclass
class RefreshReport:
    """The source changes incorporated by a completed refresh."""

    manifest: IndexManifest
    """The refreshed snapshot identity."""

    analyzed_paths: list[str]
    """Sources requiring new parsing and extraction."""

    reused_paths: list[str]
    """Sources whose previous extraction remains applicable."""

    removed_paths: list[str]
    """Paths that no longer belong to the selected source inventory."""

    published: bool
    """Whether a replacement snapshot was published."""
