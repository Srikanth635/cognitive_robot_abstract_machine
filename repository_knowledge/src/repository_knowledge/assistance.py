"""Source-bound task context and static test evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from repository_knowledge.clients import CanonicalStatus

from repository_knowledge.model import (
    IndexManifest,
    ReferenceRecord,
    Resolution,
    SourceLocation,
    SourceScope,
    SymbolRecord,
)

# %% task vocabulary


class TaskProfile(StrEnum):
    """The purpose determining context priority."""

    NAVIGATION = "navigation"
    CODE_EXTENSION = "code-extension"
    DEBUGGING = "debugging"
    REVIEW = "review"
    MIGRATION = "migration"


class ContextRole(StrEnum):
    """The subject of an evidence item."""

    DECLARATION = "declaration"
    INTERFACE = "interface"
    EXAMPLE = "example"
    TEST = "test"
    FIXTURE = "fixture"
    INSTRUCTION = "instruction"
    GUIDANCE = "guidance"
    CONCEPT = "concept"
    GENERATOR = "generator"
    CONSUMER = "consumer"


class OmissionReason(StrEnum):
    """Why evidence was not included in the context payload."""

    BUDGET = "character-budget"
    UNAVAILABLE = "unavailable"
    INVALID_KNOWLEDGE = "knowledge-not-current-or-policy-invalid"


class ChangeKind(StrEnum):
    """The observed difference between two declaration contracts."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


# %% test declarations


@dataclass(frozen=True)
class TestRecord:
    """A statically recognized test and its requested fixture names."""

    symbol_identifier: str
    """The indexed test declaration."""
    symbol_name: str
    """The source function name."""
    location: SourceLocation
    """The test declaration extent."""
    fixture_names: list[str]
    """Fixture requests excluding direct parametrization arguments."""
    uncertainties: list[str] = field(default_factory=list)
    """Runtime behavior not established by static discovery."""
    resolution: Resolution = Resolution.RESOLVED
    """Whether relevance to a requested target is exact or only a candidate."""


@dataclass(frozen=True)
class FixtureRecord:
    """A fixture declaration with visible scope and dependencies."""

    symbol_identifier: str
    """The indexed fixture declaration."""
    fixture_name: str
    """The name registered with pytest."""
    scope: str
    """The declared lifetime, or unknown for a dynamic expression."""
    location: SourceLocation
    """The fixture declaration extent."""
    dependencies: list[str]
    """The fixture names requested by its parameters."""
    autouse: bool = False
    """Whether automatic activation is statically declared."""
    owner_identifier: str = ""
    """The lexical class restricting fixture visibility, when present."""
    uncertainties: list[str] = field(default_factory=list)
    """Fixture declarations requiring runtime interpretation."""


@dataclass
class TestCatalog:
    """Static declarations without a claim of pytest collection completeness."""

    tests: list[TestRecord] = field(default_factory=list)
    """Recognized test functions and methods."""
    fixtures: list[FixtureRecord] = field(default_factory=list)
    """Recognized pytest fixture declarations."""


@dataclass
class DiscoveryReport:
    """Paged examples or tests with explicit static-analysis limits."""

    snapshot_identifier: str
    """The source snapshot supplying all records."""
    source_scope: SourceScope
    """Whether evidence represents committed or working files."""
    references: list[ReferenceRecord] = field(default_factory=list)
    """Production example references in deterministic order."""
    tests: list[TestRecord] = field(default_factory=list)
    """Relevant tests on this page."""
    fixtures: list[FixtureRecord] = field(default_factory=list)
    """The selected tests' visible fixture closure."""
    unresolved_dependencies: list[str] = field(default_factory=list)
    """Fixture requests with no unique statically visible declaration."""
    suggested_test_directories: list[str] = field(default_factory=list)
    """Directories containing the selected tests, not a complete test plan."""
    total_matches: int = 0
    """The match count before pagination."""
    next_offset: int | None = None
    """The next page offset, if additional matches exist."""
    complete_coverage: bool = False
    """Static suggestions never establish complete behavioral coverage."""
    limitations: list[str] = field(default_factory=list)
    """Restrictions affecting interpretation of this evidence."""


# %% context evidence


@dataclass(frozen=True)
class SourceCitation:
    """Bind a line range to an exact captured source identity."""

    snapshot_identifier: str
    """The captured source graph identity."""
    source_commit: str
    """The committed baseline of the captured source."""
    location: SourceLocation
    """The inclusive source extent supporting the item."""
    content_digest: str
    """The original captured file-byte digest."""


@dataclass(frozen=True)
class ContextItem:
    """An indivisible context payload and its supporting source citations."""

    role: ContextRole
    """The evidence category."""
    label: str
    """The declaration, source path or concept path being described."""
    text: str
    """The payload counted against the character budget."""
    citations: list[SourceCitation]
    """Exact source extents supporting this item."""
    resolution: Resolution = Resolution.RESOLVED
    """Whether the item's relationship to the target is resolved or a candidate."""
    lifecycle: str = ""
    """The producer-declared concept lifecycle, without review authority."""
    omitted_lines: int = 0
    """The trailing declaration lines omitted from this bounded excerpt."""
    applies_to: list[str] = field(default_factory=list)
    """Evidence paths governed by a retrieved instruction document."""


@dataclass(frozen=True)
class ContextOmission:
    """An evidence item or category absent from the payload."""

    role: ContextRole
    """The affected evidence category."""
    label: str
    """The item or category that was omitted."""
    reason: OmissionReason
    """The explicit cause of the omission."""
    characters: int = 0
    """The payload size of an item omitted for budget reasons."""


@dataclass
class TaskContext:
    """A deterministic, character-bounded task evidence package."""

    manifest: IndexManifest
    """The source identity covering every citation."""
    profile: TaskProfile
    """The requested task purpose."""
    symbol_identifier: str
    """The exact declaration selected by the request."""
    items: list[ContextItem]
    """Whole evidence items in profile priority order."""
    omissions: list[ContextOmission]
    """Unavailable, invalid or over-budget evidence."""
    character_budget: int
    """The maximum sum of payload text characters, excluding JSON metadata."""
    used_characters: int
    """The sum of included payload text characters."""
    suggested_test_directories: list[str]
    """Proposed checks without a claim of complete coverage."""
    limitations: list[str]
    """Static-analysis and authority restrictions."""
    canonical_status: CanonicalStatus | None = None
    """The authenticated publication decision, when explicitly requested."""


# %% change evidence


@dataclass(frozen=True)
class ContractChange:
    """An observed public declaration difference, not a compatibility verdict."""

    symbol_identifier: str
    """The qualified declaration compared across snapshots."""
    change_kind: ChangeKind
    """Whether its public declaration contract differs."""
    before: SymbolRecord | None
    """The baseline declaration, when present."""
    after: SymbolRecord | None
    """The requested declaration, when present."""


@dataclass
class ConsumerCount:
    """The complete static consumer counts before pagination."""

    resolved: int
    """The exact incoming reference count."""
    uncertain: int
    """The possible incoming name-match count."""


@dataclass
class ChangeReport:
    """Contract differences and consumers belonging to two explicit snapshots."""

    before_manifest: IndexManifest
    """The baseline identity covering baseline consumers and contracts."""
    after_manifest: IndexManifest
    """The requested identity covering current consumers and contracts."""
    contracts: list[ContractChange]
    """The selected declaration and its public member comparisons."""
    resolved_consumers: list[ReferenceRecord]
    """Exact incoming references in the baseline snapshot."""
    uncertain_consumers: list[ReferenceRecord]
    """Potential incoming references in the baseline snapshot."""
    current_consumers: list[ReferenceRecord]
    """Exact incoming references in the requested snapshot."""
    current_uncertain_consumers: list[ReferenceRecord]
    """Potential incoming references in the requested snapshot."""
    suggested_test_directories: list[str]
    """Directories suggested by test discovery in either snapshot."""
    limitations: list[str]
    """Limits on compatibility and dependency inference."""
    before_consumer_count: ConsumerCount
    """The baseline totals before applying the consumer page."""
    after_consumer_count: ConsumerCount
    """The requested snapshot totals before applying the consumer page."""
    consumer_limit: int
    """The maximum references in each of the four consumer lists."""
    consumer_offset: int
    """The common starting offset into each consumer list."""
    next_offset: int | None
    """The next common page offset when any consumer list has more records."""
