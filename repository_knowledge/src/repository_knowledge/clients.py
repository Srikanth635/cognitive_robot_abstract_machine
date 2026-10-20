"""Source status reported to independently installed clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from repository_knowledge.publication import ConsumptionState, TrustIssue
from repository_knowledge.concepts import KnowledgeBundle

# %% versioned startup evidence


@dataclass(frozen=True)
class CanonicalStatus:
    """The canonical decision attached to a source-bound context response."""

    state: ConsumptionState
    """Whether authenticated wiki prose was admitted."""
    source_commit: str
    """The source selected by the client request."""
    issues: list[TrustIssue]
    """The trust conditions requiring source-only context."""


@dataclass(frozen=True)
class ContextKnowledge:
    """Curated context and its independently evaluated canonical status."""

    bundle: KnowledgeBundle
    """The permitted curated concepts, possibly empty after refusal."""
    canonical_status: CanonicalStatus | None = None
    """The trust verdict when canonical context was requested."""


class SessionState(StrEnum):
    """Whether startup may use the selected source index."""

    CURRENT = "current"
    STALE = "stale"
    MISSING = "missing"


@dataclass(frozen=True)
class SessionReport:
    """A small source-status result for clients outside the CRAM environment."""

    state: SessionState
    """The source-index availability and freshness verdict."""
    source_commit: str
    """The requested exact source commit, when available."""
    snapshot_identifier: str
    """The indexed source identity, when a cache exists."""
    wiki_entrypoint: str
    """The repository-relative entry point for local draft knowledge."""
    summary: str
    """The concise startup guidance for the agent."""
    protocol_version: int = 1
    """The process-report contract understood by clients."""
