"""Bounded read-only requests shared by agent transports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import StrEnum

from repository_knowledge.assistance import (
    ChangeReport,
    DiscoveryReport,
    TaskContext,
    TaskProfile,
)
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.interfaces.cli import Command, CommandArguments
from repository_knowledge.model import (
    FreshnessReport,
    IndexManifest,
    QueryPage,
    RefreshReport,
    TraversalDirection,
)

# %% transport-independent application contract

type ApplicationResult = IndexManifest | QueryPage | FreshnessReport | RefreshReport | DiscoveryReport | TaskContext | ChangeReport


@dataclass
class RequestApplication(ABC):
    """Execute a source request through the existing operation contracts."""

    @abstractmethod
    def execute(self, arguments: CommandArguments) -> ApplicationResult:
        """Return evidence for the explicitly selected source and operation."""


class AgentTool(StrEnum):
    """The read-only operations available to an agent."""

    FIND_SYMBOLS = Command.FIND_SYMBOLS
    DESCRIBE_SYMBOL = Command.DESCRIBE_SYMBOL
    FIND_REFERENCES = Command.FIND_REFERENCES
    FIND_RELATIONSHIPS = Command.FIND_RELATIONSHIPS
    CHECK_FRESHNESS = Command.CHECK_FRESHNESS
    FIND_EXAMPLES = Command.FIND_EXAMPLES
    FIND_TESTS = Command.FIND_TESTS
    PREPARE_CONTEXT = Command.PREPARE_CONTEXT
    ASSESS_CHANGE = Command.ASSESS_CHANGE


@dataclass(frozen=True)
class AgentRequest:
    """A bounded query without filesystem or executable selectors."""

    tool: AgentTool
    """The read-only operation to perform."""
    query: str = ""
    """The lexical symbol search text."""
    symbol_identifier: str = ""
    """The qualified declaration being inspected."""
    limit: int = 20
    """The requested evidence page size."""
    offset: int = 0
    """The evidence page's initial match."""
    depth: int = 1
    """The relationship traversal distance."""
    direction: TraversalDirection = TraversalDirection.OUTGOING
    """The relationship traversal direction."""
    profile: TaskProfile = TaskProfile.CODE_EXTENSION
    """The purpose controlling context ranking."""
    character_budget: int = 16000
    """The requested source-context character allowance."""
    orm_task: bool = False
    """Whether ORM generator evidence should be included."""

    def require_bounded(
        self,
        maximum_results: int = 100,
        maximum_depth: int = 8,
        maximum_text: int = 4096,
        maximum_characters: int = 64000,
    ) -> None:
        """Refuse requests beyond the transport's work and output allowances."""
        if (
            self.tool not in AgentTool
            or not 1 <= self.limit <= maximum_results
            or self.offset < 0
            or not 1 <= self.depth <= maximum_depth
            or len(self.query) > maximum_text
            or len(self.symbol_identifier) > maximum_text
            or not 1 <= self.character_budget <= maximum_characters
        ):
            raise InvalidQuery("Agent request exceeds the read-only query bounds.")


@dataclass
class AgentService:
    """Apply agent queries to an operator-selected source configuration."""

    application: RequestApplication
    """The shared CLI operation dispatch."""
    configuration: CommandArguments
    """The source, cache and optional baseline selected by the operator."""

    def __post_init__(self) -> None:
        """Reject mutation operations at the client boundary."""
        if self.configuration.command not in AgentTool:
            raise InvalidQuery("Agent configuration must select a read-only operation.")

    def execute(self, agent_request: AgentRequest) -> ApplicationResult:
        """Return the same domain evidence and freshness failures as the CLI."""
        agent_request.require_bounded()
        arguments = replace(
            self.configuration,
            command=Command(agent_request.tool),
            query=agent_request.query,
            symbol_identifier=agent_request.symbol_identifier,
            limit=agent_request.limit,
            offset=agent_request.offset,
            depth=agent_request.depth,
            direction=agent_request.direction,
            profile=agent_request.profile,
            character_budget=agent_request.character_budget,
            orm_task=agent_request.orm_task,
        )
        return self.application.execute(arguments)
