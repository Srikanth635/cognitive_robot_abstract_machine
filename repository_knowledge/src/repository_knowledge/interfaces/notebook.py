"""
Optional provider experiments using the real read-only repository MCP server.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

import litellm
from litellm import Harness, ToolLoopOptions, sandbox
from litellm.harness.types import ToolCall
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.assistance import TaskProfile
from repository_knowledge.exceptions import SourceSnapshotMismatch
from repository_knowledge.interfaces.agents import AgentRequest, AgentTool
from repository_knowledge.interfaces.cli import Option, RevisionSelector
from repository_knowledge.interfaces.exceptions import (
    InvalidModelSettings,
    LiveRequestsDisabled,
    NotebookQueryRefused,
    NotebookRepositoryMissing,
)
from repository_knowledge.interfaces.mcp import ServerTool
from repository_knowledge.model import FreshnessReport, QueryPage, SourceScope
from repository_knowledge.model_trials import ModelTrial, QueryObservation, QueryOrigin

# %% external request vocabulary


class NotebookArgument(StrEnum):
    """
    The fixed process and resource selections for notebook experiments.
    """

    MODULE = "-m"
    SERVER = "repository_knowledge.interfaces.mcp"
    SCRATCH_PREFIX = "cram-knowledge-trial-"
    PROMPT = "notebook.md"
    REDACTED = "[redacted]"
    REPOSITORY_MARKER = "repository_knowledge/pyproject.toml"


class CompletionOption(StrEnum):
    """
    SDK completion settings exposed as bounded notebook configuration.
    """

    OUTPUT_TOKENS = "max_tokens"
    TIMEOUT = "timeout"
    RETRIES = "num_retries"


@dataclass
class ModelSettings:
    """
    An explicit provider selection with a nonrepresentable credential.
    """

    provider: str
    """
    The LiteLLM provider identifier.
    """

    model: str
    """
    The model offered by the selected provider.
    """

    api_key: str | None = field(default=None, repr=False)
    """
    The runtime-only provider credential.
    """

    api_base: str | None = None
    """
    An optional compatible or local provider endpoint.
    """

    live_enabled: bool = False
    """
    Whether a live provider request was explicitly enabled.
    """

    maximum_turns: int = 6
    """
    The maximum number of model-selected tool attempts.
    """

    timeout_seconds: float = 300.0
    """
    The deadline for the model's complete turn.
    """

    model_timeout_seconds: float = 60.0
    """
    The deadline for one provider request.
    """

    maximum_output_tokens: int = 2048
    """
    The output allowance for each model request.
    """

    def select_model(self) -> str:
        """
        Require a complete provider selection and positive request budgets.
        """
        if (
            not self.provider.strip()
            or not self.model.strip()
            or min(
                self.maximum_turns,
                self.timeout_seconds,
                self.model_timeout_seconds,
                self.maximum_output_tokens,
            )
            <= 0
        ):
            raise InvalidModelSettings()
        if self.model.startswith(self.provider + "/"):
            return self.model
        return f"{self.provider}/{self.model}"


# %% read-only MCP queries


@dataclass
class NotebookTools:
    """
    Bound model-selected queries to an operator-selected MCP connection.
    """

    client: Client
    """
    The connected repository server.
    """

    observations: list[QueryObservation] = field(default_factory=list)
    """
    Repository call receipts without full source payloads.
    """

    async def execute(
        self, request: AgentRequest, origin: QueryOrigin = QueryOrigin.MODEL
    ) -> str:
        """
        Return one complete encoded repository result or raise a visible refusal.
        """
        request.require_bounded()
        started = time.perf_counter()
        result = await self.client.call_tool(
            ServerTool.QUERY, {"agent_request": asdict(request)}
        )
        text = (
            result.content[0].text
            if len(result.content) == 1 and isinstance(result.content[0], TextContent)
            else ""
        )
        self.observations.append(
            QueryObservation(
                request.tool.value,
                origin,
                request.query,
                request.symbol_identifier,
                time.perf_counter() - started,
                len(text),
                result.is_error,
            )
        )
        if result.is_error or not text:
            raise NotebookQueryRefused(request.tool.value)
        return text

    async def query_repository(
        self,
        tool: AgentTool,
        query: str = "",
        symbol_identifier: str = "",
        limit: int = 5,
        offset: int = 0,
        depth: int = 1,
        character_budget: int = 16000,
        profile: TaskProfile = TaskProfile.CODE_EXTENSION,
    ) -> str:
        """
        Find source symbols, references, relationships, examples or tests; prepare
        bounded context or inspect freshness.

        Change assessment needs a baseline.
        """
        return await self.execute(
            AgentRequest(
                tool,
                query=query,
                symbol_identifier=symbol_identifier,
                limit=limit,
                offset=offset,
                depth=depth,
                character_budget=character_budget,
                profile=profile,
            )
        )


@dataclass
class NotebookInspection:
    """
    A credential-free inspection of source identity and a selected declaration.
    """

    freshness: FreshnessReport
    """
    The checked source identity.
    """

    available_tools: list[str]
    """
    The MCP server's advertised tool names.
    """

    definition: QueryPage
    """
    The selected declaration and its source evidence.
    """


@dataclass
class NotebookConnection:
    """
    The source selection fixed by the notebook operator.
    """

    repository_root: Path
    """
    The checkout supplying evidence.
    """

    database_path: Path
    """
    The source cache selected for this experiment.
    """

    source_scope: SourceScope = SourceScope.COMMITTED
    """
    Whether evidence includes selected working files.
    """

    revision: str = RevisionSelector.CHECKOUT
    """
    The requested committed source selector.
    """

    included_untracked_paths: tuple[str, ...] = ()
    """
    Explicit untracked paths in a working overlay.
    """

    query_timeout_seconds: float = 300.0
    """
    The maximum wait for a repository tool response.
    """

    @staticmethod
    def find_repository(directory: Path) -> Path:
        """
        Locate the nearest checkout containing the repository-knowledge package.
        """
        for candidate in [directory.resolve(), *directory.resolve().parents]:
            if (candidate / NotebookArgument.REPOSITORY_MARKER).is_file():
                return candidate
        raise NotebookRepositoryMissing(directory)

    def create_client(self) -> Client:
        """
        Create a stdio client with fixed source selectors and no provider key.
        """
        arguments = [
            NotebookArgument.MODULE,
            NotebookArgument.SERVER,
            Option.REPOSITORY,
            str(self.repository_root.resolve()),
            Option.DATABASE,
            str(self.database_path.resolve()),
            Option.SCOPE,
            self.source_scope,
            Option.REVISION,
            self.revision,
        ]
        for relative_path in self.included_untracked_paths:
            arguments.extend([Option.INCLUDE_UNTRACKED, relative_path])
        return Client(
            stdio_client(
                StdioServerParameters(command=sys.executable, args=arguments),
                errlog=sys.__stderr__,
            ),
            read_timeout_seconds=self.query_timeout_seconds,
        )

    async def check_source(self, client: Client) -> FreshnessReport:
        """
        Read source freshness through the connected MCP server.
        """
        encoded = await NotebookTools(client).execute(
            AgentRequest(AgentTool.CHECK_FRESHNESS), QueryOrigin.PREFLIGHT
        )
        return RecordCodec().decode(encoded, FreshnessReport)

    async def inspect(self, symbol_identifier: str) -> NotebookInspection:
        """
        Inspect real MCP discovery, source freshness and a declaration without an LLM.
        """

        async with self.create_client() as client:
            freshness = await self.check_source(client)
        freshness.require_current()
        async with self.create_client() as client:
            tools = (await client.list_tools()).tools
            encoded = await NotebookTools(client).execute(
                AgentRequest(
                    AgentTool.DESCRIBE_SYMBOL, symbol_identifier=symbol_identifier
                )
            )
        definition = RecordCodec().decode(encoded, QueryPage)
        if definition.snapshot_identifier != freshness.snapshot_identifier:
            raise SourceSnapshotMismatch(
                freshness.snapshot_identifier, definition.snapshot_identifier
            )
        return NotebookInspection(freshness, [tool.name for tool in tools], definition)


# %% provider-backed model trials


@dataclass
class NotebookSession:
    """
    Run a bounded model turn with repository queries and verified source identity.
    """

    connection: NotebookConnection
    """
    The source configuration unavailable to model-selected arguments.
    """

    async def run(self, question: str, settings: ModelSettings) -> ModelTrial:
        """
        Return an explicitly enabled model trial only while its source remains valid.
        """
        if not settings.live_enabled:
            raise LiveRequestsDisabled()
        model = settings.select_model()
        started = time.perf_counter()
        async with self.connection.create_client() as client:
            tools = NotebookTools(client)
            encoded = await tools.execute(
                AgentRequest(AgentTool.CHECK_FRESHNESS), QueryOrigin.PREFLIGHT
            )
        freshness = RecordCodec().decode(encoded, FreshnessReport)
        freshness.require_current()
        async with self.connection.create_client() as client:
            tools = NotebookTools(client, tools.observations)
            with TemporaryDirectory(
                prefix=NotebookArgument.SCRATCH_PREFIX
            ) as scratch_directory:
                model_result = await litellm.aagent(
                    Harness.TOOL_LOOP,
                    question,
                    sandbox=sandbox.local(scratch_directory),
                    model=model,
                    api_key=settings.api_key,
                    api_base=settings.api_base,
                    instructions=(
                        Path(__file__).parents[1]
                        / "adapters/prompts"
                        / NotebookArgument.PROMPT
                    ).read_text(encoding="utf-8"),
                    tools=[tools.query_repository],
                    max_turns=settings.maximum_turns,
                    timeout=settings.timeout_seconds,
                    options=ToolLoopOptions(
                        completion_kwargs={
                            CompletionOption.OUTPUT_TOKENS: settings.maximum_output_tokens,
                            CompletionOption.TIMEOUT: settings.model_timeout_seconds,
                            CompletionOption.RETRIES: 0,
                        }
                    ),
                )
            after = await self.connection.check_source(client)
        after.require_current()
        if after.snapshot_identifier != freshness.snapshot_identifier:
            raise SourceSnapshotMismatch(
                freshness.snapshot_identifier, after.snapshot_identifier
            )
        answer = (
            model_result.text.replace(settings.api_key, NotebookArgument.REDACTED)
            if settings.api_key
            else model_result.text
        )
        return ModelTrial(
            model,
            answer,
            model_result.stop_reason,
            freshness.indexed_commit,
            freshness.snapshot_identifier,
            freshness.source_scope,
            time.perf_counter() - started,
            model_result.usage.input_tokens or None,
            model_result.usage.output_tokens or None,
            model_result.usage.calls or None,
            sum(isinstance(event, ToolCall) for event in model_result.events),
            model_result.cost if model_result.cost > 0 else None,
            tools.observations,
        )
