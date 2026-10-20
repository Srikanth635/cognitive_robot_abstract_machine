"""
The optional SDK exposes bounded queries with the CLI evidence contract.
"""

from __future__ import annotations

import asyncio

from mcp import Client

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.composition import LocalApplication
from repository_knowledge.interfaces.agents import AgentService
from repository_knowledge.interfaces.cli import CommandParser
from repository_knowledge.interfaces.mcp import KnowledgeServer, ServerTool
from repository_knowledge.model import QueryPage

from .support import CommittedProject, FixtureSymbol

# %% real SDK exchange


def test_mcp_discovery_contains_only_a_read_only_bounded_query(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Tool discovery exposes no source mutation, path or execution controls.
    """
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "check-freshness",
        ]
    )
    server = KnowledgeServer(AgentService(LocalApplication(), arguments)).create()

    async def inspect_tools() -> None:
        """
        Inspect the SDK's advertised tool contract.
        """

        async with Client(server) as client:
            tools = (await client.list_tools()).tools
            assert [tool.name for tool in tools] == [ServerTool.QUERY]
            assert tools[0].annotations.read_only_hint is True
            schema = tools[0].input_schema
            assert set(schema["properties"]) == {"agent_request"}
            assert "repository_root" not in str(schema)
            assert "database_path" not in str(schema)

    asyncio.run(inspect_tools())


def test_mcp_query_returns_cli_record_and_invalid_requests_are_tool_errors(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    The real SDK executes source queries and refuses mutation selectors.
    """
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "find-symbols",
            FixtureSymbol.CHILD,
        ]
    )
    server = KnowledgeServer(AgentService(LocalApplication(), arguments)).create()
    expected = LocalApplication().execute(arguments)

    async def query_tools() -> None:
        """
        Call the SDK without a network service or credentials.
        """

        async with Client(server) as client:
            result = await client.call_tool(
                ServerTool.QUERY,
                {
                    "agent_request": {
                        "tool": "find-symbols",
                        "query": FixtureSymbol.CHILD,
                    }
                },
            )
            assert result.is_error is False
            assert RecordCodec().decode(result.content[0].text, QueryPage) == expected
            refused = await client.call_tool(
                ServerTool.QUERY, {"agent_request": {"tool": "index"}}
            )
            assert refused.is_error is True
            unbounded = await client.call_tool(
                ServerTool.QUERY,
                {"agent_request": {"tool": "find-symbols", "limit": 101}},
            )
            assert unbounded.is_error is True

    asyncio.run(query_tools())


def test_mcp_does_not_return_an_oversized_encoded_response(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    A wire budget rejects large results without truncating their evidence.
    """
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "check-freshness",
        ]
    )
    server = KnowledgeServer(
        AgentService(LocalApplication(), arguments), maximum_response_characters=1
    ).create()

    async def inspect_budget() -> None:
        """
        Observe the SDK's explicit tool error for over-budget output.
        """

        async with Client(server) as client:
            result = await client.call_tool(
                ServerTool.QUERY, {"agent_request": {"tool": "check-freshness"}}
            )
            assert result.is_error is True

    asyncio.run(inspect_budget())
