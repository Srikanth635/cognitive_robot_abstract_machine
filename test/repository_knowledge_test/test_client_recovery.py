"""
Agent transport preserves source refusal and recovery through real SDK calls.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from mcp import Client

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.composition import LocalApplication
from repository_knowledge.interfaces.agents import AgentService, AgentTool
from repository_knowledge.interfaces.cli import CommandParser
from repository_knowledge.interfaces.mcp import KnowledgeServer, ServerTool
from repository_knowledge.model import QueryPage
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject, FixturePath, FixtureSymbol

# %% corrupt-cache refusal


def test_mcp_corruption_returns_an_error_instead_of_freshness(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Invalid cache bytes cannot cross the SDK boundary as current evidence.
    """
    shutil.copyfile(
        Path(__file__).parent / "dataset/corrupt.sqlite3", indexed_storage.database_path
    )
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

    async def inspect_corruption() -> None:
        """
        Observe the transport error without accepting a source report.
        """

        async with Client(server) as client:
            result = await client.call_tool(
                ServerTool.QUERY, {"agent_request": {"tool": AgentTool.CHECK_FRESHNESS}}
            )
            assert result.is_error is True

    asyncio.run(inspect_corruption())


# %% stale-source recovery


def test_mcp_refuses_stale_source_then_reads_the_explicit_rebuild(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    A running server observes source changes and a completed replacement.
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
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / FixturePath.API,
    )
    committed_project.git("add", ".")
    committed_project.git("commit", "--quiet", "-m", "Replace transport source")

    async def inspect_recovery() -> None:
        """
        Reuse one connection across stale refusal and explicit publication.
        """

        async with Client(server) as client:
            request = {
                "agent_request": {
                    "tool": AgentTool.DESCRIBE_SYMBOL,
                    "symbol_identifier": FixtureSymbol.REPLACEMENT,
                }
            }
            refused = await client.call_tool(ServerTool.QUERY, request)
            assert refused.is_error is True
            rebuilt = IndexBuilder(
                GitSource(committed_project.root), PythonSource()
            ).build("HEAD")
            indexed_storage.replace(rebuilt)
            result = await client.call_tool(ServerTool.QUERY, request)
            assert result.is_error is False
            page = RecordCodec().decode(result.content[0].text, QueryPage)
            assert page.snapshot_identifier == rebuilt.manifest.snapshot_identifier
            assert page.symbols == [
                symbol
                for symbol in rebuilt.symbols
                if symbol.symbol_identifier == FixtureSymbol.REPLACEMENT
            ]

    asyncio.run(inspect_recovery())
