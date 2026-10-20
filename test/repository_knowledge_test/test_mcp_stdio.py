"""
A real stdio client uses the configured cache without protocol noise.
"""

from __future__ import annotations

import asyncio
import sys

from mcp import Client, StdioServerParameters

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.interfaces.mcp import ServerTool
from repository_knowledge.model import FreshnessReport

from .support import CommittedProject

# %% process transport


def test_stdio_handshake_and_query_use_the_operator_selected_source(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    The installed SDK negotiates and queries the actual server subprocess.
    """
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "repository_knowledge.interfaces.mcp",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
        ],
    )

    async def inspect_source() -> None:
        """
        Exchange real MCP messages without a network service.
        """

        async with Client(parameters, read_timeout_seconds=15) as client:
            result = await client.call_tool(
                ServerTool.QUERY, {"agent_request": {"tool": "check-freshness"}}
            )
            assert result.is_error is False
            report = RecordCodec().decode(result.content[0].text, FreshnessReport)
            assert report.requested_commit == committed_project.commit_identifier
            assert report.matches_requested_revision is True

    asyncio.run(inspect_source())
