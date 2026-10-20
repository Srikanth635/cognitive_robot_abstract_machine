"""
Read-only clients preserve CLI results and reject mutation requests.
"""

from __future__ import annotations

import pytest

from repository_knowledge.composition import LocalApplication
from repository_knowledge.exceptions import InvalidQuery, SourceRevisionMismatch
from repository_knowledge.interfaces.agents import AgentService, AgentRequest, AgentTool
from repository_knowledge.interfaces.cli import Command, CommandParser
from repository_knowledge.adapters.sqlite_index import SQLiteIndex

from .support import CommittedProject, FixtureSymbol

# %% shared read contract


def test_agent_symbol_results_equal_cli_application(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    The agent adapter returns exactly the same evidence as CLI dispatch.
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
    application = LocalApplication()
    service = AgentService(application, arguments)
    assert service.execute(
        AgentRequest(AgentTool.FIND_SYMBOLS, query=FixtureSymbol.CHILD)
    ) == application.execute(arguments)


@pytest.mark.parametrize("command", [Command.INDEX, Command.REFRESH, Command.RECONCILE])
def test_agent_configuration_refuses_mutations(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex, command: Command
) -> None:
    """
    A mutation command cannot be smuggled through client configuration.
    """
    arguments = CommandParser().parse(
        ["--repository", str(committed_project.root), command]
    )
    with pytest.raises(InvalidQuery):
        AgentService(LocalApplication(), arguments)


@pytest.mark.parametrize(
    "agent_request",
    [
        AgentRequest(AgentTool.FIND_SYMBOLS, limit=101),
        AgentRequest(AgentTool.FIND_SYMBOLS, offset=-1),
        AgentRequest(AgentTool.FIND_SYMBOLS, query="x" * 4097),
        AgentRequest(AgentTool.FIND_RELATIONSHIPS, depth=9),
        AgentRequest(AgentTool.PREPARE_CONTEXT, character_budget=64001),
    ],
)
def test_agent_request_rejects_unbounded_work(agent_request: AgentRequest) -> None:
    """
    Tool requests cannot exceed the configured page, depth or text budgets.
    """
    with pytest.raises(InvalidQuery):
        agent_request.require_bounded()


def test_agent_rechecks_source_revision(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    A long-lived client refuses source changes after its cache was built.
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
    service = AgentService(LocalApplication(), arguments)
    committed_project.git("commit", "--quiet", "--allow-empty", "-m", "Advance fixture")
    with pytest.raises(SourceRevisionMismatch):
        service.execute(AgentRequest(AgentTool.FIND_SYMBOLS, query=FixtureSymbol.CHILD))
