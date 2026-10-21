"""
Planning context preserves source evidence and the shared query boundaries.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from litellm.types.utils import ModelResponse

from repository_knowledge.adapters.plan_sources import CoraplexPlanningSources
from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.assistance import (
    ContextReference,
    ContextRole,
    OmissionReason,
    PlanningLimitation,
    TaskContext,
    TaskProfile,
    TestCatalog as StaticTestCatalog,
)
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.interfaces.agents import AgentTool
from repository_knowledge.interfaces.notebook import (
    ModelSettings,
    NotebookConnection,
    NotebookSession,
    NotebookTools,
)
from repository_knowledge.model import IndexData, Resolution
from repository_knowledge.operations.assistance import TaskAssistance

from .dataset.scripted_completion import ScriptedCompletion
from .support import CommittedProject, FixtureSymbol

# %% supplementary source evidence


def test_planning_reference_keeps_exact_source_and_candidate_relevance(
    source_index: IndexData,
) -> None:
    """
    A selected reference is cited exactly without inventing a call relationship.
    """
    reference = ContextReference(ContextRole.INTERFACE, FixtureSymbol.HELPER)
    service = TaskAssistance(
        source_index, StaticTestCatalog(), context_references=[reference]
    )
    context = service.prepare_context(
        FixtureSymbol.CHILD, KnowledgeBundle([]), TaskProfile.PLAN_GENERATION, 64000
    )
    item = next(
        item
        for item in context.items
        if item.role == reference.role and item.label == reference.symbol_identifier
    )
    declaration = next(
        symbol
        for symbol in source_index.symbols
        if symbol.symbol_identifier == reference.symbol_identifier
    )
    source = next(
        source
        for source in source_index.sources
        if source.relative_path == declaration.location.relative_path
    )
    assert item.text == "\n".join(
        source.source_text.splitlines()[
            declaration.location.start_line - 1 : declaration.location.end_line
        ]
    )
    assert item.citations[0].location == declaration.location
    assert (
        item.citations[0].snapshot_identifier
        == source_index.manifest.snapshot_identifier
    )
    assert item.citations[0].source_commit == source_index.manifest.commit_identifier
    assert item.citations[0].content_digest == source.content_digest
    assert item.resolution == Resolution.CANDIDATE


def test_absent_planning_reference_is_reported(source_index: IndexData) -> None:
    """
    Source drift cannot silently remove a configured planning contract.
    """
    reference = ContextReference(
        ContextRole.INTERFACE, FixtureSymbol.HELPER + ".absent"
    )
    context = TaskAssistance(
        source_index, StaticTestCatalog(), context_references=[reference]
    ).prepare_context(
        FixtureSymbol.CHILD, KnowledgeBundle([]), TaskProfile.PLAN_GENERATION, 64000
    )
    assert [
        (item.role, item.reason)
        for item in context.omissions
        if item.label == reference.symbol_identifier
    ] == [(reference.role, OmissionReason.UNAVAILABLE)]


def test_ambiguous_planning_reference_is_omitted(source_index: IndexData) -> None:
    """
    A colliding qualified identity cannot be selected as unique evidence.
    """
    declaration = next(
        item
        for item in source_index.symbols
        if item.symbol_identifier == FixtureSymbol.HELPER
    )
    index = replace(source_index, symbols=[*source_index.symbols, declaration])
    reference = ContextReference(ContextRole.INTERFACE, declaration.symbol_identifier)
    context = TaskAssistance(
        index, StaticTestCatalog(), context_references=[reference]
    ).prepare_context(
        FixtureSymbol.CHILD, KnowledgeBundle([]), TaskProfile.PLAN_GENERATION, 64000
    )
    assert [
        (item.role, item.reason)
        for item in context.omissions
        if item.label == reference.symbol_identifier
    ] == [(reference.role, OmissionReason.UNAVAILABLE)]


def test_supplementary_evidence_obeys_the_shared_budget(
    source_index: IndexData,
) -> None:
    """
    Planning references cannot exceed the caller's total context allowance.
    """
    reference = ContextReference(ContextRole.INTERFACE, FixtureSymbol.HELPER)
    context = TaskAssistance(
        source_index, StaticTestCatalog(), context_references=[reference]
    ).prepare_context(
        FixtureSymbol.CHILD, KnowledgeBundle([]), TaskProfile.PLAN_GENERATION, 1
    )
    assert context.items == []
    assert context.used_characters == 0
    assert [
        item.reason
        for item in context.omissions
        if item.label == reference.symbol_identifier
    ] == [OmissionReason.BUDGET]


def test_resolved_interface_is_not_downgraded_by_profile_selection(
    source_index: IndexData,
) -> None:
    """
    An inherited contract retains its established relationship to the target.
    """
    reference = ContextReference(ContextRole.INTERFACE, FixtureSymbol.BASE)
    context = TaskAssistance(
        source_index, StaticTestCatalog(), context_references=[reference, reference]
    ).prepare_context(
        FixtureSymbol.CHILD, KnowledgeBundle([]), TaskProfile.PLAN_GENERATION, 64000
    )
    assert [
        item.resolution
        for item in context.items
        if item.role == reference.role and item.label == reference.symbol_identifier
    ] == [Resolution.RESOLVED]


def test_planning_references_extend_instruction_scope(
    assistance_index: IndexData,
) -> None:
    """
    An additional test source brings its own scoped instructions.
    """
    catalog = PytestSource().discover(assistance_index)
    test = next(item for item in catalog.tests if item.symbol_name == "test_direct")
    reference = ContextReference(ContextRole.TEST, test.symbol_identifier)
    context = TaskAssistance(
        assistance_index, catalog, context_references=[reference]
    ).prepare_context(
        FixtureSymbol.CHILD, KnowledgeBundle([]), TaskProfile.PLAN_GENERATION, 64000
    )
    instruction = next(
        item
        for item in context.items
        if item.role == ContextRole.INSTRUCTION and item.label == "tests/AGENTS.md"
    )
    assert test.location.relative_path in instruction.applies_to


# %% shared executable interfaces


def test_cli_selects_planning_sources_and_preserves_static_limits(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    The profile reaches normal CLI composition and reports unavailable anchors.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "prepare-context",
            FixtureSymbol.CHILD,
            "--profile",
            TaskProfile.PLAN_GENERATION,
            "--characters",
            "64000",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    context = RecordCodec().decode(result.stdout, TaskContext)
    assert context.profile == TaskProfile.PLAN_GENERATION
    assert context.manifest.commit_identifier == committed_project.commit_identifier
    assert set(PlanningLimitation).issubset(context.limitations)
    reference = CoraplexPlanningSources().select()[0]
    assert [
        (item.role, item.reason)
        for item in context.omissions
        if item.label == reference.symbol_identifier
    ] == [(reference.role, OmissionReason.UNAVAILABLE)]


def test_notebook_requests_planning_context_over_real_mcp(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    The model-facing schema can select the same planning profile without a provider.
    """

    async def query() -> TaskContext:
        """
        Read a bounded planning result over the actual stdio transport.
        """

        async with NotebookConnection(
            committed_project.root, indexed_storage.database_path
        ).create_client() as client:
            encoded = await NotebookTools(client).query_repository(
                AgentTool.PREPARE_CONTEXT,
                symbol_identifier=FixtureSymbol.CHILD,
                character_budget=64000,
                profile=TaskProfile.PLAN_GENERATION,
            )
        return RecordCodec().decode(encoded, TaskContext)

    context = asyncio.run(query())
    assert context.profile == TaskProfile.PLAN_GENERATION
    assert context.manifest.commit_identifier == committed_project.commit_identifier
    assert set(PlanningLimitation).issubset(context.limitations)


def test_model_selects_planning_profile_through_the_sdk(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A provider tool call can request planning evidence through the notebook schema.
    """
    payload = json.loads(
        (Path(__file__).parent / "dataset/planning_provider_responses.json").read_text(
            encoding="utf-8"
        )
    )
    responses = [ModelResponse.model_validate(item) for item in payload]
    provider = ScriptedCompletion(responses)
    monkeypatch.setattr("litellm.acompletion", provider)
    trial = asyncio.run(
        NotebookSession(
            NotebookConnection(committed_project.root, indexed_storage.database_path)
        ).run(
            "Prepare planning evidence.",
            ModelSettings("openai", "offline-test", live_enabled=True),
        )
    )
    message = next(
        message
        for message in provider.requests[-1]["messages"]
        if message["role"] == "tool"
    )
    context = RecordCodec().decode(message["content"], TaskContext)
    assert context.profile == TaskProfile.PLAN_GENERATION
    assert context.manifest.commit_identifier == committed_project.commit_identifier
    assert trial.observations[-1].operation == AgentTool.PREPARE_CONTEXT
    assert trial.observations[-1].is_error is False
