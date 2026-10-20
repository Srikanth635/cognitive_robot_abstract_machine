"""
Task assistance retains source identity and bounded uncertainty.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.assistance import (
    ChangeKind,
    ContextRole,
    TaskContext,
    TaskProfile,
)
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.model import (
    IndexData,
    IndexSelection,
    SelectionKind,
    Resolution,
)
from repository_knowledge.operations.assistance import TaskAssistance
from repository_knowledge.operations.changes import ChangeAssessment

from .support import FixtureSymbol

# %% static examples and tests


def test_examples_prefer_resolved_production_references(
    assistance_index: IndexData,
) -> None:
    """
    Test calls do not replace a source-backed implementation example.
    """
    report = TaskAssistance(
        assistance_index, PytestSource().discover(assistance_index)
    ).find_examples(FixtureSymbol.HELPER)
    assert FixtureSymbol.CALLER in [
        item.source_identifier for item in report.references
    ]
    assert all(
        not item.location.relative_path.startswith("tests/")
        for item in report.references
    )
    assert all(item.resolution == Resolution.RESOLVED for item in report.references)


def test_tests_include_fixture_consumers_and_nearest_override(
    assistance_index: IndexData,
) -> None:
    """
    An inherited autouse fixture resolves its dependency in the test scope.
    """
    report = TaskAssistance(
        assistance_index, PytestSource().discover(assistance_index)
    ).find_tests(FixtureSymbol.HELPER)
    assert [item.symbol_name for item in report.tests] == [
        "test_direct",
        "test_dynamic",
        "test_fixture",
        "test_indirect",
        "test_parameter",
    ]
    assert [
        (item.fixture_name, item.scope, item.location.relative_path)
        for item in report.fixtures
    ] == [
        ("environment", "function", "conftest.py"),
        ("seed", "module", "tests/conftest.py"),
    ]
    parameter = next(
        item for item in report.tests if item.symbol_name == "test_parameter"
    )
    assert "value" not in parameter.fixture_names
    assert report.complete_coverage is False
    assert report.suggested_test_directories == ["tests"]


def test_dynamic_and_indirect_tests_keep_uncertainty(
    assistance_index: IndexData,
) -> None:
    """
    Static discovery cannot claim runtime fixture selection is complete.
    """
    catalog = PytestSource().discover(assistance_index)
    indirect = next(
        item for item in catalog.tests if item.symbol_name == "test_indirect"
    )
    dynamic = next(item for item in catalog.tests if item.symbol_name == "test_dynamic")
    assert indirect.uncertainties
    assert dynamic.uncertainties


def test_context_projection_omits_reusable_raw_analysis(
    assistance_index: IndexData, tmp_path: Path
) -> None:
    """
    Context reads one complete resolved graph without decompressing raw analyses.
    """
    storage = SQLiteIndex(tmp_path / "context.sqlite3")
    storage.replace(assistance_index)
    projection = storage.read(IndexSelection(SelectionKind.TASK_CONTEXT))
    assert projection == replace(assistance_index, analyses=[])


# %% deterministic context budgets


def test_context_citations_bind_every_source_item(assistance_index: IndexData) -> None:
    """
    Generation context cites definitions, examples, instructions and tests.
    """
    service = TaskAssistance(
        assistance_index, PytestSource().discover(assistance_index)
    )
    context = service.prepare_context(
        FixtureSymbol.HELPER, KnowledgeBundle([]), TaskProfile.CODE_EXTENSION, 30000
    )
    roles = {item.role for item in context.items}
    assert {
        ContextRole.DECLARATION,
        ContextRole.EXAMPLE,
        ContextRole.INSTRUCTION,
        ContextRole.TEST,
        ContextRole.FIXTURE,
    } <= roles
    sources = {item.relative_path: item for item in assistance_index.sources}
    for item in context.items:
        for citation in item.citations:
            assert (
                citation.snapshot_identifier
                == assistance_index.manifest.snapshot_identifier
            )
            assert (
                citation.content_digest
                == sources[citation.location.relative_path].content_digest
            )
            assert citation.location.end_line <= len(
                sources[citation.location.relative_path].source_text.splitlines()
            )
    assert context.used_characters == sum(len(item.text) for item in context.items)
    assert RecordCodec().decode(RecordCodec().encode(context), TaskContext) == context


def test_context_budget_reports_omissions_without_clipping_citations(
    assistance_index: IndexData,
) -> None:
    """
    An item that cannot fit is explicitly omitted as a whole.
    """
    service = TaskAssistance(
        assistance_index, PytestSource().discover(assistance_index)
    )
    first = service.prepare_context(
        FixtureSymbol.HELPER, KnowledgeBundle([]), TaskProfile.DEBUGGING, 1
    )
    second = service.prepare_context(
        FixtureSymbol.HELPER, KnowledgeBundle([]), TaskProfile.DEBUGGING, 1
    )
    assert first == second
    assert first.items == []
    assert first.used_characters == 0
    assert first.omissions
    assert first.character_budget == 1


def test_context_rejects_nonpositive_budget(assistance_index: IndexData) -> None:
    """
    Invalid budgets fail before producing misleading context.
    """
    with pytest.raises(InvalidQuery):
        TaskAssistance(
            assistance_index, PytestSource().discover(assistance_index)
        ).prepare_context(
            FixtureSymbol.HELPER, KnowledgeBundle([]), TaskProfile.NAVIGATION, 0
        )


# %% public contract comparison


def test_change_assessment_compares_contracts_and_retains_old_consumers(
    assistance_index: IndexData,
) -> None:
    """
    A removed definition still has callers in the baseline graph.
    """
    before = next(
        item
        for item in assistance_index.symbols
        if item.symbol_identifier == FixtureSymbol.HELPER
    )
    after = replace(
        assistance_index,
        symbols=[item for item in assistance_index.symbols if item != before],
    )
    report = ChangeAssessment(assistance_index, after).assess_change(
        FixtureSymbol.HELPER
    )
    assert report.contracts[0].change_kind == ChangeKind.REMOVED
    assert report.contracts[0].before == before
    assert report.contracts[0].after is None
    assert FixtureSymbol.CALLER in [
        item.source_identifier for item in report.resolved_consumers
    ]


def test_change_assessment_detects_signature_and_public_member_changes(
    assistance_index: IndexData,
) -> None:
    """
    A public class contract includes its directly declared fields.
    """
    field = next(
        item
        for item in assistance_index.symbols
        if item.symbol_identifier == FixtureSymbol.BASE + ".stored_value"
    )
    updated = replace(field, signature=field.signature + " = 1")
    after = replace(
        assistance_index,
        symbols=[
            updated if item == field else item for item in assistance_index.symbols
        ],
    )
    report = ChangeAssessment(assistance_index, after).assess_change(FixtureSymbol.BASE)
    assert [
        (item.symbol_identifier, item.change_kind)
        for item in report.contracts
        if item.change_kind != ChangeKind.UNCHANGED
    ] == [(field.symbol_identifier, ChangeKind.CHANGED)]
