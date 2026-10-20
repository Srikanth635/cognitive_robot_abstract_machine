"""
Context prioritizes resolved tests and makes generator inputs explicit.
"""

from __future__ import annotations

from dataclasses import replace

from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.assistance import (
    ContextRole,
    TaskProfile,
    TestCatalog as StaticTestCatalog,
)
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.model import IndexData, Resolution
from repository_knowledge.operations.assistance import TaskAssistance

from .support import FixtureSymbol

# %% confidence before lexical ordering


def test_test_discovery_ranks_resolved_consumers_before_name_candidates(
    assistance_index: IndexData,
) -> None:
    """
    An early lexical name cannot displace an exactly resolved test call.
    """
    catalog = PytestSource().discover(assistance_index)
    direct = next(item for item in catalog.tests if item.symbol_name == "test_direct")
    parameter = next(
        item for item in catalog.tests if item.symbol_name == "test_parameter"
    )
    references = [
        (
            replace(item, resolution=Resolution.UNRESOLVED, target_identifier="")
            if item.source_identifier == direct.symbol_identifier
            and item.target_identifier == FixtureSymbol.HELPER
            else item
        )
        for item in assistance_index.references
    ]
    index = replace(assistance_index, references=references)
    report = TaskAssistance(
        index, StaticTestCatalog([direct, parameter], [])
    ).find_tests(FixtureSymbol.HELPER)
    assert [(item.symbol_name, item.resolution) for item in report.tests] == [
        (parameter.symbol_name, Resolution.RESOLVED),
        (direct.symbol_name, Resolution.CANDIDATE),
    ]


def test_class_context_includes_public_member_signatures(
    source_index: IndexData,
) -> None:
    """
    The public contract remains available even when a class body is long.
    """
    context = TaskAssistance(
        source_index, PytestSource().discover(source_index)
    ).prepare_context(FixtureSymbol.BASE, KnowledgeBundle([]), character_budget=30000)
    field = next(
        item
        for item in source_index.symbols
        if item.symbol_identifier == FixtureSymbol.BASE + ".stored_value"
    )
    interfaces = [item for item in context.items if item.role == ContextRole.INTERFACE]
    assert any(field.signature in item.text for item in interfaces)


def test_orm_context_routes_to_generator_and_source_inputs(
    assistance_index: IndexData,
) -> None:
    """
    An explicit ORM task gets a generator and its resolved source inputs.
    """
    service = TaskAssistance(
        assistance_index, PytestSource().discover(assistance_index)
    )
    context = service.prepare_context(
        FixtureSymbol.BASE,
        KnowledgeBundle([]),
        TaskProfile.MIGRATION,
        30000,
        orm_task=True,
    )
    generators = [
        citation.location.relative_path
        for item in context.items
        if item.role == ContextRole.GENERATOR
        for citation in item.citations
    ]
    assert "scripts/regenerate_fixture_orm.py" in generators
    assert "first/src/first/base.py" in generators
    ordinary = service.prepare_context(
        FixtureSymbol.BASE, KnowledgeBundle([]), character_budget=30000
    )
    assert [item for item in ordinary.items if item.role == ContextRole.GENERATOR] == []
