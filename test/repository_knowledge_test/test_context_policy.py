"""
Context omits unverifiable knowledge and records source truncation.
"""

from __future__ import annotations

from dataclasses import replace

from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.assistance import ContextRole, OmissionReason, TaskProfile
from repository_knowledge.concepts import Concept, KnowledgeBundle, Lifecycle
from repository_knowledge.model import IndexData
from repository_knowledge.operations.assistance import TaskAssistance

from .support import FixtureSymbol

# %% knowledge eligibility


def test_context_keeps_current_draft_knowledge_without_review_authority(
    source_index: IndexData, evidenced_concept: Concept
) -> None:
    """
    A policy-valid draft can aid navigation while retaining its lifecycle.
    """
    concept = replace(
        evidenced_concept,
        metadata={
            "sources": [{"id": "base", "resource": "https://example.invalid/base"}]
        },
        attributions=["base"],
    )
    context = TaskAssistance(
        source_index, PytestSource().discover(source_index)
    ).prepare_context(
        FixtureSymbol.BASE, KnowledgeBundle([concept]), character_budget=30000
    )
    concepts = [item for item in context.items if item.role == ContextRole.CONCEPT]
    assert [item.label for item in concepts] == [concept.relative_path]
    assert concepts[0].lifecycle == Lifecycle.DRAFT
    assert concepts[0].text == concept.body


def test_context_omits_policy_invalid_knowledge(
    source_index: IndexData, evidenced_concept: Concept
) -> None:
    """
    A source-current concept with missing attribution declarations stays omitted.
    """
    context = TaskAssistance(
        source_index, PytestSource().discover(source_index)
    ).prepare_context(
        FixtureSymbol.BASE, KnowledgeBundle([evidenced_concept]), character_budget=30000
    )
    assert [item for item in context.items if item.role == ContextRole.CONCEPT] == []
    assert [
        (item.label, item.reason)
        for item in context.omissions
        if item.reason == OmissionReason.INVALID_KNOWLEDGE
    ] == [(evidenced_concept.relative_path, OmissionReason.INVALID_KNOWLEDGE)]


# %% source excerpts and task profiles


def test_context_marks_partial_declaration_excerpts(
    assistance_index: IndexData,
) -> None:
    """
    Large definitions keep a bounded source excerpt and a visible omitted count.
    """
    service = TaskAssistance(
        assistance_index,
        PytestSource().discover(assistance_index),
        maximum_excerpt_lines=2,
    )
    context = service.prepare_context(
        FixtureSymbol.HELPER, KnowledgeBundle([]), character_budget=30000
    )
    declaration = next(
        item for item in context.items if item.role == ContextRole.DECLARATION
    )
    symbol = next(
        item
        for item in assistance_index.symbols
        if item.symbol_identifier == FixtureSymbol.HELPER
    )
    assert (
        declaration.omitted_lines
        == symbol.location.end_line - symbol.location.start_line - 1
    )
    assert declaration.citations[0].location.end_line == symbol.location.start_line + 1


def test_profiles_prioritize_tests_for_debugging(assistance_index: IndexData) -> None:
    """
    Debugging and extension share evidence but rank it for their task.
    """
    service = TaskAssistance(
        assistance_index, PytestSource().discover(assistance_index)
    )
    debugging = service.prepare_context(
        FixtureSymbol.HELPER, KnowledgeBundle([]), TaskProfile.DEBUGGING, 30000
    )
    extension = service.prepare_context(
        FixtureSymbol.HELPER, KnowledgeBundle([]), TaskProfile.CODE_EXTENSION, 30000
    )
    debug_roles = [item.role for item in debugging.items]
    extension_roles = [item.role for item in extension.items]
    assert debug_roles.index(ContextRole.TEST) < debug_roles.index(ContextRole.EXAMPLE)
    assert extension_roles.index(ContextRole.EXAMPLE) < extension_roles.index(
        ContextRole.TEST
    )


def test_context_instructions_are_limited_to_evidence_paths(
    assistance_index: IndexData,
) -> None:
    """
    Nested test instructions remain scoped to the tests they govern.
    """
    context = TaskAssistance(
        assistance_index, PytestSource().discover(assistance_index)
    ).prepare_context(FixtureSymbol.HELPER, KnowledgeBundle([]), character_budget=30000)
    instruction = next(
        item for item in context.items if item.label == "tests/AGENTS.md"
    )
    assert instruction.applies_to == ["tests/conftest.py", "tests/test_values.py"]


def test_generated_interfaces_never_become_context_payload(
    source_index: IndexData,
) -> None:
    """
    Inventory exclusions cannot leak generated mappings into agent context.
    """
    context = TaskAssistance(
        source_index, PytestSource().discover(source_index)
    ).prepare_context(FixtureSymbol.BASE, KnowledgeBundle([]), character_budget=30000)
    assert [
        citation.location.relative_path
        for item in context.items
        for citation in item.citations
        if citation.location.relative_path.endswith("ormatic_interface.py")
    ] == []
