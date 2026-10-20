"""
Dispatch task requests through coherent source and test-analysis contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.exceptions import InvalidQuery, SourceScopeMismatch
from repository_knowledge.interfaces.cli import Command, CommandArguments
from repository_knowledge.model import IndexSelection, SelectionKind, SourceScope
from repository_knowledge.operations.assistance import TaskAssistance
from repository_knowledge.operations.changes import ChangeAssessment
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.queries import IndexContents
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.contracts import IndexReader, SnapshotReader, TestAnalyzer
    from repository_knowledge.concepts import KnowledgeBundle
    from repository_knowledge.assistance import (
        ChangeReport,
        DiscoveryReport,
        TaskContext,
    )

# %% task operation dispatch


@dataclass
class ContextApplication:
    """
    Keep one source projection and enforce freshness around task queries.
    """

    index_reader: IndexReader
    """The requested source graph boundary."""
    source_reader: SnapshotReader
    """
    The requested committed or working source boundary.
    """

    test_analyzer: TestAnalyzer
    """The static test declaration boundary."""
    baseline_reader: IndexReader | None = None
    """
    The separately selected committed baseline graph.
    """

    baseline_source: SnapshotReader | None = None
    """
    The committed baseline source boundary.
    """

    def execute(
        self, arguments: CommandArguments, bundle: KnowledgeBundle
    ) -> DiscoveryReport | TaskContext | ChangeReport:
        """
        Return read-only task evidence only for the requested source identities.
        """
        if arguments.source_scope != self.source_reader.source_scope:
            raise SourceScopeMismatch(
                self.source_reader.source_scope, arguments.source_scope
            )
        index = self.index_reader.read(IndexSelection(SelectionKind.TASK_CONTEXT))
        freshness = FreshnessCheck(IndexContents(index), self.source_reader)
        freshness.check(arguments.revision).require_current()
        catalog = self.test_analyzer.discover(index)
        service = TaskAssistance(index, catalog)
        result: DiscoveryReport | TaskContext | ChangeReport
        match arguments.command:
            case Command.FIND_EXAMPLES:
                result = service.find_examples(
                    arguments.symbol_identifier, arguments.limit, arguments.offset
                )
            case Command.FIND_TESTS:
                result = service.find_tests(
                    arguments.symbol_identifier, arguments.limit, arguments.offset
                )
            case Command.PREPARE_CONTEXT:
                result = service.prepare_context(
                    arguments.symbol_identifier,
                    bundle,
                    arguments.profile,
                    arguments.character_budget,
                    arguments.orm_task,
                )
            case Command.ASSESS_CHANGE:
                if (
                    self.baseline_reader is None
                    or self.baseline_source is None
                    or not arguments.baseline_revision
                ):
                    raise InvalidQuery(
                        "Change assessment requires an explicit baseline index and revision."
                    )
                baseline = self.baseline_reader.read(
                    IndexSelection(SelectionKind.TASK_CONTEXT)
                )
                if baseline.manifest.source_scope != SourceScope.COMMITTED:
                    raise SourceScopeMismatch(
                        SourceScope.COMMITTED, baseline.manifest.source_scope
                    )
                FreshnessCheck(IndexContents(baseline), self.baseline_source).check(
                    arguments.baseline_revision
                ).require_current()
                result = ChangeAssessment(baseline, index).assess_change(
                    arguments.symbol_identifier,
                    self.test_analyzer.discover(baseline),
                    catalog,
                    arguments.limit,
                    arguments.offset,
                )
            case _:
                raise InvalidQuery("Unsupported task assistance request.")
        if arguments.source_scope == SourceScope.WORKING_TREE:
            freshness.check(arguments.revision).require_current()
        return result
