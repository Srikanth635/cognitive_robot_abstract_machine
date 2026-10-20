"""
Compare declaration contracts and retain consumers from both snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.assistance import (
    ChangeKind,
    ChangeReport,
    ContractChange,
    ConsumerCount,
)
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.operations.assistance import TaskAssistance
from repository_knowledge.assistance import TestCatalog
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.model import IndexData

# %% contract comparison


@dataclass
class ChangeAssessment:
    """
    Assess a selected public declaration across compatible repository graphs.
    """

    before: IndexData
    """The baseline source graph."""
    after: IndexData
    """
    The requested source graph.
    """

    maximum_results: int = 100
    """
    The maximum references returned in each consumer list.
    """

    def assess_change(
        self,
        symbol_identifier: str,
        before_catalog: TestCatalog | None = None,
        after_catalog: TestCatalog | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ChangeReport:
        """
        Compare public members and report static consumers without safety claims.
        """
        if limit < 1 or limit > self.maximum_results or offset < 0:
            raise InvalidQuery("Consumer page bounds exceed the configured limit.")
        if (
            self.before.manifest.repository_identifier
            != self.after.manifest.repository_identifier
        ):
            raise InvalidQuery(
                "Change assessment requires the same repository history."
            )
        before_service = TaskAssistance(self.before, before_catalog or TestCatalog())
        after_service = TaskAssistance(self.after, after_catalog or TestCatalog())
        before = {
            item.symbol_identifier: item
            for item in before_service.declarations(symbol_identifier)
        }
        after = {
            item.symbol_identifier: item
            for item in after_service.declarations(symbol_identifier)
        }
        if symbol_identifier not in before and symbol_identifier not in after:
            raise InvalidQuery(
                "Change assessment requires an exact declaration in either snapshot."
            )
        contracts = []
        for identifier in sorted(before.keys() | after.keys()):
            old, new = before.get(identifier), after.get(identifier)
            if old is None:
                kind = ChangeKind.ADDED
            elif new is None:
                kind = ChangeKind.REMOVED
            elif (
                old.signature,
                old.base_expressions,
                old.decorators,
                old.symbol_kind,
            ) != (new.signature, new.base_expressions, new.decorators, new.symbol_kind):
                kind = ChangeKind.CHANGED
            else:
                kind = ChangeKind.UNCHANGED
            contracts.append(ContractChange(identifier, kind, old, new))
        directories = set()
        for service in (before_service, after_service):
            report = service.find_tests(symbol_identifier, service.maximum_results)
            directories.update(report.suggested_test_directories)
        resolved = before_service.incoming(symbol_identifier)
        uncertain = before_service.incoming(symbol_identifier, uncertain=True)
        current = after_service.incoming(symbol_identifier)
        current_uncertain = after_service.incoming(symbol_identifier, uncertain=True)
        maximum_count = max(
            len(resolved), len(uncertain), len(current), len(current_uncertain)
        )
        return ChangeReport(
            self.before.manifest,
            self.after.manifest,
            contracts,
            resolved[offset : offset + limit],
            uncertain[offset : offset + limit],
            current[offset : offset + limit],
            current_uncertain[offset : offset + limit],
            sorted(directories),
            [
                "Contract comparison covers declared signatures, decorators, bases and public descendants; body semantics and runtime compatibility are not proved.",
                "Baseline and current consumer records belong to their respective manifests; removed APIs retain baseline callers.",
                "Uncertain name candidates may be unrelated; dynamic dispatch and indirect registrations can be absent entirely.",
                "Suggested test directories are incomplete static proposals; no test execution or complete coverage is implied.",
                "Consumer lists are paginated independently at a common offset; totals describe the complete static matches.",
            ],
            ConsumerCount(len(resolved), len(uncertain)),
            ConsumerCount(len(current), len(current_uncertain)),
            limit,
            offset,
            offset + limit if offset + limit < maximum_count else None,
        )
