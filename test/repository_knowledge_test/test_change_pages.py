"""
Impact reports bound consumer output without hiding match totals.
"""

from __future__ import annotations

import pytest

from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.model import IndexData
from repository_knowledge.operations.assistance import TaskAssistance
from repository_knowledge.operations.changes import ChangeAssessment
from repository_knowledge.assistance import TestCatalog as StaticTestCatalog

from .support import FixtureSymbol

# %% bounded consumers


def test_change_report_pages_consumers_and_preserves_totals(
    source_index: IndexData,
) -> None:
    """
    A page limit bounds both snapshots' resolved and uncertain references.
    """
    service = TaskAssistance(source_index, StaticTestCatalog())
    resolved = service.incoming(FixtureSymbol.BASE)
    uncertain = service.incoming(FixtureSymbol.BASE, uncertain=True)
    report = ChangeAssessment(source_index, source_index).assess_change(
        FixtureSymbol.BASE, limit=1
    )
    assert report.resolved_consumers == resolved[:1]
    assert report.uncertain_consumers == uncertain[:1]
    assert report.current_consumers == resolved[:1]
    assert report.current_uncertain_consumers == uncertain[:1]
    assert (
        report.before_consumer_count.resolved
        == report.after_consumer_count.resolved
        == len(resolved)
    )
    assert (
        report.before_consumer_count.uncertain
        == report.after_consumer_count.uncertain
        == len(uncertain)
    )
    assert report.next_offset == (1 if max(len(resolved), len(uncertain)) > 1 else None)


def test_change_report_rejects_invalid_page_bounds(source_index: IndexData) -> None:
    """
    Negative offsets cannot quietly select unrelated tail references.
    """
    with pytest.raises(InvalidQuery):
        ChangeAssessment(source_index, source_index).assess_change(
            FixtureSymbol.BASE, offset=-1
        )
