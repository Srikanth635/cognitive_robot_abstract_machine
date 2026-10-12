"""
The kitchen cases replayed from recorded answers, so that resolving coraplex designators
on the breakfast scene is checked without a model: a value, a refusal, contradictory
conditions, a task and a search, and a request consulting the knowledge.
"""

from __future__ import annotations

import pytest

from llm_reasoner.knowledge.access import HypothesisStore
from llm_reasoner.resolver.grounds import GroundKey

from experiments.llm_reasoner_kitchen.cases import (
    RESOLUTION_CASES,
    ResolutionCase,
    pours_from_the_cup_into_the_bowl,
)
from experiments.llm_reasoner_kitchen.resolution import KitchenResolver

from .recording import Recording

RECORDED = ("C1", "T1", "P1")
"""
The resolved cases with a recording.
"""

RECORDED_REFUSALS = ("C11", "C12")
"""
The refused cases with a recording.
"""


def case_named(identifier: str) -> ResolutionCase:
    """
    :return: The resolution case with the given identifier.
    """
    (case,) = [case for case in RESOLUTION_CASES if case.identifier == identifier]
    return case


def resolver_replaying(identifier: str, store: HypothesisStore) -> KitchenResolver:
    """
    :return: A resolver whose models replay the case's recording.
    """
    return KitchenResolver(reasoner=Recording.of(identifier).reasoner(store))


@pytest.mark.parametrize("identifier", RECORDED)
def test_a_recorded_case_resolves_to_what_it_expects(identifier, kitchen, tmp_path):
    case = case_named(identifier)
    resolution = resolver_replaying(
        identifier, HypothesisStore(directory=tmp_path)
    ).resolve(case.query(kitchen), case.task)
    assert case.meets(resolution.actions)


@pytest.mark.parametrize("identifier", RECORDED_REFUSALS)
def test_a_recorded_refusal_ends_in_its_refusal(identifier, kitchen, tmp_path):
    case = case_named(identifier)
    resolver = resolver_replaying(identifier, HypothesisStore(directory=tmp_path))
    with pytest.raises(case.refusal):
        resolver.resolve(case.query(kitchen), case.task)


def test_a_pouring_request_consults_the_knowledge_and_rests_on_it(kitchen, tmp_path):
    store = HypothesisStore(directory=tmp_path)
    resolution = resolver_replaying("P1", store).resolve(
        kitchen.pour_from_the_cup_into_the_bowl()
    )
    (justification,) = resolution.backend.justifications
    assert pours_from_the_cup_into_the_bowl(resolution.actions)
    assert [record.record_id for record in store.records()] == [
        resolution.backend.consultation.record_id
    ]
    assert [ground.source for ground in justification.grounds] == [GroundKey.KNOWLEDGE]
