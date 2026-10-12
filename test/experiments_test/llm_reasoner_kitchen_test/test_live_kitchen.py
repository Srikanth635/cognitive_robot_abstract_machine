"""
The kitchen cases against the live model, run only when asked to
(``LLM_REASONER_LIVE=1`` and an API key), since every case costs money.

Their scripted twins, which replay recorded answers, run everywhere.
"""

from __future__ import annotations

import pytest

from llm_reasoner.knowledge.access import HypothesisStore

from experiments.llm_reasoner_kitchen.cases import (
    KNOWLEDGE_CASES,
    RESOLUTION_CASES,
    KnowledgeCase,
    ResolutionCase,
    pours_from_the_cup_into_the_bowl,
)
from experiments.llm_reasoner_kitchen.live import LiveCheck, LiveCheckVariable
from experiments.llm_reasoner_kitchen.resolution import KitchenResolver
from llm_reasoner.clients.usage import SONNET_5_5_PRICES

LIVE = LiveCheck()
"""
Whether and how the cases are checked against the live model.
"""

pytestmark = pytest.mark.skipif(
    not LIVE.enabled(),
    reason=f"live checks run only with {LiveCheckVariable.OPT_IN} set and an API key",
)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> HypothesisStore:
    """
    Where the knowledge component stores the hypotheses it generates during the run.
    """
    return HypothesisStore(
        directory=LIVE.store_directory(tmp_path_factory.mktemp("knowledge"))
    )


def resolver(store: HypothesisStore) -> KitchenResolver:
    """
    :return: A resolver asking the live model.
    """
    return KitchenResolver(reasoner=LIVE.reasoner(store))


def identifier_of_case(case: ResolutionCase | KnowledgeCase) -> str:
    """
    :return: The case's identifier, naming its test.
    """
    return case.identifier


@pytest.mark.parametrize(
    "case",
    [case for case in RESOLUTION_CASES if case.refusal is None and not case.furnished],
    ids=identifier_of_case,
)
def test_a_case_resolves_to_what_it_expects(case, kitchen, store):
    resolution = resolver(store).resolve(case.query(kitchen), case.task)
    print(f"{case.identifier}: ${resolution.cost(SONNET_5_5_PRICES):.4f}")
    assert case.meets(resolution.actions)


@pytest.mark.parametrize(
    "case",
    [case for case in RESOLUTION_CASES if case.furnished],
    ids=identifier_of_case,
)
def test_a_case_among_the_furniture_resolves_to_what_it_expects(
    case, furnished_kitchen, store
):
    resolution = resolver(store).resolve(case.query(furnished_kitchen), case.task)
    print(f"{case.identifier}: ${resolution.cost(SONNET_5_5_PRICES):.4f}")
    assert case.meets(resolution.actions)


@pytest.mark.parametrize(
    "case",
    [case for case in RESOLUTION_CASES if case.refusal is not None],
    ids=identifier_of_case,
)
def test_a_case_ends_in_the_refusal_it_expects(case, kitchen, store):
    with pytest.raises(case.refusal):
        resolver(store).resolve(case.query(kitchen), case.task)


@pytest.mark.parametrize("case", KNOWLEDGE_CASES, ids=identifier_of_case)
def test_a_task_consults_the_knowledge_for_its_situation(case, kitchen, store):
    resolution = resolver(store).resolve(
        kitchen.pour_from_the_cup_into_the_bowl(), case.task
    )
    print(f"{case.identifier}: ${resolution.cost(SONNET_5_5_PRICES):.4f}")
    assert set(resolution.backend.consultation.situation.values()) == set(
        case.situation
    )
    assert pours_from_the_cup_into_the_bowl(resolution.actions)
