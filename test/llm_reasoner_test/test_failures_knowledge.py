"""
Tests of knowledge about the failures of an action: a hypothesis about pouring in the
failures vocabulary, generated on top of a stored core hypothesis, checked, stored and
read with it.
"""

from __future__ import annotations

import json

import pytest
from krrood.entity_query_language.factories import the

from llm_reasoner.clients.tools import ToolBox
from llm_reasoner.knowledge.admission.hermit import HermitCheck
from llm_reasoner.knowledge.generation.prompts import GenerationSection
from llm_reasoner.knowledge.hypotheses.knowledge_access import KnowledgeToolName
from llm_reasoner.knowledge.ontologies import ActionPerspectiveFile
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    failures,
)
from llm_reasoner.knowledge.storage.record import ConsistencyVerdict
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective

from .prompt_sections import bodies
from .test_action_perspective_knowledge import (
    AnswerCase,
    answer_text as answer_in_cases,
    asked,
    core_query,
    service_answering,
    system_prompt_for,
)


class FailuresCase:
    """
    The scripted answer about the failures of pouring, by file.
    """

    FAILURES = "pouring_failures.json"


def failures_query_on(core_view):
    """
    :return: A query for the failure stereotype of a stored core hypothesis's frame.
    """
    return the(failures.FailureStereotype)(
        failure_stereotype_of=core_view.citation_of(core_view.root)
    )


@pytest.fixture
def stored(tmp_path):
    """
    The scripted core and failures of pouring, generated and stored, with the service
    holding both and the views of the two.
    """
    service = service_answering(
        HypothesisStore(directory=tmp_path),
        answer_in_cases(AnswerCase.CORE),
        answer_in_cases(FailuresCase.FAILURES),
    )
    core_record = service.serve(core_query())
    failures_record = service.serve(failures_query_on(service.view(core_record)))
    return service, core_record, failures_record


# %% the hypothesis


class TestFailureHypothesis:
    def test_it_is_about_the_stored_core_frame_itself(self, stored):
        service, core_record, failures_record = stored
        assert (
            service.view(failures_record).root.failure_stereotype_of
            is service.view(core_record).root
        )

    def test_its_failures_name_the_core_slots_and_dimensions_themselves(self, stored):
        service, core_record, failures_record = stored
        core = service.view(core_record).root
        core_slots = {id(slot) for slot in core.defines_slot}
        core_dimensions = {id(dimension) for dimension in core.leaves_open}
        failures = service.view(failures_record).root.has_failure
        assert failures
        for failure in failures:
            assert {id(slot) for slot in failure.failure_concerns_slot} <= core_slots
            assert {
                id(dimension) for dimension in failure.caused_by_dimension
            } <= core_dimensions

    def test_a_correction_changes_a_dimension_the_failure_comes_from(self, stored):
        service, _, failures_record = stored
        for failure in service.view(failures_record).root.has_failure:
            changed = {
                id(change.changes_dimension)
                for correction in failure.has_correction
                for change in correction.has_change
            }
            assert changed <= {
                id(dimension) for dimension in failure.caused_by_dimension
            }

    def test_a_correction_moves_the_lip_towards_the_opening(self, stored):
        service, core_record, failures_record = stored
        core = service.view(core_record).root
        (lip_failure,) = [
            failure
            for failure in service.view(failures_record).root.has_failure
            if any(
                dimension in core.leaves_open
                and dimension.has_quantity_kind is action_core.QuantityKind.POSITION
                for dimension in failure.caused_by_dimension
            )
        ]
        (change,) = [
            change
            for correction in lip_failure.has_correction
            for change in correction.has_change
        ]
        assert change.has_direction is failures.ParameterChangeDirection.TOWARDS_TARGET

    def test_its_record_builds_on_the_core_record(self, stored):
        service, core_record, failures_record = stored
        assert service.view(failures_record).record.builds_on == [core_record]


# %% asking for it


class TestGenerationBrief:
    def test_it_is_asked_with_the_description_its_vocabulary_was_proposed_from(self):
        role = bodies(
            system_prompt_for(failures.FailureStereotype), GenerationSection.ROLE
        )
        assert ActionPerspectiveFile.FAILURES.description in role[0]
        assert (
            Perspective.FAILURES.description
            == ActionPerspectiveFile.FAILURES.description
        )


# %% reading it with the core


class TestReadingWithTheCore:
    def test_the_overview_names_the_perspective_read_with_the_core(self, stored):
        service, core_record, failures_record = stored
        overview = asked(
            ToolBox(tools=service.tools(core_record, failures_record)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        modules = {perspective["module"] for perspective in overview["perspectives"]}
        assert modules == {ActionPerspectiveFile.FAILURES.value}


# %% reasoning


@pytest.mark.slow
class TestAdmission:
    def test_it_is_admitted_by_the_reasoner_with_the_core_it_builds_on(self, tmp_path):
        check = HermitCheck.from_environment()
        if check is None or HermitCheck.shipped_directory() is None:
            pytest.skip("needs the linked ontologies and owlready2's HermiT")
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_in_cases(AnswerCase.CORE),
            answer_in_cases(FailuresCase.FAILURES),
            consistency=check,
        )
        core_view = service.view_for(core_query())
        failures_view = service.view_for(failures_query_on(core_view))
        assert failures_view.record.consistency == ConsistencyVerdict.CONSISTENT
