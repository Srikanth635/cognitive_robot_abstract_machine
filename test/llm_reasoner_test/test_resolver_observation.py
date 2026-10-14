"""
Tests of the resolver reporting its own work: a resolution starting with the slots it
must fill, each attempt at the model's answer and why one is rejected, and how the
resolution ends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from krrood.entity_query_language.factories import the, variable
from typing_extensions import Any

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.observation.events import (
    AttemptRejected,
    AttemptStarted,
    KnowledgeConsulted,
    ResolutionFailed,
    ResolutionFinished,
    ResolutionPurpose,
    ResolutionStarted,
)
from llm_reasoner.observation.sinks import MemorySink
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.exceptions import (
    RESPONSE_REASON,
    ModelResponseUnparseable,
    NoCandidateReferents,
    OutcomeFellShort,
    RepairBudgetExhausted,
)
from llm_reasoner.resolver.knowledge import KnowledgeRequester
from llm_reasoner.resolver.reflection import reflect_slots
from llm_reasoner.world.access import identifier_of

from .test_observed_components import KnowledgeServingOneRecord
from .test_resolver_outcomes import MinimumAngleCheck
from .world_mimics import ActionWithBodyAndValueSlots, WorldBody, body_with_geometry

CASES = Path(__file__).parent / "resolver_cases"
"""
The scripted model answers the tests replay.
"""

# %% helpers


def answer(name: str) -> str:
    return (CASES / name).read_text()


def tilt_query():
    """
    :return: A query for a tilting action whose body and angle are both left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=variable(WorldBody), angle=...)


def resolving(*answers: str, **settings: Any):
    """
    :return: A backend whose model gives the answers in turn, reporting to a memory
        sink, which is returned beside it.
    """
    sink = MemorySink()
    backend = LanguageModelBackend(
        client=ScriptedClient(responses=[answer(name) for name in answers]),
        observer=sink,
        **settings,
    )
    return backend, sink


def kinds(sink: MemorySink) -> list:
    """
    :return: The kinds of the events the sink holds, in order.
    """
    return [type(event) for event in sink.events]


# %% starting


class TestStart:
    def test_a_resolution_is_reported_with_the_action_it_resolves(self):
        body = body_with_geometry()
        backend, sink = resolving("body_and_angle_chosen.json")
        list(tilt_query().evaluate(backend=backend))
        started = sink.events[0]
        assert type(started) is ResolutionStarted
        assert started.action == identifier_of(ActionWithBodyAndValueSlots)
        assert started.purpose is ResolutionPurpose.ACTION
        assert started.max_repairs == backend.max_repairs

    def test_a_resolution_is_reported_with_the_slots_it_must_fill(self):
        body = body_with_geometry()
        backend, sink = resolving("body_and_angle_chosen.json")
        query = tilt_query()
        list(query.evaluate(backend=backend))
        expected = [
            (slot.path, slot.kind.name, slot.type_name) for slot in reflect_slots(query)
        ]
        assert [
            (slot.path, slot.kind, slot.type_name) for slot in sink.events[0].slots
        ] == expected

    def test_the_candidates_of_a_slot_chosen_from_individuals_are_counted(self):
        body = body_with_geometry()
        backend, sink = resolving("body_and_angle_chosen.json")
        query = tilt_query()
        list(query.evaluate(backend=backend))
        expected = [
            len(slot.domain) if slot.selects_from_domain else None
            for slot in reflect_slots(query)
        ]
        assert [slot.candidates for slot in sink.events[0].slots] == expected

    def test_the_situation_of_an_action_is_resolved_for_a_purpose_of_its_own(self):
        backend, sink = resolving()
        situation = backend.situation_backend_for(tilt_query())
        assert situation.purpose is ResolutionPurpose.SITUATION
        assert situation.observer is sink


# %% attempts


class TestAttempts:
    def test_a_usable_first_answer_is_the_only_attempt(self):
        body = body_with_geometry()
        backend, sink = resolving("body_and_angle_chosen.json")
        list(tilt_query().evaluate(backend=backend))
        assert kinds(sink) == [ResolutionStarted, AttemptStarted, ResolutionFinished]
        started, attempt, finished = sink.events
        assert attempt.resolution == started.resolution == finished.resolution
        assert attempt.attempt == 0
        assert finished.attempts == 1
        assert finished.instances == [ActionWithBodyAndValueSlots.__name__]

    def test_an_unusable_answer_is_reported_as_rejected_and_its_repair_as_the_next(
        self,
    ):
        body = body_with_geometry()
        backend, sink = resolving(
            "prose_instead_of_json.txt", "body_and_angle_chosen.json"
        )
        list(tilt_query().evaluate(backend=backend))
        assert kinds(sink) == [
            ResolutionStarted,
            AttemptStarted,
            AttemptRejected,
            AttemptStarted,
            ResolutionFinished,
        ]
        _, first, rejected, second, finished = sink.events
        assert (first.attempt, rejected.attempt, second.attempt) == (0, 0, 1)
        assert rejected.kind == ModelResponseUnparseable.__name__
        assert rejected.answer == answer("prose_instead_of_json.txt")
        assert finished.attempts == 2

    def test_what_a_rejection_found_is_reported_as_data(self):
        body = body_with_geometry()
        backend, sink = resolving(
            "prose_instead_of_json.txt", "body_and_angle_chosen.json"
        )
        list(tilt_query().evaluate(backend=backend))
        (rejected,) = [e for e in sink.events if type(e) is AttemptRejected]
        assert RESPONSE_REASON in rejected.details

    def test_an_answer_falling_short_in_the_simulation_is_reported_with_its_outcome(
        self,
    ):
        body = body_with_geometry()
        check = MinimumAngleCheck(required=1.5)
        backend, sink = resolving(
            "body_and_angle_chosen.json",
            "body_and_steeper_angle_chosen.json",
            outcome_check=check,
        )
        list(tilt_query().evaluate(backend=backend))
        (rejected,) = [e for e in sink.events if type(e) is AttemptRejected]
        assert rejected.kind == OutcomeFellShort.__name__
        assert rejected.details["outcome"]["angle_required"] == check.required


# %% ending


class TestEnd:
    def test_the_models_explanation_is_reported_with_the_instances_it_chose(self):
        body = body_with_geometry()
        backend, sink = resolving("body_and_angle_chosen_with_why.json")
        list(tilt_query().evaluate(backend=backend))
        (chosen,) = json.loads(answer("body_and_angle_chosen_with_why.json"))
        (justification,) = sink.events[-1].justifications
        assert justification.why == chosen["why"]

    def test_a_resolution_out_of_repairs_is_reported_as_failed_and_fails_as_before(
        self,
    ):
        body = body_with_geometry()
        backend, sink = resolving("prose_instead_of_json.txt", max_repairs=0)
        with pytest.raises(RepairBudgetExhausted):
            list(tilt_query().evaluate(backend=backend))
        assert kinds(sink) == [
            ResolutionStarted,
            AttemptStarted,
            AttemptRejected,
            ResolutionFailed,
        ]
        failed = sink.events[-1]
        assert failed.resolution == sink.events[0].resolution
        assert failed.error_type == RepairBudgetExhausted.__name__

    def test_a_query_with_nothing_to_choose_from_is_reported_as_failed(self):
        backend, sink = resolving()
        query = the(ActionWithBodyAndValueSlots)(
            source=variable(WorldBody, domain=[]), angle=...
        )
        with pytest.raises(NoCandidateReferents):
            list(query.evaluate(backend=backend))
        assert kinds(sink) == [ResolutionStarted, ResolutionFailed]
        assert sink.events[-1].error_type == NoCandidateReferents.__name__

    def test_a_query_with_every_slot_given_is_finished_without_asking(self):
        query = the(ActionWithBodyAndValueSlots)(source=body_with_geometry(), angle=1.0)
        backend, sink = resolving()
        list(query.evaluate(backend=backend))
        assert kinds(sink) == [ResolutionStarted, ResolutionFinished]
        assert sink.events[-1].attempts == 0


# %% the knowledge component


class TestKnowledgeConsultation:
    def test_the_consultation_of_the_knowledge_component_is_reported(self):
        body = body_with_geometry()
        backend, sink = resolving(
            "body_and_angle_chosen.json",
            knowledge=KnowledgeRequester(service=KnowledgeServingOneRecord()),
        )
        list(tilt_query().evaluate(backend=backend))
        assert kinds(sink)[:3] == [
            ResolutionStarted,
            KnowledgeConsulted,
            AttemptStarted,
        ]
        consulted = sink.events[1]
        assert consulted.resolution == sink.events[0].resolution
        assert consulted.designator == identifier_of(ActionWithBodyAndValueSlots)
        assert consulted.record_id is None
        assert consulted.situation == {}

    def test_no_consultation_is_reported_without_a_knowledge_component(self):
        body = body_with_geometry()
        backend, sink = resolving("body_and_angle_chosen.json")
        list(tilt_query().evaluate(backend=backend))
        assert KnowledgeConsulted not in kinds(sink)
