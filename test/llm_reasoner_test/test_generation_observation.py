"""
Tests of the knowledge component reporting a hypothesis being generated: what was
requested, each attempt at the model's answer and why one is rejected, and whether the
hypothesis came from the store or was generated.
"""

from __future__ import annotations

import pytest

from llm_reasoner.knowledge.generation.dialogue import HypothesisRejected
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.storage.record import qualified_name
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.observation.events import (
    GenerationAttemptRejected,
    GenerationAttemptStarted,
    GenerationFailed,
    HypothesisRequested,
    HypothesisServed,
)
from llm_reasoner.observation.sinks import MemorySink

from .test_generation import GenerationCase, backend_answering, pouring_query

# %% helpers


def reporting(*cases: GenerationCase):
    """
    :return: A backend whose model gives the scripted answers in turn, reporting to a
        memory sink, which is returned beside it.
    """
    backend = backend_answering(*cases)
    sink = MemorySink()
    backend.observer = sink
    return backend, sink


def kinds(sink: MemorySink) -> list:
    """
    :return: The kinds of the events the sink holds, in order.
    """
    return [type(event) for event in sink.events]


# %% generating


class TestGeneration:
    def test_what_is_requested_is_reported(self):
        backend, sink = reporting(GenerationCase.CHAINED_STRUCTURE)
        list(pouring_query().evaluate(backend=backend))
        request = HypothesisRequest.from_match(pouring_query(), backend.instruction)
        requested = sink.events[0]
        assert type(requested) is HypothesisRequested
        assert requested.root_type == qualified_name(request.root_type)
        assert requested.fixed_values == request.written_fixed_values
        assert requested.instruction == backend.instruction

    def test_an_admitted_first_answer_is_the_only_attempt(self):
        backend, sink = reporting(GenerationCase.CHAINED_STRUCTURE)
        list(pouring_query().evaluate(backend=backend))
        assert kinds(sink) == [
            HypothesisRequested,
            GenerationAttemptStarted,
            HypothesisServed,
        ]
        served = sink.events[-1]
        assert served.from_store is False
        assert served.attempts == 1
        assert served.record_id == backend.last_record.record_id

    def test_a_rejected_answer_is_reported_with_its_problems_and_repaired(self):
        backend, sink = reporting(
            GenerationCase.LOOPING_STRUCTURE, GenerationCase.LOOP_REPLACED_PATCH
        )
        list(pouring_query().evaluate(backend=backend))
        assert kinds(sink) == [
            HypothesisRequested,
            GenerationAttemptStarted,
            GenerationAttemptRejected,
            GenerationAttemptStarted,
            HypothesisServed,
        ]
        _, first, rejected, second, served = sink.events
        assert (first.attempt, rejected.attempt, second.attempt) == (0, 0, 1)
        assert rejected.problems
        assert all(isinstance(problem, str) for problem in rejected.problems)
        assert served.attempts == 2

    def test_a_hypothesis_nobody_admits_is_reported_as_failed_and_fails_as_before(self):
        backend, sink = reporting(GenerationCase.LOOPING_STRUCTURE)
        backend.max_repairs = 0
        with pytest.raises(HypothesisRejected):
            list(pouring_query().evaluate(backend=backend))
        assert kinds(sink) == [
            HypothesisRequested,
            GenerationAttemptStarted,
            GenerationAttemptRejected,
            GenerationFailed,
        ]
        rejected, failed = sink.events[-2:]
        assert failed.attempts == 1
        assert failed.problems == rejected.problems


# %% the store


class TestStore:
    def test_a_hypothesis_served_from_the_store_is_reported_without_an_attempt(
        self, tmp_path
    ):
        backend, sink = reporting(GenerationCase.CHAINED_STRUCTURE)
        backend.store = HypothesisStore(directory=tmp_path)
        list(pouring_query().evaluate(backend=backend))
        sink.events.clear()
        list(pouring_query().evaluate(backend=backend))
        assert kinds(sink) == [HypothesisRequested, HypothesisServed]
        served = sink.events[-1]
        assert served.from_store is True
        assert served.attempts == 0
        assert served.record_id == backend.last_record.record_id
