"""
Tests of the names of events the viewer reads, which the processes writing a run's log
spell the same way: the events of llm_reasoner and of cram_viz are checked here.
"""

from __future__ import annotations

from cram_viz.events import (
    ExecutionFailed,
    ExecutionFinished,
    ExecutionStarted,
    RunFailed,
    RunFinished,
    RunStarted,
)
from llm_reasoner.observation.events import EventKey, ModelAnswered

from llm_reasoner_viz.vocabulary import EventField, KnownEvent


class TestVocabulary:
    def test_the_key_naming_an_event_is_the_one_llm_reasoner_writes(self):
        assert EventField.EVENT == EventKey.EVENT

    def test_a_model_answering_is_named_as_llm_reasoner_names_it(self):
        assert KnownEvent.MODEL_ANSWERED == ModelAnswered.__name__

    def test_the_events_of_a_run_are_named_as_cram_viz_names_them(self):
        assert KnownEvent.RUN_STARTED == RunStarted.__name__
        assert KnownEvent.RUN_FINISHED == RunFinished.__name__
        assert KnownEvent.RUN_FAILED == RunFailed.__name__

    def test_the_events_of_performing_a_run_are_named_as_cram_viz_names_them(self):
        assert KnownEvent.EXECUTION_STARTED == ExecutionStarted.__name__
        assert KnownEvent.EXECUTION_FINISHED == ExecutionFinished.__name__
        assert KnownEvent.EXECUTION_FAILED == ExecutionFailed.__name__

    def test_the_fields_the_viewer_reads_are_written_by_the_events(self):
        started = RunStarted(run="a-run", query="the(...)", scene="pouring").to_json()
        assert started[EventField.QUERY] == "the(...)"
        assert started[EventField.SCENE] == "pouring"
        assert EventField.AT in started
