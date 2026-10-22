"""
Tests of the log each resolution run keeps of its work: that it started and how it ends,
what the resolver and the knowledge component asked the model, the hypothesis served,
and each pour tried in the simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_reasoner.clients.base import ScriptExhausted
from llm_reasoner.knowledge.access import HypothesisStore
from llm_reasoner.observation.events import (
    Component,
    CompletionStyle,
    EventKey,
    HypothesisRequested,
    HypothesisServed,
    ModelAsked,
    ResolutionFinished,
)
from typing_extensions import Any, Dict, List

from cramera.resolution.events import (
    RunFailed,
    RunFinished,
    RunStarted,
    TrialFinished,
    TrialStarted,
)
from cramera.resolution.resolver.designators import DesignatorQuery, PouringDesignator
from cramera.resolution.resolver.runs import RunFile

from .resolutions import LANDS, MISSES, recorded, resolver_replaying

# %% helpers


def logged(directory: Path) -> List[Dict[str, Any]]:
    """
    :return: The events a run's directory holds, as written.
    """
    return [
        json.loads(line)
        for line in (directory / RunFile.EVENTS).read_text().splitlines()
    ]


def of_kind(events: List[Dict[str, Any]], kind: type) -> List[Dict[str, Any]]:
    """
    :return: The events of one kind, in order.
    """
    return [event for event in events if event[EventKey.EVENT] == kind.__name__]


def resolve(
    pouring_bundle_with_arms: Path,
    tmp_path: Path,
    recording: str = "objects_given",
    **settings,
):
    """
    :return: The run of resolving the designator giving the objects, replaying the
        recording, and the events the run logged.
    """
    resolver = resolver_replaying(
        recording,
        HypothesisStore(directory=tmp_path / "store"),
        tmp_path / "runs",
        **settings,
    )
    run = resolver.resolve(
        DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN),
        "pouring",
        pouring_bundle_with_arms,
    )
    return run, logged(tmp_path / "runs" / run.resolution.run)


# %% the run


class TestRun:
    def test_a_run_logs_that_it_started_with_its_query_and_scene(
        self, pouring_bundle_with_arms, tmp_path
    ):
        run, events = resolve(pouring_bundle_with_arms, tmp_path)
        started = events[0]
        assert started[EventKey.EVENT] == RunStarted.__name__
        assert started["run"] == run.resolution.run
        assert started["query"] == run.resolution.query
        assert started["scene"] == "pouring"

    def test_a_run_logs_that_it_finished_and_how_long_it_took(
        self, pouring_bundle_with_arms, tmp_path
    ):
        run, events = resolve(pouring_bundle_with_arms, tmp_path)
        finished = events[-1]
        assert finished[EventKey.EVENT] == RunFinished.__name__
        assert finished["run"] == run.resolution.run
        assert finished["seconds"] >= 0.0

    def test_a_run_that_fails_is_logged_as_failed_and_fails_as_before(
        self, pouring_bundle_with_arms, tmp_path
    ):
        resolver = resolver_replaying(
            PouringDesignator.OBJECTS_GIVEN,
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        resolver.client.tool_turns.clear()
        resolver.baseline.backend.client.responses.clear()
        with pytest.raises(ScriptExhausted):
            resolver.resolve(
                DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN),
                "pouring",
                pouring_bundle_with_arms,
            )
        (directory,) = (tmp_path / "runs").iterdir()
        events = logged(directory)
        assert events[0][EventKey.EVENT] == RunStarted.__name__
        assert events[-1][EventKey.EVENT] == RunFailed.__name__
        assert events[-1]["error_type"] == ScriptExhausted.__name__
        assert events[-1]["run"] == directory.name

    def test_every_event_is_a_line_of_json_naming_its_kind(
        self, pouring_bundle_with_arms, tmp_path
    ):
        _, events = resolve(pouring_bundle_with_arms, tmp_path)
        assert all(isinstance(event[EventKey.EVENT], str) for event in events)

    def test_two_runs_keep_logs_apart(self, pouring_bundle_with_arms, tmp_path):
        first, first_events = resolve(pouring_bundle_with_arms, tmp_path / "first")
        second, second_events = resolve(pouring_bundle_with_arms, tmp_path / "second")
        assert len(of_kind(first_events, RunStarted)) == 1
        assert len(of_kind(second_events, RunStarted)) == 1


# %% what the components did


class TestComponents:
    def test_what_the_resolver_asked_the_model_is_logged_as_the_resolvers(
        self, pouring_bundle_with_arms, tmp_path
    ):
        _, events = resolve(pouring_bundle_with_arms, tmp_path)
        asked = of_kind(events, ModelAsked)
        resolver_asked = [
            event for event in asked if event["source"] == Component.RESOLVER
        ]
        assert len(resolver_asked) == len(recorded("objects_given")["turns"])

    def test_the_resolution_is_logged(self, pouring_bundle_with_arms, tmp_path):
        _, events = resolve(pouring_bundle_with_arms, tmp_path)
        assert len(of_kind(events, ResolutionFinished)) >= 1

    def test_what_the_knowledge_component_asked_the_model_is_logged_as_the_knowledges(
        self, pouring_bundle_with_arms, tmp_path
    ):
        _, events = resolve(pouring_bundle_with_arms, tmp_path)
        generation = [
            event
            for event in of_kind(events, ModelAsked)
            if event["source"] == Component.KNOWLEDGE
        ]
        assert len(generation) == len(recorded("objects_given")["generations"])
        assert all(event["style"] == CompletionStyle.STRUCTURED for event in generation)

    def test_the_hypothesis_generated_is_logged_as_served(
        self, pouring_bundle_with_arms, tmp_path
    ):
        run, events = resolve(pouring_bundle_with_arms, tmp_path)
        (requested,) = of_kind(events, HypothesisRequested)
        (served,) = of_kind(events, HypothesisServed)
        assert served["record_id"] == run.resolution.record_id
        assert served["from_store"] is False
        assert requested["instruction"]


# %% the pours tried in the simulation


class TestTrials:
    def test_each_pour_tried_is_logged_with_what_it_brought_about(
        self, pouring_bundle_with_arms, tmp_path
    ):
        run, events = resolve(
            pouring_bundle_with_arms,
            tmp_path,
            recording="objects_given_retried",
            trial_outcomes=[MISSES, LANDS],
        )
        started = of_kind(events, TrialStarted)
        finished = of_kind(events, TrialFinished)
        assert [event["outcome"] for event in finished] == [MISSES, LANDS]
        assert [event["values"] for event in started] == [
            trial.values for trial in run.resolution.trials
        ]
        assert [event["trial"] for event in started] == [
            event["trial"] for event in finished
        ]
