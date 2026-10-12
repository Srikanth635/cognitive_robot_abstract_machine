"""
Tests of the viewer's predefined designators: krrood match queries read from their files
and evaluated on a scene's world, which scenes offer them, and what a resolution
reports, replayed from recorded answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from krrood.entity_query_language.query.match import Match
from llm_reasoner.knowledge.access import HypothesisStore
from llm_reasoner.resolver.grounds import GroundKey
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup
from typing_extensions import Any, Dict

from cram_viz.query_source import QuerySource
from cram_viz.resolution.designators import (
    DESIGNATOR_DIRECTORY,
    QUERY_SUFFIX,
    DesignatorQuery,
    NotAMatchQuery,
    PouringDesignator,
    QuerySlots,
    SceneNamespace,
    UnknownDesignator,
)
from cram_viz.resolution.runs import ResolutionRuns, RunDirectory, RunFile
from cram_viz.resolution.scene_world import SceneKey, SceneWorldBuilder

from .resolutions import resolver_replaying

SOURCES = Path(__file__).parent / "query_sources"
"""
The query sources the tests read.
"""

# %% helpers


def without_object(bundle: Path, identifier: str) -> Path:
    """
    :return: The bundle with the scene object of the given identifier left out.
    """
    scene_file = bundle / "scene.json"
    scene = json.loads(scene_file.read_text())
    scene[SceneKey.OBJECTS] = [
        entry
        for entry in scene[SceneKey.OBJECTS]
        if entry[SceneKey.IDENTIFIER] != identifier
    ]
    scene_file.write_text(json.dumps(scene))
    return bundle


def names_in(bundle: Path) -> Dict[str, Any]:
    """
    :return: The names a designator is evaluated with on the bundle's scene.
    """
    return SceneNamespace(world=SceneWorldBuilder().build(bundle)).names()


# %% the designators


class TestDesignators:
    def test_every_query_file_is_a_designator(self):
        assert sorted(
            path.stem for path in DESIGNATOR_DIRECTORY.glob(f"*{QUERY_SUFFIX}")
        ) == sorted(PouringDesignator)

    def test_every_designator_is_a_match_query_on_a_scene_with_a_pouring_cup(
        self, pouring_bundle
    ):
        names = names_in(pouring_bundle)
        assert all(
            isinstance(DesignatorQuery.shipped(identifier).query_in(names), Match)
            for identifier in PouringDesignator
        )

    def test_a_designator_is_offered_as_its_query_text(self, pouring_bundle):
        identifier = PouringDesignator.OBJECTS_GIVEN
        offered = DesignatorQuery.shipped(identifier).offered_in(
            names_in(pouring_bundle)
        )
        assert (
            offered.query
            == (DESIGNATOR_DIRECTORY / f"{identifier}{QUERY_SUFFIX}").read_text()
        )
        assert offered.available is True

    def test_a_scene_lacking_an_object_the_query_names_cannot_offer_it(
        self, pouring_bundle
    ):
        names = names_in(without_object(pouring_bundle, "Static_MilkPitcher"))
        offered = DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN).offered_in(
            names
        )
        assert offered.unbound == [SceneNamespace.name_of(PouringCup)]
        assert offered.available is False

    def test_a_scene_with_nothing_to_take_for_a_slot_cannot_offer_it(
        self, pouring_bundle
    ):
        names = names_in(without_object(pouring_bundle, "Static_MilkPitcher"))
        offered = DesignatorQuery.shipped(PouringDesignator.OBJECTS_OPEN).offered_in(
            names
        )
        assert offered.without_candidates == ["source_container"]
        assert offered.available is False

    def test_an_unknown_designator_is_refused(self):
        with pytest.raises(UnknownDesignator):
            DesignatorQuery.named("no_such_designator")

    def test_a_query_that_is_not_a_match_is_refused(self, pouring_bundle):
        designator = DesignatorQuery(
            identifier=PouringDesignator.OBJECTS_OPEN,
            source=QuerySource.read(SOURCES / "not_a_match.py"),
        )
        with pytest.raises(NotAMatchQuery):
            designator.query_in(names_in(pouring_bundle))


# %% the slots


class TestQuerySlots:
    def test_the_slots_are_named_in_the_order_the_query_states_them(
        self, pouring_bundle
    ):
        query = DesignatorQuery.shipped(PouringDesignator.OBJECTS_OPEN).query_in(
            names_in(pouring_bundle)
        )
        assert QuerySlots.of(query).names == [
            "source_container",
            "target_container",
            "arm",
            "tilt_angle",
        ]

    def test_only_a_value_the_query_states_is_given(self, pouring_bundle):
        query = DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN).query_in(
            names_in(pouring_bundle)
        )
        assert QuerySlots.of(query).given == ["source_container", "target_container"]


# %% resolving


def resolve_objects_given(bundle: Path, tmp_path: Path, times: int = 1) -> list:
    """
    :return: The runs of resolving the designator giving the objects, the given number
        of times in turn, each replaying the recording from the empty store at
        ``tmp_path / "store"``.
    """
    resolver = resolver_replaying(
        PouringDesignator.OBJECTS_GIVEN,
        HypothesisStore(directory=tmp_path / "store"),
        tmp_path / "runs",
        times=times,
    )
    designator = DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN)
    return [resolver.resolve(designator, "pouring", bundle) for _ in range(times)]


class TestResolution:
    def test_a_resolution_names_the_value_of_every_slot(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        (action,) = run.resolution.actions
        assert {slot.name: slot.value for slot in action.slots} == {
            "target_container": "bowl.stl",
            "source_container": "PouringCup (Static_MilkPitcher.stl)",
            "arm": "RIGHT",
            "tilt_angle": "1.85",
        }

    def test_a_resolution_marks_the_slots_its_query_states_as_given(
        self, pouring_bundle, tmp_path
    ):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        (action,) = run.resolution.actions
        query = DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN).query_in(
            names_in(pouring_bundle)
        )
        assert [slot.name for slot in action.slots if slot.given] == (
            QuerySlots.of(query).given
        )

    def test_a_resolution_states_the_query_it_resolved(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        designator = DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN)
        assert run.resolution.identifier is designator.identifier
        assert run.resolution.query == designator.source.text

    def test_a_resolution_names_what_its_answer_rests_on(
        self, pouring_bundle, tmp_path
    ):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        (action,) = run.resolution.actions
        assert [(ground.source, ground.name) for ground in action.grounds] == [
            (GroundKey.KNOWLEDGE, "pour"),
        ]

    def test_a_resolution_reports_the_knowledge_it_consulted(
        self, pouring_bundle, tmp_path
    ):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        (record,) = run.knowledge.backend.store.records()
        assert run.resolution.record_id == record.record_id
        assert run.resolution.situation == {}


# %% runs kept apart


class TestRuns:
    def test_a_run_starts_from_a_copy_of_the_baseline_store(self, tmp_path):
        baseline = tmp_path / "store"
        (baseline / "record").mkdir(parents=True)
        (baseline / "record" / "content").write_text("stored")
        run = ResolutionRuns(directory=tmp_path / "runs").start(
            HypothesisStore(directory=baseline)
        )
        assert (
            run.knowledge_store.directory / "record" / "content"
        ).read_text() == "stored"

    def test_what_a_run_generates_stays_out_of_the_baseline_store(
        self, pouring_bundle, tmp_path
    ):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        assert HypothesisStore(directory=tmp_path / "store").records() == []
        assert len(run.knowledge.backend.store.records()) == 1

    def test_every_run_starts_from_the_same_baseline_store(
        self, pouring_bundle, tmp_path
    ):
        first, second = resolve_objects_given(pouring_bundle, tmp_path, times=2)
        assert first.resolution.run != second.resolution.run
        assert [
            len(run.knowledge.backend.store.records()) for run in (first, second)
        ] == [
            1,
            1,
        ]

    def test_a_run_keeps_its_resolution_beside_its_knowledge(
        self, pouring_bundle, tmp_path
    ):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        kept = RunDirectory(directory=tmp_path / "runs" / run.resolution.run)
        assert json.loads((kept.directory / RunFile.RESOLUTION).read_text()) == (
            json.loads(json.dumps(run.resolution.payload()))
        )
