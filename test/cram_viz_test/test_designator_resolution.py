"""
Tests of the viewer's predefined designators: krrood match queries read from their files
and evaluated on a scene's world, which scenes offer them, and what a resolution
reports, replayed from recorded answers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from krrood.entity_query_language.query.match import Match
from llm_reasoner.knowledge.access import HypothesisStore
from llm_reasoner.resolver.exceptions import OutcomeFellShort
from llm_reasoner.resolver.grounds import GroundKey
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup
from typing_extensions import Any, Dict

from cram_viz.query_source import QuerySource
from cram_viz.resolution.designators import (
    DESIGNATOR_DIRECTORY,
    QUERY_SUFFIX,
    DesignatorQuery,
    DesignatorUnavailable,
    NotAMatchQuery,
    OfferedDesignator,
    PouringDesignator,
    QuerySlots,
    SceneNamespace,
)
from cram_viz.resolution.resolver import WORLD_MODEL_MODULES
from cram_viz.resolution.runs import (
    ResolutionRuns,
    RunDirectory,
    RunEnvironment,
    RunFile,
    UnknownRun,
)
from cram_viz.resolution.scene_geometry import SceneGeometry
from cram_viz.resolution.scene_world import SceneKey, SceneWorldBuilder

from .resolutions import LANDS, MISSES, resolver_replaying

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
        offered = OfferedDesignator.in_scene(identifier, names_in(pouring_bundle))
        assert (
            offered.query
            == (DESIGNATOR_DIRECTORY / f"{identifier}{QUERY_SUFFIX}").read_text()
        )
        assert offered.available is True

    def test_a_designator_is_offered_on_one_line_too(self, pouring_bundle):
        identifier = PouringDesignator.OBJECTS_GIVEN
        offered = OfferedDesignator.in_scene(identifier, names_in(pouring_bundle))
        assert offered.payload()["query_on_one_line"] == (
            DesignatorQuery.shipped(identifier).source.on_one_line
        )

    def test_a_scene_lacking_an_object_the_query_names_cannot_offer_it(
        self, pouring_bundle
    ):
        names = names_in(without_object(pouring_bundle, "Static_MilkPitcher"))
        offered = OfferedDesignator.in_scene(PouringDesignator.OBJECTS_GIVEN, names)
        assert offered.unbound == [SceneNamespace.name_of(PouringCup)]
        assert offered.available is False

    def test_a_scene_with_nothing_to_take_for_a_slot_cannot_offer_it(
        self, pouring_bundle
    ):
        names = names_in(without_object(pouring_bundle, "Static_MilkPitcher"))
        offered = OfferedDesignator.in_scene(PouringDesignator.OBJECTS_OPEN, names)
        assert offered.without_candidates == ["source_container"]
        assert offered.available is False

    def test_a_query_that_is_not_a_match_is_refused(self, pouring_bundle):
        designator = DesignatorQuery(
            source=QuerySource.read(SOURCES / "not_a_match.py")
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

    def test_the_pour_geometry_designator_leaves_the_geometry_to_the_resolver(
        self, pouring_bundle
    ):
        query = DesignatorQuery.shipped(PouringDesignator.POUR_GEOMETRY_OPEN).query_in(
            names_in(pouring_bundle)
        )
        slots = QuerySlots.of(query)
        assert slots.given == ["source_container", "target_container"]
        assert [name for name in slots.names if name not in slots.given] == [
            "arm",
            "tilt_angle",
            "pour_side",
            "pour_side_offset",
            "pour_approach_offset",
            "pour_height",
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


def reopening_resolver(tmp_path: Path):
    """
    :return: A resolver over the runs ``resolve_objects_given`` keeps in ``tmp_path``.
    """
    return resolver_replaying(
        PouringDesignator.OBJECTS_GIVEN,
        HypothesisStore(directory=tmp_path / "store"),
        tmp_path / "runs",
    )


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

    def test_a_resolved_slot_keeps_its_value_as_data(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        (action,) = run.resolution.actions
        assert {slot.name: slot.serialized for slot in action.slots} == {
            "source_container": "Static_MilkPitcher.stl",
            "target_container": "bowl.stl",
            "arm": "RIGHT",
            "tilt_angle": 1.85,
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

    def test_the_resolver_reads_the_scene_geometry(self, pouring_bundle, tmp_path):
        resolver = resolver_replaying(
            PouringDesignator.OBJECTS_GIVEN,
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        resolver.resolve(
            DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN),
            "pouring",
            pouring_bundle,
        )
        geometry = SceneGeometry(
            world=SceneWorldBuilder().build(pouring_bundle)
        ).context_for(None)
        assert [geometry in user for _, user in resolver.client.calls] == [True] * len(
            resolver.client.calls
        )

    def test_a_resolution_states_the_query_it_resolved(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        designator = DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN)
        assert run.resolution.query == designator.source.text

    def test_a_resolution_folds_its_query_onto_one_line(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        one_line = run.resolution.query_on_one_line
        assert "\n" not in one_line
        assert one_line.split() == run.resolution.query.split()

    def test_the_resolvers_world_view_loads_the_modules_of_the_world_model(
        self, tmp_path, monkeypatch
    ):
        resolver = resolver_replaying(
            PouringDesignator.OBJECTS_GIVEN,
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        for name in WORLD_MODEL_MODULES:
            monkeypatch.delitem(sys.modules, name, raising=False)
        resolver.world_view()
        assert all(name in sys.modules for name in WORLD_MODEL_MODULES)

    def test_a_query_the_scene_cannot_resolve_is_refused(
        self, pouring_bundle, tmp_path
    ):
        resolver = resolver_replaying(
            PouringDesignator.OBJECTS_GIVEN,
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        with pytest.raises(DesignatorUnavailable):
            resolver.resolve(
                DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN),
                "pouring",
                without_object(pouring_bundle, "Static_MilkPitcher"),
            )

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


# %% trying pours in the simulation


class TestTrials:
    def test_a_pour_that_misses_is_resolved_again_with_its_outcome(
        self, pouring_bundle, tmp_path
    ):
        resolver = resolver_replaying(
            "objects_given_retried",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
            trial_outcomes=[MISSES, LANDS],
        )
        resolver.resolve(
            DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN),
            "pouring",
            pouring_bundle,
        )
        _, retried = resolver.client.calls[-1]
        assert f'kind="{OutcomeFellShort.__name__}"' in retried
        assert json.dumps({"outcome": MISSES}, indent=2) in retried

    def test_a_resolution_keeps_the_pours_it_tried(self, pouring_bundle, tmp_path):
        resolver = resolver_replaying(
            "objects_given_retried",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
            trial_outcomes=[MISSES, LANDS],
        )
        run = resolver.resolve(
            DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN),
            "pouring",
            pouring_bundle,
        )
        (action,) = run.resolution.actions
        assert [trial.outcome for trial in run.resolution.trials] == [MISSES, LANDS]
        assert run.resolution.trials[-1].values == {
            slot.name: slot.serialized for slot in action.slots
        }


# %% runs kept apart


class TestRuns:
    def test_a_run_starts_from_a_copy_of_the_baseline_store(self, tmp_path):
        baseline = tmp_path / "store"
        (baseline / "record").mkdir(parents=True)
        (baseline / "record" / "content").write_text("stored")
        run = ResolutionRuns(directory=tmp_path / "runs").start(
            HypothesisStore(directory=baseline),
            RunEnvironment(scene="pouring", bundle=tmp_path / "pouring"),
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

    def test_a_run_keeps_the_environment_it_started_from(
        self, pouring_bundle, tmp_path
    ):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        kept = RunDirectory(directory=tmp_path / "runs" / run.resolution.run)
        assert kept.environment() == RunEnvironment(
            scene="pouring", bundle=pouring_bundle
        )

    def test_a_run_is_found_by_its_name(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        found = ResolutionRuns(directory=tmp_path / "runs").named(run.resolution.run)
        assert found.directory == tmp_path / "runs" / run.resolution.run

    @pytest.mark.parametrize("name", ["no-such-run", "../store", ""])
    def test_a_name_that_is_no_run_is_refused(self, tmp_path, name):
        (tmp_path / "runs").mkdir()
        with pytest.raises(UnknownRun):
            ResolutionRuns(directory=tmp_path / "runs").named(name)

    def test_a_run_is_reopened_with_what_it_resolved(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        reopened = reopening_resolver(tmp_path).reopen(run.resolution.run)
        assert reopened.resolution == run.resolution

    def test_a_reopened_run_consults_its_own_store(self, pouring_bundle, tmp_path):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        reopened = reopening_resolver(tmp_path).reopen(run.resolution.run)
        assert reopened.knowledge.backend.store == run.knowledge.backend.store
        assert reopened.knowledge.backend.store.records() == (
            run.knowledge.backend.store.records()
        )

    def test_a_run_that_is_not_kept_cannot_be_reopened(self, tmp_path):
        (tmp_path / "runs").mkdir()
        with pytest.raises(UnknownRun):
            reopening_resolver(tmp_path).reopen("no-such-run")

    def test_a_run_keeps_its_resolution_beside_its_knowledge(
        self, pouring_bundle, tmp_path
    ):
        (run,) = resolve_objects_given(pouring_bundle, tmp_path)
        kept = RunDirectory(directory=tmp_path / "runs" / run.resolution.run)
        assert (kept.directory / RunFile.RESOLUTION).is_file()
        assert kept.resolution() == run.resolution
