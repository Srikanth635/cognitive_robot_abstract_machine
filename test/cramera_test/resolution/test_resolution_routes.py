"""
Tests of the resolution routes over the real service, on scenes with a bowl and a
pouring vessel: the designators a scene offers, resolving one with a scripted model,
showing a run again, choosing what to consult, generating knowledge and performing a
run.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from llm_reasoner.attribute_docstrings import attribute_docstrings
from llm_reasoner.knowledge.access import (
    PERSPECTIVE_ROOTS,
    ActionPerspectiveFile,
    HypothesisStore,
    core,
)
from semantic_digital_twin.robots.robot_parts import Arm
from typing_extensions import Any

from cramera.resolution import paths
from cramera.resolution.exceptions import EmptyInstruction, NoPerformanceWaiting
from cramera.resolution.payload import PayloadKey
from cramera.resolution.resolver.action_vocabulary import (
    VocabularyKey,
    VocabularyKind,
)
from cramera.resolution.resolver.designators import (
    DesignatorQuery,
    NotAMatchQuery,
    PouringDesignator,
)
from cramera.resolution.resolver.perspective_choice import UnknownPerspective
from cramera.resolution.resolver.resolver import DesignatorResolver
from cramera.resolution.resolver.runs import RunFile, UnknownRun
from cramera.resolution.query_source import QuerySource
from cramera.resolution.resolver.scene_world import SCENE_FILE, SceneKey
from cramera.resolution.routes import ResolutionRoute as Route
from cramera.server import ViewerServer
from cramera.resolution.service import ResolutionService

from .resolutions import resolver_replaying

# %% helpers


@dataclass
class ServedResolution:
    """
    The viewer's server, serving, and the service behind its routes.
    """

    address: str
    """
    Where the server answers.
    """

    service: ResolutionService
    """
    What the routes run, whose resolver a test may replace.
    """

    def get(self, route: str, **query: str) -> dict[str, Any]:
        """
        :return: What the server answers a read of the route with.
        """
        parameters = "&".join(f"{name}={value}" for name, value in query.items())
        with urlopen(f"{self.address}{route}?{parameters}", timeout=60) as response:
            return json.loads(response.read())

    def post(self, route: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        :return: What the server answers the body posted to the route with.
        """
        request = Request(self.address + route, data=json.dumps(body).encode())
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read())


@pytest.fixture
def served(tmp_path, pouring_bundle_with_arms) -> ServedResolution:
    """
    The viewer's server over scenes that hold a pouring vessel and a bowl, and over one
    with the bowl alone.
    """
    scenes = tmp_path / "scenes"
    shutil.copytree(pouring_bundle_with_arms, scenes / "pouring")
    shutil.copytree(pouring_bundle_with_arms, scenes / "bowl_only")
    scene_file = scenes / "bowl_only" / SCENE_FILE
    scene = json.loads(scene_file.read_text())
    scene[SceneKey.OBJECTS] = [
        entry
        for entry in scene[SceneKey.OBJECTS]
        if entry[SceneKey.IDENTIFIER] != "Static_MilkPitcher"
    ]
    scene_file.write_text(json.dumps(scene))
    with ViewerServer(0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield ServedResolution(
            address=f"http://127.0.0.1:{server.server_address[1]}",
            service=server.resolution,
        )
        server.shutdown()
        thread.join()


@pytest.fixture
def scripted_resolver(tmp_path) -> DesignatorResolver:
    """
    A resolver replaying the recorded answers to the designator giving the objects.
    """
    return resolver_replaying(
        PouringDesignator.OBJECTS_GIVEN,
        HypothesisStore(directory=tmp_path / "store"),
        tmp_path / "runs",
    )


@pytest.fixture
def resolver_keeping_runs_in_the_data_directory(tmp_path) -> DesignatorResolver:
    """
    A resolver replaying the recorded answers twice, keeping its runs where the
    resolution feature lists them from.
    """
    return resolver_replaying(
        PouringDesignator.OBJECTS_GIVEN,
        HypothesisStore(directory=tmp_path / "store"),
        paths.resolution_runs_dir(),
        times=2,
    )


@pytest.fixture
def objects_given_query() -> str:
    """
    The match query of the designator giving the objects, as its chip copies it into the
    search bar.
    """
    return DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN).source.text


@pytest.fixture
def query_sources() -> Path:
    """
    The directory of the query sources the tests read.
    """
    return Path(__file__).parent / "query_sources"


# %% the designators a scene offers


class TestDesignators:
    def test_a_scene_with_a_pouring_cup_offers_every_designator(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        assert [
            entry["identifier"] for entry in payload[PayloadKey.DESIGNATORS]
        ] == list(PouringDesignator)
        assert all(entry["available"] for entry in payload[PayloadKey.DESIGNATORS])

    def test_a_designator_is_offered_as_its_match_query(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        assert [entry["query"] for entry in payload[PayloadKey.DESIGNATORS]] == [
            DesignatorQuery.shipped(identifier).source.text
            for identifier in PouringDesignator
        ]

    def test_a_scene_without_a_pouring_cup_offers_none(self, served):
        payload = served.get(Route.DESIGNATORS, scene="bowl_only")
        assert not any(entry["available"] for entry in payload[PayloadKey.DESIGNATORS])

    def test_the_scene_offers_every_perspective_the_knowledge_registers(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        assert [
            (entry["name"], entry["summary"])
            for entry in payload[PayloadKey.PERSPECTIVES]
        ] == [(module.value, module.summary) for module in PERSPECTIVE_ROOTS]

    def test_the_scene_offers_the_phase_structure_to_choose_too(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        assert payload[PayloadKey.PHASE_STRUCTURE]["summary"] == inspect.getdoc(
            core.ActionPhaseStructure
        )

    def test_the_scene_offers_adjusting_the_pour_geometry_too(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        assert (
            payload[PayloadKey.GEOMETRY_ADJUSTMENT]["summary"]
            == attribute_docstrings(DesignatorResolver)["adjusts_pour_geometry"]
        )

    def test_the_scene_offers_the_action_designators_to_complete(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        assert [
            entry[VocabularyKey.NAME]
            for entry in payload[PayloadKey.ACTION_VOCABULARY]
            if entry[VocabularyKey.KIND] == VocabularyKind.CLASS
        ] == [PouringAction.__name__]

    def test_the_parameters_of_an_action_designator_are_offered_with_it(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        assert [
            entry[VocabularyKey.NAME]
            for entry in payload[PayloadKey.ACTION_VOCABULARY]
            if entry[VocabularyKey.KIND] == VocabularyKind.FIELD
            and entry[VocabularyKey.OWNER] == PouringAction.__name__
        ] == [field.name for field in dataclasses.fields(PouringAction) if field.init]

    def test_a_parameter_with_a_default_states_it(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        (tilt,) = [
            entry
            for entry in payload[PayloadKey.ACTION_VOCABULARY]
            if entry[VocabularyKey.NAME] == "tilt_angle"
        ]
        default = next(
            field.default
            for field in dataclasses.fields(PouringAction)
            if field.name == "tilt_angle"
        )
        assert tilt[VocabularyKey.DETAIL].endswith(f"= {default}")

    def test_a_parameter_without_a_default_states_only_its_type(self, served):
        payload = served.get(Route.DESIGNATORS, scene="pouring")
        (arm,) = [
            entry
            for entry in payload[PayloadKey.ACTION_VOCABULARY]
            if entry[VocabularyKey.NAME] == "arm"
        ]
        assert arm[VocabularyKey.DETAIL] == Arm.__name__

    def test_a_scene_that_is_not_held_is_reported(self, served):
        payload = served.get(Route.DESIGNATORS, scene="no-such-scene")
        assert payload[PayloadKey.OK] is False


# %% resolving


class TestResolving:
    def test_resolving_a_designator_returns_its_resolution(
        self, served, scripted_resolver, objects_given_query
    ):
        served.service.resolver = scripted_resolver
        payload = served.post(
            Route.RESOLVE, {"query": objects_given_query, "scene": "pouring"}
        )
        (action,) = payload[PayloadKey.RESOLUTION]["actions"]
        assert {slot["name"]: slot["value"] for slot in action["slots"]}[
            "target_container"
        ] == "bowl.stl"
        assert payload[PayloadKey.HYPOTHESIS][PayloadKey.OK] is True

    def test_a_resolution_links_to_the_viewer_of_its_run(
        self, served, scripted_resolver, objects_given_query
    ):
        served.service.resolver = scripted_resolver
        payload = served.post(
            Route.RESOLVE, {"query": objects_given_query, "scene": "pouring"}
        )
        assert payload[PayloadKey.VIEWER] == paths.viewer_url(
            payload[PayloadKey.RESOLUTION]["run"]
        )

    def test_a_resolution_names_the_run_it_was_made_in(
        self, served, scripted_resolver, objects_given_query, tmp_path
    ):
        served.service.resolver = scripted_resolver
        payload = served.post(
            Route.RESOLVE, {"query": objects_given_query, "scene": "pouring"}
        )
        assert [path.name for path in (tmp_path / "runs").iterdir()] == [
            payload[PayloadKey.RESOLUTION]["run"]
        ]

    def test_after_resolving_the_designator_view_shows_why(
        self, served, scripted_resolver, objects_given_query
    ):
        served.service.resolver = scripted_resolver
        resolved = served.post(
            Route.RESOLVE, {"query": objects_given_query, "scene": "pouring"}
        )
        view = served.get(Route.VIEW, name="designator", scene="pouring")
        keys = [entry["key"] for entry in view[PayloadKey.DESIGNATOR]["properties"]]
        assert "why" in keys
        assert resolved[PayloadKey.RESOLUTION]["actions"][0]["why"] in json.dumps(view)

    def test_a_query_that_is_not_a_match_is_reported(self, served, query_sources):
        payload = served.post(
            Route.RESOLVE,
            {
                "query": (query_sources / "not_a_match.py").read_text(),
                "scene": "pouring",
            },
        )
        assert payload[PayloadKey.OK] is False
        assert payload[PayloadKey.ERROR].startswith(NotAMatchQuery.__name__)

    def test_a_perspective_the_knowledge_does_not_register_is_refused(
        self, served, objects_given_query
    ):
        payload = served.post(
            Route.RESOLVE,
            {
                "query": objects_given_query,
                "scene": "pouring",
                "perspectives": ["action-core.ttl"],
            },
        )
        assert payload[PayloadKey.OK] is False
        assert payload[PayloadKey.ERROR].startswith(UnknownPerspective.__name__)


class TestShowingARunAgain:
    def test_a_run_is_reopened_with_its_resolution_and_hypothesis(
        self, served, scripted_resolver, objects_given_query
    ):
        served.service.resolver = scripted_resolver
        resolved = served.post(
            Route.RESOLVE, {"query": objects_given_query, "scene": "pouring"}
        )
        run = resolved[PayloadKey.RESOLUTION]["run"]
        reopened = served.get(Route.RUN, name=run, scene="pouring")
        assert reopened[PayloadKey.RESOLUTION] == resolved[PayloadKey.RESOLUTION]
        assert reopened[PayloadKey.HYPOTHESIS][PayloadKey.OK] is True
        assert reopened[PayloadKey.VIEWER] == paths.viewer_url(run)

    def test_a_run_that_is_not_kept_is_reported(
        self, served, scripted_resolver, tmp_path
    ):
        (tmp_path / "runs").mkdir()
        served.service.resolver = scripted_resolver
        payload = served.get(Route.RUN, name="no-such-run", scene="pouring")
        assert payload[PayloadKey.OK] is False
        assert payload[PayloadKey.ERROR].startswith(UnknownRun.__name__)


class TestKeptRuns:
    def test_no_run_is_listed_before_one_is_made(self, served):
        assert served.get(Route.RUNS)[PayloadKey.RUNS] == []

    def test_every_run_made_is_listed_newest_first(
        self, served, resolver_keeping_runs_in_the_data_directory, objects_given_query
    ):
        served.service.resolver = resolver_keeping_runs_in_the_data_directory
        request = {"query": objects_given_query, "scene": "pouring"}
        first = served.post(Route.RESOLVE, request)[PayloadKey.RESOLUTION]["run"]
        second = served.post(Route.RESOLVE, request)[PayloadKey.RESOLUTION]["run"]
        listed = served.get(Route.RUNS)[PayloadKey.RUNS]
        assert [run[PayloadKey.NAME] for run in listed] == [second, first]

    def test_a_run_is_listed_with_what_it_resolved(
        self, served, resolver_keeping_runs_in_the_data_directory, objects_given_query
    ):
        served.service.resolver = resolver_keeping_runs_in_the_data_directory
        resolution = served.post(
            Route.RESOLVE, {"query": objects_given_query, "scene": "pouring"}
        )[PayloadKey.RESOLUTION]
        (listed,) = served.get(Route.RUNS)[PayloadKey.RUNS]
        assert listed[PayloadKey.SCENE] == "pouring"
        assert listed[PayloadKey.DESIGNATOR] == resolution["designator"]
        assert (
            listed[PayloadKey.QUERY]
            == QuerySource(text=resolution["query"]).on_one_line
        )
        assert listed[PayloadKey.COST] == resolution["cost"]
        assert listed[PayloadKey.TRIALS] == len(resolution["trials"])


# %% what the request chooses to consult


class TestChosenKnowledge:
    def test_by_default_no_perspective_is_consulted(self, served, scripted_resolver):
        served.service.resolver = scripted_resolver
        assert served.service.resolver_for({}).perspectives == ()

    def test_the_perspectives_the_request_chooses_are_consulted(
        self, served, scripted_resolver
    ):
        served.service.resolver = scripted_resolver
        chosen = (ActionPerspectiveFile.IMAGE_SCHEMAS, ActionPerspectiveFile.FAILURES)
        consulting = served.service.resolver_for(
            {"perspectives": [module.value for module in chosen]}
        )
        assert consulting.perspectives == chosen

    def test_by_default_the_phase_structure_is_consulted(
        self, served, scripted_resolver
    ):
        served.service.resolver = scripted_resolver
        assert served.service.resolver_for({}).consults_phase_structure is True

    def test_the_request_may_leave_out_the_phase_structure(
        self, served, scripted_resolver
    ):
        served.service.resolver = scripted_resolver
        consulting = served.service.resolver_for({"phase_structure": False})
        assert consulting.consults_phase_structure is False

    def test_by_default_the_pour_geometry_is_adjusted(self, served, scripted_resolver):
        served.service.resolver = scripted_resolver
        assert served.service.resolver_for({}).adjusts_pour_geometry is True

    def test_the_request_may_leave_out_adjusting_the_pour_geometry(
        self, served, scripted_resolver
    ):
        served.service.resolver = scripted_resolver
        consulting = served.service.resolver_for({"adjust_geometry": False})
        assert consulting.adjusts_pour_geometry is False


# %% generating knowledge and performing a run


class TestGeneratingKnowledge:
    def test_an_empty_instruction_is_rejected_before_any_model_is_asked(self, served):
        payload = served.post(Route.GENERATE, {"instruction": "   "})
        assert payload[PayloadKey.OK] is False
        assert payload[PayloadKey.ERROR].startswith(EmptyInstruction.__name__)


class LauncherRecordingTheRuns:
    """
    A launcher that only records the runs it is asked to perform and to start.
    """

    def __init__(self):
        self.performed = []
        self.starts = 0

    def launch(self, run):
        self.performed.append(run.directory)

    def request_start(self):
        self.starts += 1


class TestPerformingARun:
    def test_the_named_run_is_performed(self, served, tmp_path):
        run = tmp_path / "data" / "runs" / "a-run"
        run.mkdir(parents=True)
        (run / RunFile.RESOLUTION).write_text("{}")
        served.service.launcher = LauncherRecordingTheRuns()
        payload = served.post(Route.EXECUTE, {"run": "a-run"})
        assert payload[PayloadKey.OK] is True
        assert served.service.launcher.performed == [run]

    def test_a_run_that_is_not_kept_is_reported(self, served):
        served.service.launcher = LauncherRecordingTheRuns()
        payload = served.post(Route.EXECUTE, {"run": "no-such-run"})
        assert payload[PayloadKey.OK] is False
        assert payload[PayloadKey.ERROR].startswith(UnknownRun.__name__)
        assert served.service.launcher.performed == []

    def test_a_request_naming_no_run_is_reported(self, served):
        served.service.launcher = LauncherRecordingTheRuns()
        payload = served.post(Route.EXECUTE, {})
        assert payload[PayloadKey.ERROR].startswith(UnknownRun.__name__)

    def test_a_performance_that_waits_is_started(self, served):
        served.service.launcher = LauncherRecordingTheRuns()
        payload = served.post(Route.START, {})
        assert payload[PayloadKey.OK] is True
        assert served.service.launcher.starts == 1

    def test_a_start_with_nothing_waiting_is_reported(self, served):
        payload = served.post(Route.START, {})
        assert payload[PayloadKey.OK] is False
        assert payload[PayloadKey.ERROR].startswith(NoPerformanceWaiting.__name__)
