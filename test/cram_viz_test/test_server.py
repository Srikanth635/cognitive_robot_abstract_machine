"""
End-to-end tests of the HTTP server: static frontend, scenes and JSON API.
"""

import importlib
import json
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture()
def server(fixture_scene):
    """
    The real server on an ephemeral port, bound to the fixture scene.
    """
    from cram_viz import server as server_module

    importlib.reload(server_module)  # rebind kb_module under the fixture env
    httpd = server_module.make_server(0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://localhost:%d" % httpd.server_address[1]
    httpd.shutdown()


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def get_json(url):
    status, body = get(url)
    assert status == 200
    return json.loads(body)


class TestStatic:
    def test_index_is_served(self, server):
        status, body = get(server + "/")
        assert status == 200
        assert b"CRAM Vizulation" in body
        assert b'data-slot="left"' in body

    def test_panel_scripts_are_served(self, server):
        for path in (
            "/core/bus.js",
            "/core/registry.js",
            "/config.js",
            "/panels/robot_scene/panel.js",
            "/panels/eql/panel.js",
            "/panels/graph/panel.js",
            "/panels/graph/graph.js",
        ):
            status, _ = get(server + path)
            assert status == 200, path

    def test_scene_bundle_is_served_from_data_dir(self, server):
        scene = get_json(server + "/scenes/fixture/scene.json")
        assert scene["name"] == "fixture"
        index = get_json(server + "/scenes/index.json")
        assert index["default"] == "fixture"

    def test_scene_path_traversal_is_blocked(self, server):
        request = urllib.request.Request(server + "/scenes/../../etc/passwd")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as err:
            status = err.code
        assert status in (403, 404)


class TestApi:
    def test_kb_overview(self, server):
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/kb")
        assert payload["ok"]
        assert any(n["id"] == "milk" for n in payload["nodes"])

    def test_kb_views(self, server):
        pytest.importorskip("krrood")
        for name, expect_live in (
            ("kinematics", None),
            ("plan", "plan"),
            ("chart", "chart"),
        ):
            payload = get_json(server + "/api/kb/view?name=" + name)
            assert payload["ok"], name
            assert payload.get("live") == expect_live

    def test_eql_query_roundtrip(self, server):
        pytest.importorskip("krrood")
        request = urllib.request.Request(
            server + "/api/eql",
            data=json.dumps(
                {"code": "the(entity(obj).where(obj.name == 'milk'))"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["ok"] and payload["count"] == 1

    def test_broken_query_returns_json_error(self, server):
        pytest.importorskip("krrood")
        request = urllib.request.Request(
            server + "/api/eql",
            data=json.dumps({"code": "definitely not python ((("}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is False and "error" in payload

    def test_hypothesis_generation_rejects_an_empty_instruction(self, server):
        # validation happens before any model client is built, so this needs no
        # credentials and is safe to run in CI
        request = urllib.request.Request(
            server + "/api/hypothesis/generate",
            data=json.dumps({"instruction": "   "}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is False
        assert "instruction" in payload["error"]

    def test_execute_launches_the_hardcoded_demo(self, server, monkeypatch):
        # the real launcher is replaced so this never starts a real subprocess or
        # binds the live bridge's port
        from cram_viz import server as server_module

        launched = []

        class FakeLauncher:
            def launch(self, demo):
                launched.append(demo)

        monkeypatch.setattr(server_module, "_live_demo_launcher", FakeLauncher)
        request = urllib.request.Request(server + "/api/execute", data=b"{}")

        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())

        assert payload["ok"] is True
        assert launched == [server_module.paths.default_live_demo()]

    def test_execute_relaunches_over_a_still_running_demo(self, server, monkeypatch):
        # the real launcher already restarts instead of refusing (see
        # test_live_demo_launcher.py); this just confirms the route reports the
        # relaunch as a success rather than translating it into an error
        from cram_viz import server as server_module

        launched = []

        class RelaunchingLauncher:
            def launch(self, demo):
                launched.append(demo)

        monkeypatch.setattr(server_module, "_live_demo_launcher", RelaunchingLauncher)
        request = urllib.request.Request(server + "/api/execute", data=b"{}")

        with urllib.request.urlopen(request, timeout=10) as response:
            first_payload = json.loads(response.read())
        with urllib.request.urlopen(request, timeout=10) as response:
            second_payload = json.loads(response.read())

        assert first_payload["ok"] is True and second_payload["ok"] is True
        assert launched == [server_module.paths.default_live_demo()] * 2

    def test_execute_runs_the_named_run_in_the_simulation(
        self, server, monkeypatch, tmp_path
    ):
        from cram_viz import server as server_module
        from cram_viz.resolution.runs import RunFile

        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path))
        run = tmp_path / "runs" / "a-run"
        run.mkdir(parents=True)
        (run / RunFile.RESOLUTION).write_text("{}")
        launched = []

        class FakeLauncher:
            def launch_run(self, run_directory):
                launched.append(run_directory.directory)

        monkeypatch.setattr(server_module, "_live_demo_launcher", FakeLauncher)
        request = urllib.request.Request(
            server + "/api/execute", data=json.dumps({"run": "a-run"}).encode()
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())

        assert payload["ok"] is True
        assert launched == [run]

    def test_executing_an_unknown_run_is_reported(self, server, monkeypatch, tmp_path):
        from cram_viz.resolution.runs import UnknownRun

        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path))
        request = urllib.request.Request(
            server + "/api/execute", data=json.dumps({"run": "no-such-run"}).encode()
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())

        assert payload["ok"] is False
        assert payload["error"].startswith(UnknownRun.__name__)

    def test_unknown_post_endpoint_is_json_404(self, server):
        request = urllib.request.Request(server + "/api/nope", data=b"{}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as err:
            status, body = err.code, err.read()
        assert status == 404
        assert json.loads(body)["ok"] is False


# %% Scene-scoped knowledge
class TestSceneScopedKnowledge:
    def test_requested_scene_determines_the_knowledge_episodes(
        self, server, fixture_scene
    ):
        """
        Ensure the knowledge graph follows the scene selected by the browser.
        """
        source_directory = fixture_scene / "scenes" / "fixture"
        pouring_directory = fixture_scene / "scenes" / "pouring"
        pouring_directory.mkdir()
        scene = json.loads((source_directory / "scene.json").read_text())
        scene["name"] = "pouring"
        scene["segments"] = [
            {
                "step": "pouring_bowl",
                "action": "PouringAction",
                "arm": "left",
                "start": 0,
                "end": 3,
            }
        ]
        scene["planTrees"] = []
        (pouring_directory / "scene.json").write_text(json.dumps(scene))
        for filename in ("trajectory.json", "robot.urdf"):
            (pouring_directory / filename).write_bytes(
                (source_directory / filename).read_bytes()
            )

        payload = get_json(server + "/api/kb?scene=pouring")
        node_ids = {node["id"] for node in payload["nodes"]}

        assert "pouring_bowl" in node_ids
        assert "transport_milk" not in node_ids


# %% predefined designators


@pytest.fixture()
def pouring_server(fixture_scene, pouring_bundle):
    """
    The real server on an ephemeral port, offering the fixture scene and, beside it, a
    scene with a bowl and a pouring vessel and one with the bowl alone.
    """
    from cram_viz import server as server_module
    from cram_viz.resolution.scene_world import SCENE_FILE, SceneKey

    scenes = fixture_scene / "scenes"
    shutil.copytree(pouring_bundle, scenes / "pouring")
    shutil.copytree(pouring_bundle, scenes / "bowl_only")
    scene_file = scenes / "bowl_only" / SCENE_FILE
    scene = json.loads(scene_file.read_text())
    scene[SceneKey.OBJECTS] = [
        entry
        for entry in scene[SceneKey.OBJECTS]
        if entry[SceneKey.IDENTIFIER] != "Static_MilkPitcher"
    ]
    scene_file.write_text(json.dumps(scene))
    (scenes / "index.json").write_text(
        json.dumps(
            {"default": "fixture", "scenes": ["fixture", "pouring", "bowl_only"]}
        )
    )
    importlib.reload(server_module)
    httpd = server_module.make_server(0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://localhost:%d" % httpd.server_address[1]
    httpd.shutdown()


def post_json(url, payload, timeout=60):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


QUERY_SOURCES = Path(__file__).parent / "query_sources"
"""
The query sources the tests read.
"""


def objects_given_query() -> str:
    """
    :return: The match query of the designator giving the objects, as its chip copies it
        into the search bar.
    """
    from cram_viz.resolution.designators import DesignatorQuery, PouringDesignator

    return DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN).source.text


class TestDesignators:
    def test_a_scene_with_a_pouring_cup_offers_every_designator(self, pouring_server):
        from cram_viz.resolution.designators import PouringDesignator

        payload = get_json(pouring_server + "/api/designators?scene=pouring")
        assert [entry["identifier"] for entry in payload["designators"]] == list(
            PouringDesignator
        )
        assert all(entry["available"] for entry in payload["designators"])

    def test_a_designator_is_offered_as_its_match_query(self, pouring_server):
        from cram_viz.resolution.designators import DesignatorQuery, PouringDesignator

        payload = get_json(pouring_server + "/api/designators?scene=pouring")
        assert [entry["query"] for entry in payload["designators"]] == [
            DesignatorQuery.shipped(identifier).source.text
            for identifier in PouringDesignator
        ]

    def test_a_scene_without_a_pouring_cup_offers_none(self, pouring_server):
        payload = get_json(pouring_server + "/api/designators?scene=bowl_only")
        assert not any(entry["available"] for entry in payload["designators"])

    def test_resolving_a_designator_returns_its_resolution(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {"query": objects_given_query(), "scene": "pouring"},
        )
        (action,) = payload["resolution"]["actions"]
        assert {slot["name"]: slot["value"] for slot in action["slots"]}[
            "target_container"
        ] == "bowl.stl"
        assert payload["hypothesis"]["ok"] is True

    def test_a_resolution_links_to_the_viewer_of_its_run(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import paths
        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {"query": objects_given_query(), "scene": "pouring"},
        )
        assert payload["viewer"] == paths.viewer_url(payload["resolution"]["run"])

    def test_a_run_is_reopened_with_its_resolution_and_hypothesis(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import paths
        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        resolved = post_json(
            pouring_server + "/api/designator/resolve",
            {"query": objects_given_query(), "scene": "pouring"},
        )
        run = resolved["resolution"]["run"]
        reopened = get_json(
            pouring_server + f"/api/designator/run?name={run}&scene=pouring"
        )
        assert reopened["ok"] is True
        assert reopened["resolution"] == resolved["resolution"]
        assert reopened["hypothesis"]["ok"] is True
        assert reopened["viewer"] == paths.viewer_url(run)

    def test_a_run_that_is_not_kept_is_reported_when_reopened(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module
        from cram_viz.resolution.runs import UnknownRun

        from .resolutions import resolver_replaying

        (tmp_path / "runs").mkdir()
        resolver = resolver_replaying(
            "objects_given",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        payload = get_json(
            pouring_server + "/api/designator/run?name=no-such-run&scene=pouring"
        )
        assert payload["ok"] is False
        assert payload["error"].startswith(UnknownRun.__name__)

    def test_a_resolution_names_the_run_it_was_made_in(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {"query": objects_given_query(), "scene": "pouring"},
        )
        assert [path.name for path in (tmp_path / "runs").iterdir()] == [
            payload["resolution"]["run"]
        ]

    def test_after_resolving_the_designator_tab_shows_why(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {"query": objects_given_query(), "scene": "pouring"},
        )
        view = get_json(pouring_server + "/api/kb/view?name=designator&scene=pouring")
        keys = [entry["key"] for entry in view["designator"]["properties"]]
        assert "why" in keys
        assert payload["resolution"]["actions"][0]["why"] in json.dumps(view)

    def test_a_query_that_is_not_a_match_is_reported(self, pouring_server):
        from cram_viz.resolution.designators import NotAMatchQuery

        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {
                "query": (QUERY_SOURCES / "not_a_match.py").read_text(),
                "scene": "pouring",
            },
        )
        assert payload["ok"] is False
        assert payload["error"].startswith(NotAMatchQuery.__name__)


# %% consulting the perspectives the request chooses


class TestChosenPerspectives:
    def test_the_scene_offers_every_perspective_the_knowledge_registers(
        self, pouring_server
    ):
        from llm_reasoner.knowledge.access import PERSPECTIVE_ROOTS

        payload = get_json(pouring_server + "/api/designators?scene=pouring")
        assert [entry["name"] for entry in payload["perspectives"]] == [
            module.value for module in PERSPECTIVE_ROOTS
        ]
        assert [entry["summary"] for entry in payload["perspectives"]] == [
            module.summary for module in PERSPECTIVE_ROOTS
        ]

    def test_by_default_no_perspective_is_consulted(self, monkeypatch, tmp_path):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given", HypothesisStore(directory=tmp_path / "store"), tmp_path
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        assert server_module._resolver_for({}).perspectives == ()

    def test_the_perspectives_the_request_chooses_are_consulted(
        self, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import ActionPerspectiveFile, HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given", HypothesisStore(directory=tmp_path / "store"), tmp_path
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        chosen = (ActionPerspectiveFile.IMAGE_SCHEMAS, ActionPerspectiveFile.FAILURES)
        consulting = server_module._resolver_for(
            {"perspectives": [module.value for module in chosen]}
        )
        assert consulting.perspectives == chosen

    def test_a_perspective_the_knowledge_does_not_register_is_refused(
        self, pouring_server
    ):
        from cram_viz.resolution.perspective_choice import UnknownPerspective

        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {
                "query": objects_given_query(),
                "scene": "pouring",
                "perspectives": ["action-core.ttl"],
            },
        )
        assert payload["ok"] is False
        assert payload["error"].startswith(UnknownPerspective.__name__)

    def test_the_scene_offers_the_phase_structure_to_choose_too(self, pouring_server):
        import inspect

        from llm_reasoner.knowledge.access import core

        payload = get_json(pouring_server + "/api/designators?scene=pouring")
        assert payload["phase_structure"]["summary"] == inspect.getdoc(
            core.ActionPhaseStructure
        )

    def test_by_default_the_phase_structure_is_consulted(self, monkeypatch, tmp_path):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given", HypothesisStore(directory=tmp_path / "store"), tmp_path
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        assert server_module._resolver_for({}).consults_phase_structure is True

    def test_the_request_may_leave_out_the_phase_structure(self, monkeypatch, tmp_path):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "objects_given", HypothesisStore(directory=tmp_path / "store"), tmp_path
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        consulting = server_module._resolver_for({"phase_structure": False})
        assert consulting.consults_phase_structure is False
