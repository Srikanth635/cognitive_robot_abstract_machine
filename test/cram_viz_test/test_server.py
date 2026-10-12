"""
End-to-end tests of the HTTP server: static frontend, scenes and JSON API.
"""

import importlib
import json
import shutil
import threading
import urllib.request

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


class TestDesignators:
    def test_a_scene_with_a_pouring_vessel_offers_every_designator(
        self, pouring_server
    ):
        from cram_viz.resolution.designators import POURING_VARIANTS

        payload = get_json(pouring_server + "/api/designators?scene=pouring")
        assert [entry["identifier"] for entry in payload["designators"]] == [
            variant.identifier for variant in POURING_VARIANTS
        ]
        assert all(entry["available"] for entry in payload["designators"])

    def test_a_scene_without_a_pouring_vessel_says_what_it_lacks(self, pouring_server):
        from semantic_digital_twin.semantic_annotations.semantic_annotations import (
            PouringCup,
        )

        payload = get_json(pouring_server + "/api/designators?scene=bowl_only")
        assert all(
            not entry["available"] and PouringCup.__name__ in entry["missing"]
            for entry in payload["designators"]
        )

    def test_resolving_a_designator_returns_its_resolution(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "vessel_into_bowl", HypothesisStore(directory=tmp_path / "store")
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {"variant": "vessel_into_bowl", "scene": "pouring"},
        )
        (action,) = payload["resolution"]["actions"]
        assert {slot["name"]: slot["value"] for slot in action["slots"]}[
            "target_container"
        ] == "bowl.stl"
        assert payload["hypothesis"]["ok"] is True

    def test_after_resolving_the_designator_tab_shows_why(
        self, pouring_server, monkeypatch, tmp_path
    ):
        from llm_reasoner.knowledge.access import HypothesisStore

        from cram_viz import server as server_module

        from .resolutions import resolver_replaying

        resolver = resolver_replaying(
            "vessel_into_bowl", HypothesisStore(directory=tmp_path / "store")
        )
        monkeypatch.setattr(server_module, "_designator_resolver", lambda: resolver)
        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {"variant": "vessel_into_bowl", "scene": "pouring"},
        )
        view = get_json(pouring_server + "/api/kb/view?name=designator&scene=pouring")
        keys = [entry["key"] for entry in view["designator"]["properties"]]
        assert "why" in keys
        assert payload["resolution"]["actions"][0]["why"] in json.dumps(view)

    def test_an_unknown_designator_is_reported(self, pouring_server):
        payload = post_json(
            pouring_server + "/api/designator/resolve",
            {"variant": "no_such_designator", "scene": "pouring"},
        )
        assert payload["ok"] is False
        assert "no_such_designator" in payload["error"]
