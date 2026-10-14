"""
Tests of the viewer's server: the runs it lists, the events of a run as a batch or as a
stream that follows the log, and the pages it serves.
"""

from __future__ import annotations

import http.client
import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from typing_extensions import Any, Dict, Iterator, List

from llm_reasoner_viz.paths import (
    CRAM_VIZ_ENVIRONMENT,
    DEFAULT_CRAM_VIZ_URL,
    DATA_ENVIRONMENT,
    RUNS_ENVIRONMENT,
    default_runs_directory,
)
from llm_reasoner_viz.runs import EventLog, RunLibrary
from llm_reasoner_viz.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WEB_ROOT,
    make_server,
    parse_arguments,
)

from .logs import answered, append, finished, started, write_run

# %% helpers


@dataclass
class RunningViewer:
    """
    A viewer serving a directory of runs on a port of its own.
    """

    host: str
    """
    The address it listens on.
    """

    port: int
    """
    The port it listens on.
    """

    runs: Path
    """
    The directory of runs it serves.
    """

    @property
    def url(self) -> str:
        """
        The address of the viewer.
        """
        return f"http://{self.host}:{self.port}"


@pytest.fixture
def viewer(tmp_path: Path) -> Iterator[RunningViewer]:
    """
    :return: A viewer serving the empty directory of runs of the test.
    """
    server = make_server(RunLibrary(directory=tmp_path), host=DEFAULT_HOST, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield RunningViewer(host=DEFAULT_HOST, port=server.server_address[1], runs=tmp_path)
    server.shutdown()
    server.server_close()


def get(url: str) -> urllib.request.addinfourl:
    """
    :return: The response to a request for the address.
    """
    return urllib.request.urlopen(url, timeout=5)


def get_json(url: str) -> Dict[str, Any]:
    """
    :return: The JSON a request for the address answers with.
    """
    with get(url) as response:
        return json.loads(response.read())


def status_of(url: str) -> int:
    """
    :return: The status a request for the address is answered with.
    """
    try:
        with get(url) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def frames_from(
    viewer: RunningViewer, path: str, count: int, headers: Dict[str, str] | None = None
) -> List[Dict[str, Any]]:
    """
    :return: The first events of a stream, read as they arrive.
    """
    connection = http.client.HTTPConnection(viewer.host, viewer.port, timeout=5)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        assert response.status == 200
        events: List[Dict[str, Any]] = []
        while len(events) < count:
            line = response.fp.readline().decode()
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
        return events
    finally:
        connection.close()


# %% the runs


class TestRuns:
    def test_the_runs_are_listed_as_the_library_sums_them_up(self, viewer):
        write_run(viewer.runs, "a-run", [started(), answered(), finished()])
        payload = get_json(viewer.url + "/api/runs")
        library = RunLibrary(directory=viewer.runs)
        assert payload["ok"] is True
        assert payload["runs"] == [run.to_json() for run in library.runs()]

    def test_no_runs_is_an_empty_list(self, viewer):
        assert get_json(viewer.url + "/api/runs")["runs"] == []


# %% the events of a run


class TestEvents:
    def test_the_events_of_a_run_come_in_one_batch(self, viewer):
        events = [started(), answered(), finished()]
        path = write_run(viewer.runs, "a-run", events)
        payload = get_json(viewer.url + "/api/runs/a-run/events")
        assert payload["events"] == events
        assert payload["next"] == path.stat().st_size
        assert payload["size"] == path.stat().st_size

    def test_a_batch_goes_on_from_the_offset_asked_for(self, viewer):
        first, second, third = started(), answered(), finished()
        path = write_run(viewer.runs, "a-run", [first, second])
        stopped_at = get_json(viewer.url + "/api/runs/a-run/events")["next"]
        append(path, [third])
        payload = get_json(viewer.url + f"/api/runs/a-run/events?after={stopped_at}")
        assert payload["events"] == [third]

    def test_a_run_that_is_not_kept_is_not_found(self, viewer):
        assert status_of(viewer.url + "/api/runs/no-such-run/events") == 404

    def test_an_offset_that_is_no_number_is_refused(self, viewer):
        write_run(viewer.runs, "a-run", [started()])
        assert status_of(viewer.url + "/api/runs/a-run/events?after=soon") == 400


# %% following a run


class TestStream:
    def test_the_events_so_far_are_streamed_first(self, viewer):
        events = [started(), answered(), finished()]
        write_run(viewer.runs, "a-run", events)
        assert frames_from(viewer, "/api/runs/a-run/stream", 3) == events

    def test_an_event_appended_while_streaming_arrives(self, viewer):
        path = write_run(viewer.runs, "a-run", [started()])
        later = answered()
        timer = threading.Timer(0.3, append, args=(path, [later]))
        timer.start()
        assert frames_from(viewer, "/api/runs/a-run/stream", 2) == [started(), later]
        timer.join()

    def test_a_stream_goes_on_from_the_offset_asked_for(self, viewer):
        events = [started(), answered(), finished()]
        path = write_run(viewer.runs, "a-run", events)
        first_end = EventLog(path=path).read_from(0).entries[0].end
        assert (
            frames_from(viewer, f"/api/runs/a-run/stream?after={first_end}", 2)
            == events[1:]
        )

    def test_a_stream_resumes_where_the_last_event_a_reader_saw_ended(self, viewer):
        events = [started(), answered(), finished()]
        path = write_run(viewer.runs, "a-run", events)
        first_end = EventLog(path=path).read_from(0).entries[0].end
        assert (
            frames_from(
                viewer,
                "/api/runs/a-run/stream",
                2,
                headers={"Last-Event-ID": str(first_end)},
            )
            == events[1:]
        )

    def test_a_stream_of_a_run_that_is_not_kept_is_not_found(self, viewer):
        assert status_of(viewer.url + "/api/runs/no-such-run/stream") == 404


# %% where cram-viz is


class TestConfiguration:
    def test_the_viewer_says_where_cram_viz_is(self, viewer, monkeypatch):
        monkeypatch.delenv(CRAM_VIZ_ENVIRONMENT, raising=False)
        payload = get_json(viewer.url + "/api/config")
        assert payload == {"ok": True, "cram_viz_url": DEFAULT_CRAM_VIZ_URL}

    def test_where_cram_viz_is_can_be_set_without_a_trailing_slash(
        self, viewer, monkeypatch
    ):
        monkeypatch.setenv(CRAM_VIZ_ENVIRONMENT, "http://elsewhere:9000/")
        payload = get_json(viewer.url + "/api/config")
        assert payload["cram_viz_url"] == "http://elsewhere:9000"


# %% the pages


class TestPages:
    def test_the_front_page_is_served(self, viewer):
        with get(viewer.url + "/") as response:
            assert response.status == 200
            assert "text/html" in response.headers["Content-Type"]
            assert b"<title>" in response.read()

    def test_the_script_of_the_page_is_served_as_javascript(self, viewer):
        with get(viewer.url + "/app.mjs") as response:
            assert "javascript" in response.headers["Content-Type"]

    @pytest.mark.parametrize(
        "module", sorted(WEB_ROOT.glob("*.mjs")), ids=lambda module: module.name
    )
    def test_every_module_of_the_page_is_served_as_javascript(self, viewer, module):
        with get(viewer.url + "/" + module.name) as response:
            assert "javascript" in response.headers["Content-Type"]

    def test_a_page_that_is_not_there_is_not_found(self, viewer):
        assert status_of(viewer.url + "/no-such-page.html") == 404

    def test_a_file_outside_the_pages_is_not_served(self, viewer):
        assert status_of(viewer.url + "/../server.py") == 404
        assert status_of(viewer.url + "/%2e%2e/server.py") == 404

    def test_the_front_page_refers_only_to_files_that_exist(self):
        page = (WEB_ROOT / "index.html").read_text()
        local = [
            reference
            for reference in re.findall(r'(?:src|href)="([^"]+)"', page)
            if not reference.startswith(("http", "#", "/api"))
        ]
        assert local
        assert all((WEB_ROOT / reference.lstrip("/")).is_file() for reference in local)


# %% starting the viewer


class TestStarting:
    def test_by_default_the_viewer_listens_on_its_own_port_of_this_machine(self):
        arguments = parse_arguments([])
        assert arguments.port == DEFAULT_PORT
        assert arguments.host == DEFAULT_HOST
        assert arguments.runs == default_runs_directory()

    def test_the_port_and_the_runs_can_be_asked_for(self, tmp_path: Path):
        arguments = parse_arguments(["--port", "9100", "--runs", str(tmp_path)])
        assert arguments.port == 9100
        assert arguments.runs == tmp_path


class TestRunsDirectory:
    def test_a_directory_named_in_the_environment_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv(RUNS_ENVIRONMENT, str(tmp_path / "mine"))
        assert default_runs_directory() == tmp_path / "mine"

    def test_otherwise_the_runs_are_where_cram_viz_keeps_them(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv(RUNS_ENVIRONMENT, raising=False)
        monkeypatch.setenv(DATA_ENVIRONMENT, str(tmp_path))
        assert default_runs_directory() == tmp_path / "runs"

    def test_without_either_they_are_in_the_users_data_directory(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv(RUNS_ENVIRONMENT, raising=False)
        monkeypatch.delenv(DATA_ENVIRONMENT, raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_runs_directory() == tmp_path / ".cram_viz" / "runs"
