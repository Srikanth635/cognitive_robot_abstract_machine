"""
Tests of the cramera viewer served with the resolution add-on: its page, its panels and
its routes, and that every request it does not answer is cramera's own.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from cramera import paths
from cramera.server import ViewerServer

from cramera_resolution.exceptions import IndexPageHasNoBootScript
from cramera_resolution.server import (
    ADDON_SCRIPTS,
    ADDON_URL_PREFIX,
    IndexPage,
    ResolutionServer,
    Route,
)

# %% helpers


@pytest.fixture(autouse=True)
def isolated_cramera_directories(monkeypatch, tmp_path):
    """
    No scene bundles and an empty data directory, so the viewer reads nothing of the
    machine it runs on.
    """
    monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "scenes"))
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))


@contextmanager
def serving(server_class):
    """
    :return: The address of a server of the class, answering on a free port while the
        block runs.
    """
    with server_class(0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join()


def fetched(address: str, path: str) -> str:
    """
    :return: What the server answers for the path, as text.
    """
    with urlopen(address + path) as response:
        return response.read().decode("utf-8")


def status_of(address: str, path: str) -> int:
    """
    :return: The status the server answers the path with.
    """
    try:
        with urlopen(address + path) as response:
            return response.status
    except HTTPError as refused:
        return refused.code


CRAMERA_PAGE = (paths.WEB_ROOT / "index.html").read_text(encoding="utf-8")
"""
The page as cramera ships it.
"""


# %% the page


class TestIndexPage:
    def test_the_add_on_scripts_come_before_the_script_that_mounts_the_panels(self):
        page = IndexPage(html=CRAMERA_PAGE).with_scripts(ADDON_SCRIPTS)
        boot = page.index(IndexPage.BOOT_MARKER)
        assert all(page.index(source) < boot for source in ADDON_SCRIPTS)

    def test_nothing_but_the_script_tags_is_added(self):
        page = IndexPage(html=CRAMERA_PAGE).with_scripts(ADDON_SCRIPTS)
        without_tags = page
        for source in ADDON_SCRIPTS:
            without_tags = without_tags.replace(
                f'<script src="{source}"></script>\n', ""
            )
        assert without_tags == CRAMERA_PAGE

    def test_a_page_with_no_script_to_boot_the_panels_is_refused(self):
        with pytest.raises(IndexPageHasNoBootScript):
            IndexPage(html="<html></html>").with_scripts(ADDON_SCRIPTS)


# %% the served viewer


class TestServedViewer:
    @pytest.mark.parametrize("route", [Route.INDEX, Route.INDEX_FILE])
    def test_the_page_served_carries_the_add_on_scripts(self, route):
        with serving(ResolutionServer) as address:
            page = fetched(address, route)
        assert page == IndexPage(html=CRAMERA_PAGE).with_scripts(ADDON_SCRIPTS)

    def test_an_add_on_script_is_served(self):
        with serving(ResolutionServer) as address:
            script = fetched(
                address, ADDON_URL_PREFIX + ADDON_SCRIPTS[0].split("/", 1)[1]
            )
        assert "Panels.define" in script

    def test_a_file_outside_the_add_on_is_refused(self):
        with serving(ResolutionServer) as address:
            status = status_of(address, ADDON_URL_PREFIX + "../server.py")
        assert status == 404

    def test_the_add_on_answers_its_own_route(self):
        with serving(ResolutionServer) as address:
            answer = json.loads(fetched(address, Route.PING))
        assert answer["ok"] is True


class TestEveryOtherRequestIsCrameras:
    @pytest.mark.parametrize("path", ["/scenes/index.json", "/api/recording/status"])
    def test_a_route_of_cramera_answers_as_it_does_alone(self, path):
        with serving(ViewerServer) as alone, serving(ResolutionServer) as extended:
            assert fetched(extended, path) == fetched(alone, path)

    def test_a_file_of_cramera_is_served_as_it_is(self):
        with serving(ResolutionServer) as address:
            served = fetched(address, "/core/registry.js")
        assert served == (paths.WEB_ROOT / "core" / "registry.js").read_text(
            encoding="utf-8"
        )

    def test_an_unknown_route_is_refused_as_cramera_refuses_it(self):
        with serving(ViewerServer) as alone, serving(ResolutionServer) as extended:
            assert status_of(extended, "/api/unknown") == status_of(
                alone, "/api/unknown"
            )
