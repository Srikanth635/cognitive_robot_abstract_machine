"""
Tests of the cramera viewer served with the resolution add-on: its page, its panels and
the plumbing of its routes, and that every request it does not answer is cramera's own.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from cramera import paths
from cramera.server import ViewerServer
from typing_extensions import Any

from cramera_resolution.exceptions import IndexPageHasNoBootScript, NoSuchScene
from cramera_resolution.paths import AddOnDirectory
from cramera_resolution.payload import PayloadKey
from cramera_resolution.server import (
    AddOnScript,
    AddOnStylesheet,
    AddOnUrl,
    IndexPage,
    PageMarker,
    ResolutionServer,
    Route,
)
from cramera_resolution.service import ResolutionService
from cramera_resolution.views import ViewName

# %% helpers


@pytest.fixture
def cramera_page() -> str:
    """
    The viewer's page as cramera ships it.
    """
    return (paths.WEB_ROOT / "index.html").read_text(encoding="utf-8")


@contextmanager
def serving(server: ViewerServer):
    """
    :param server: A server not yet serving.
    :return: The address of the server, answering while the block runs.
    """
    with server:
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


def answered(request: Request) -> tuple[int, dict[str, Any]]:
    """
    :return: The status and the JSON body the server answers the request with.
    """
    try:
        with urlopen(request) as response:
            return response.status, json.loads(response.read())
    except HTTPError as refused:
        return refused.code, json.loads(refused.read())


@dataclass
class ServiceRecordingItsCalls(ResolutionService):
    """
    A service that only records what it is asked, so the routes can be tested without a
    model, credentials or a simulation.
    """

    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    """
    What was asked, in order: the operation and its arguments.
    """

    def kept_runs(self):
        self.calls.append(("kept_runs", ()))
        return {PayloadKey.OK: True}

    def offered_designators(self, requested_scene):
        self.calls.append(("offered_designators", (requested_scene,)))
        return {PayloadKey.OK: True}

    def open_run(self, name, requested_scene):
        self.calls.append(("open_run", (name, requested_scene)))
        return {PayloadKey.OK: True}

    def view(self, name, requested_scene):
        self.calls.append(("view", (name, requested_scene)))
        return {PayloadKey.OK: True}

    def resolve(self, request):
        self.calls.append(("resolve", (request,)))
        return {PayloadKey.OK: True}

    def generate_hypothesis(self, request):
        self.calls.append(("generate_hypothesis", (request,)))
        return {PayloadKey.OK: True}

    def execute_run(self, request):
        self.calls.append(("execute_run", (request,)))
        return {PayloadKey.OK: True}


class ServiceRefusingEveryRequest(ResolutionService):
    """
    A service that fails whatever it is asked.
    """

    def offered_designators(self, requested_scene):
        raise NoSuchScene(requested=requested_scene)


@contextmanager
def serving_with(service: ResolutionService):
    """
    :param service: What the add-on's routes are to run.
    :return: The address of an add-on server over the service.
    """
    server = ResolutionServer(0)
    server.service = service
    with serving(server) as address:
        yield address


# %% the page


class TestIndexPage:
    def test_the_add_on_assets_come_before_the_script_that_mounts_the_panels(
        self, cramera_page
    ):
        page = IndexPage(html=cramera_page).with_assets(
            list(AddOnScript), list(AddOnStylesheet)
        )
        boot = page.index(PageMarker.BOOT_SCRIPT)
        assert all(
            page.index(asset) < boot for asset in (*AddOnScript, *AddOnStylesheet)
        )

    def test_the_scripts_run_in_the_order_they_are_given(self, cramera_page):
        page = IndexPage(html=cramera_page).with_assets(list(AddOnScript))
        positions = [page.index(script) for script in AddOnScript]
        assert positions == sorted(positions)

    def test_nothing_but_the_tags_is_added(self, cramera_page):
        page = IndexPage(html=cramera_page).with_assets(
            list(AddOnScript), list(AddOnStylesheet)
        )
        for script in AddOnScript:
            page = page.replace(f'<script src="{script}"></script>\n', "")
        for sheet in AddOnStylesheet:
            page = page.replace(f'<link rel="stylesheet" href="{sheet}">\n', "")
        assert page == cramera_page

    def test_a_page_with_no_script_to_boot_the_panels_is_refused(self):
        with pytest.raises(IndexPageHasNoBootScript):
            IndexPage(html="<html></html>").with_assets(list(AddOnScript))


# %% the served viewer


class TestServedViewer:
    @pytest.mark.parametrize("route", [Route.INDEX, Route.INDEX_FILE])
    def test_the_page_served_carries_the_add_on_scripts(self, route, cramera_page):
        with serving(ResolutionServer(0)) as address:
            page = fetched(address, route)
        assert page == IndexPage(html=cramera_page).with_assets(
            list(AddOnScript), list(AddOnStylesheet)
        )

    @pytest.mark.parametrize("asset", [*AddOnScript, *AddOnStylesheet])
    def test_an_add_on_asset_is_served_as_it_is_kept(self, asset):
        with serving(ResolutionServer(0)) as address:
            served = fetched(address, AddOnUrl.PREFIX + asset.removeprefix("addon/"))
        assert served == (AddOnDirectory.WEB / asset).read_text(encoding="utf-8")

    def test_a_file_outside_the_add_on_is_refused(self):
        with serving(ResolutionServer(0)) as address:
            status = status_of(address, AddOnUrl.PREFIX + "../server.py")
        assert status == 404

    def test_the_add_on_answers_its_own_route(self):
        with serving(ResolutionServer(0)) as address:
            answer = json.loads(fetched(address, Route.PING))
        assert answer[PayloadKey.OK] is True


class TestEveryOtherRequestIsCrameras:
    @pytest.mark.parametrize("path", ["/scenes/index.json", "/api/recording/status"])
    def test_a_route_of_cramera_answers_as_it_does_alone(self, path):
        with serving(ViewerServer(0)) as alone, serving(
            ResolutionServer(0)
        ) as extended:
            assert fetched(extended, path) == fetched(alone, path)

    def test_a_file_of_cramera_is_served_as_it_is(self):
        with serving(ResolutionServer(0)) as address:
            served = fetched(address, "/core/registry.js")
        assert served == (paths.WEB_ROOT / "core" / "registry.js").read_text(
            encoding="utf-8"
        )

    def test_an_unknown_route_is_refused_as_cramera_refuses_it(self):
        with serving(ViewerServer(0)) as alone, serving(
            ResolutionServer(0)
        ) as extended:
            assert status_of(extended, "/api/unknown") == status_of(
                alone, "/api/unknown"
            )


# %% the routes


class TestRoutesRunTheService:
    @pytest.mark.parametrize(
        ("route", "query", "operation", "arguments"),
        [
            (Route.RUNS, "", "kept_runs", ()),
            (Route.DESIGNATORS, "?scene=pouring", "offered_designators", ("pouring",)),
            (Route.DESIGNATORS, "", "offered_designators", (None,)),
            (Route.RUN, "?name=a-run&scene=pouring", "open_run", ("a-run", "pouring")),
            (
                Route.VIEW,
                "?name=ontology&scene=pouring",
                "view",
                (ViewName.ONTOLOGY, "pouring"),
            ),
            (Route.VIEW, "", "view", (ViewName.HYPOTHESIS, None)),
        ],
    )
    def test_a_read_route_asks_the_service_what_the_request_names(
        self, route, query, operation, arguments
    ):
        service = ServiceRecordingItsCalls()
        with serving_with(service) as address:
            fetched(address, route + query)
        assert service.calls == [(operation, arguments)]

    @pytest.mark.parametrize(
        ("route", "operation"),
        [
            (Route.RESOLVE, "resolve"),
            (Route.GENERATE, "generate_hypothesis"),
            (Route.EXECUTE, "execute_run"),
        ],
    )
    def test_a_route_that_changes_something_hands_the_service_the_request_body(
        self, route, operation
    ):
        service = ServiceRecordingItsCalls()
        body = {"scene": "pouring", "query": "the(PouringAction)()"}
        with serving_with(service) as address:
            answered(Request(address + route, data=json.dumps(body).encode()))
        assert service.calls == [(operation, (body,))]

    def test_what_the_service_answers_is_what_the_route_answers(self):
        with serving_with(ServiceRecordingItsCalls()) as address:
            status, payload = answered(Request(address + Route.EXECUTE, data=b"{}"))
        assert (status, payload) == (200, {PayloadKey.OK: True})

    def test_a_failure_of_the_service_is_answered_as_an_error_the_panels_show(self):
        with serving_with(ServiceRefusingEveryRequest()) as address:
            status, payload = answered(Request(address + Route.DESIGNATORS))
        assert payload[PayloadKey.OK] is False
        assert payload[PayloadKey.ERROR].startswith(NoSuchScene.__name__)

    def test_a_request_from_another_origin_is_refused_before_the_service_runs(self):
        service = ServiceRecordingItsCalls()
        with serving_with(service) as address:
            foreign = Request(
                address + Route.EXECUTE,
                data=b"{}",
                headers={"Origin": "http://elsewhere.example"},
            )
            status, payload = answered(foreign)
        assert (status, service.calls) == (403, [])
        assert payload[PayloadKey.OK] is False
