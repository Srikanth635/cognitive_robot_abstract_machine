"""
Tests of the plumbing of the resolution routes in cramera's server: that each route asks
the service what the request names, and what a failure and a foreign origin are
answered.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from cramera.server import ViewerServer
from typing_extensions import Any

from cramera.resolution.exceptions import NoSuchScene
from cramera.resolution.payload import PayloadKey, ViewName
from cramera.resolution.routes import ResolutionRoute as Route
from cramera.resolution.service import ResolutionService

# %% helpers


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

    def start_performance(self):
        self.calls.append(("start_performance", ()))
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
    :param service: What the resolution routes are to run.
    :return: The address of a server over the service.
    """
    server = ViewerServer(0)
    server.resolution = service
    with serving(server) as address:
        yield address


# %% the routes served


class TestServedResolution:
    def test_the_server_answers_its_resolution_route(self):
        with serving(ViewerServer(0)) as address:
            answer = json.loads(fetched(address, Route.PING))
        assert answer[PayloadKey.OK] is True


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

    def test_the_start_route_asks_the_service_to_start_what_waits(self):
        service = ServiceRecordingItsCalls()
        with serving_with(service) as address:
            answered(Request(address + Route.START, data=b"{}"))
        assert service.calls == [("start_performance", ())]

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
