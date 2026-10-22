"""
The routes of the resolution feature: resolving underspecified action designators, the
runs kept, the knowledge views and performing a run.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from typing_extensions import Any, Callable

from cramera.resolution.payload import PayloadKey, ViewName

logger = logging.getLogger(__name__)

# %% the addresses


class ResolutionRoute(StrEnum):
    """
    The addresses the resolution feature answers.
    """

    PING = "/api/resolution/ping"
    """
    Tells that the resolution feature is served.
    """

    RUNS = "/api/resolution/runs"
    """
    The runs kept, newest first.
    """

    DESIGNATORS = "/api/resolution/designators"
    """
    The predefined designators a scene offers.
    """

    RUN = "/api/resolution/run"
    """
    A resolution run kept earlier, shown as it was made.
    """

    RESOLVE = "/api/resolution/resolve"
    """
    Resolves a designator query in a scene.
    """

    GENERATE = "/api/resolution/hypothesis"
    """
    Generates hypothesis knowledge for an instruction.
    """

    EXECUTE = "/api/resolution/execute"
    """
    Performs a resolution run in the simulation.
    """

    VIEW = "/api/resolution/view"
    """
    One of the views of the knowledge: the hypothesis graph, the ontology graph or the
    designator.
    """


class ResolutionMessage(StrEnum):
    """
    What the resolution routes say when they cannot answer.
    """

    UNAVAILABLE = (
        "resolution is not available in this environment: it needs coraplex's ROS "
        "dependency, rclpy"
    )
    """
    The viewer was started where the resolution feature cannot be imported.
    """


class QueryParameter(StrEnum):
    """
    The query parameters of the resolution routes.
    """

    NAME = "name"
    """
    The name of a view or of a run.
    """


# %% answering them


class ResolutionRoutes:
    """
    The resolution routes, for the request handler of the viewer to answer.

    A mixin of the handler, which provides ``_query_parameters``, ``_requested_scene``,
    ``_request_body``, ``_send_json`` and ``_send_exception``, and whose server provides
    ``query_lock`` and the :class:`~cramera.resolution.service.ResolutionService` as
    ``resolution``, which is ``None`` where the service cannot be imported.
    """

    def answer_resolution_get(self, route: str) -> bool:
        """
        Answer a read request, if the route is one of the resolution feature's.

        :param route: The request path, without its query.
        :return: Whether the route is the resolution feature's and has been answered.
        """
        if route not in ResolutionRoute:
            return False
        if self.server.resolution is None:
            self._send_error(ResolutionMessage.UNAVAILABLE)
        elif route == ResolutionRoute.PING:
            self._answer(lambda: {PayloadKey.OK: True, PayloadKey.ADDON: __package__})
        elif route == ResolutionRoute.RUNS:
            self._answer(self.server.resolution.kept_runs)
        elif route == ResolutionRoute.DESIGNATORS:
            self._answer(
                lambda: self.server.resolution.offered_designators(
                    self._requested_scene()
                )
            )
        elif route == ResolutionRoute.RUN:
            self._answer(
                lambda: self.server.resolution.open_run(
                    self._named(""), self._requested_scene()
                )
            )
        elif route == ResolutionRoute.VIEW:
            self._answer(
                lambda: self.server.resolution.view(
                    self._named(ViewName.HYPOTHESIS), self._requested_scene()
                )
            )
        else:
            return False
        return True

    def answer_resolution_post(self, route: str) -> bool:
        """
        Answer a request that changes something, if the route is one of the resolution
        feature's.

        :param route: The request path, without its query.
        :return: Whether the route is the resolution feature's and has been answered.
        """
        if route not in ResolutionRoute:
            return False
        if self.server.resolution is None:
            self._send_error(ResolutionMessage.UNAVAILABLE)
        elif route == ResolutionRoute.RESOLVE:
            self._answer(lambda: self.server.resolution.resolve(self._request_body()))
        elif route == ResolutionRoute.GENERATE:
            self._answer(
                lambda: self.server.resolution.generate_hypothesis(self._request_body())
            )
        elif route == ResolutionRoute.EXECUTE:
            self._answer(
                lambda: self.server.resolution.execute_run(self._request_body())
            )
        else:
            return False
        return True

    def _named(self, default: str) -> str:
        """
        :param default: What to answer when the request names nothing.
        :return: The name the request carries as its ``name`` parameter.
        """
        return (self._query_parameters().get(QueryParameter.NAME) or [default])[0]

    def _answer(self, operation: Callable[[], dict[str, Any]]) -> None:
        """
        Run an operation of the service and send what it answers, or the exception it
        raised, as the viewer's panels read an error.

        The operation runs holding the server's query lock, as krrood's symbol graph is not
        safe to use from two requests at once, and the viewer's own queries use it too.

        :param operation: What to run.
        """
        try:
            with self.server.query_lock:
                payload = operation()
        except Exception as error:
            logger.exception("%s failed", self.path)
            return self._send_exception(error)
        return self._send_json(payload)
