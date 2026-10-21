"""
The cramera viewer with the resolution add-on: cramera's own handler and server, extended
by subclassing, so nothing of cramera is edited.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import webbrowser
from dataclasses import dataclass
from enum import StrEnum

from cramera import paths
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.server import (
    EQL_AVAILABLE,
    Handler,
    ViewerServer,
    parse_arguments,
)
from krrood.symbol_graph.symbol_graph import SymbolGraph
from typing_extensions import Any, Callable, Sequence

from cramera_resolution.exceptions import IndexPageHasNoBootScript
from cramera_resolution.paths import AddOnDirectory
from cramera_resolution.payload import PayloadKey
from cramera_resolution.service import ResolutionService
from cramera_resolution.views import ViewName

logger = logging.getLogger(__name__)

# %% what the add-on is called in the viewer's addresses


class Route(StrEnum):
    """
    The addresses the add-on answers itself, besides the files under its own prefix.
    """

    INDEX = "/"
    """
    The viewer's page.
    """

    INDEX_FILE = "/index.html"
    """
    The viewer's page, by its file name.
    """

    PING = "/api/resolution/ping"
    """
    Tells that the add-on's server is the one answering.
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
    One of the add-on's views: the hypothesis graph, the ontology graph or the designator.
    """


class AddOnUrl(StrEnum):
    """
    Where the add-on's own files are served from.
    """

    PREFIX = "/addon/"
    """
    The start of the address of every file of the add-on's panels.
    """


class AddOnScript(StrEnum):
    """
    The scripts inserted into the viewer's page, by their address relative to it, in the
    order they run.
    """

    HYPOTHESIS_GRAPH = "addon/hypothesis_graph.js"
    """
    Draws the hypothesis as a graph that opens branch by branch.
    """

    ONTOLOGY_GRAPH = "addon/ontology_graph.js"
    """
    Draws the classes behind the hypothesis.
    """

    DESIGNATOR_VIEW = "addon/designator_view.js"
    """
    Draws the designator as collapsible Lisp or JSON.
    """

    RESOLUTION_VIEW = "addon/resolution_view.js"
    """
    Writes a resolution and the runs kept as markup.
    """

    DESIGNATOR_COMPLETION = "addon/designator_completion.js"
    """
    Decides which action designators and parameters the designator box suggests.
    """

    DESIGNATOR_PANEL = "addon/designator_panel.js"
    """
    The panel that resolves designators, chooses what to consult and performs runs.
    """

    KNOWLEDGE_TABS = "addon/knowledge_tabs.js"
    """
    Adds the hypothesis graph, the ontology graph and the designator to the tabs of the
    graph panel.
    """


class AddOnStylesheet(StrEnum):
    """
    The stylesheets inserted into the viewer's page, by their address relative to it.
    """

    DESIGNATOR_PANEL = "addon/designator.css"
    """
    The look of the designator panel.
    """

    KNOWLEDGE_VIEWS = "addon/knowledge.css"
    """
    The look of the knowledge views in the graph panel.
    """


class QueryParameter(StrEnum):
    """
    The query parameters of the add-on's routes.
    """

    NAME = "name"
    """
    The name of a view or of a run.
    """


class AddOnMessage(StrEnum):
    """
    What the add-on says when it refuses a request.
    """

    NO_BOOT_SCRIPT = "the viewer's page has no script that boots its panels"
    """
    The page cannot take the add-on's panels.
    """

    UNKNOWN_FILE = "unknown add-on file"
    """
    A file of the add-on that is not held.
    """


# %% the viewer's page


class PageMarker(StrEnum):
    """
    What the viewer's page is recognised by.
    """

    BOOT_SCRIPT = "Panels.boot()"
    """
    What the script that mounts the configured panels calls.
    """


@dataclass(frozen=True)
class IndexPage:
    """
    The viewer's page, as text, that scripts can be inserted into.
    """

    html: str
    """
    The page as cramera ships it.
    """

    @property
    def boot_script(self) -> re.Match | None:
        """
        The script that mounts the configured panels, ``None`` when the page has none.
        """
        return re.search(
            r"<script>(?:(?!</script>).)*" + re.escape(PageMarker.BOOT_SCRIPT),
            self.html,
            re.S,
        )

    def with_assets(
        self, scripts: Sequence[str], stylesheets: Sequence[str] = ()
    ) -> str:
        """
        :param scripts: The scripts to insert, by their address relative to the page.
        :param stylesheets: The stylesheets to insert, by their address relative to the
            page.
        :return: The page with a tag for each, just before the script that mounts the
            panels, so the scripts define their panels and change the layout in time.
        :raises IndexPageHasNoBootScript: When the page has no such script.
        """
        boot_script = self.boot_script
        if boot_script is None:
            raise IndexPageHasNoBootScript(marker=PageMarker.BOOT_SCRIPT)
        tags = "".join(
            f'<link rel="stylesheet" href="{sheet}">\n' for sheet in stylesheets
        ) + "".join(f'<script src="{source}"></script>\n' for source in scripts)
        return (
            self.html[: boot_script.start()] + tags + self.html[boot_script.start() :]
        )


# %% handler and server


class ResolutionHandler(Handler):
    """
    Cramera's handler, also answering the add-on's routes and serving its panels; every
    other request is cramera's own.
    """

    def do_GET(self) -> None:
        """
        Answer the add-on's routes, and leave every other request to cramera.
        """
        route = self.path.split("?")[0]
        if route in (Route.INDEX, Route.INDEX_FILE):
            return self._serve_index_with_addon()
        if route.startswith(AddOnUrl.PREFIX):
            return self._serve_addon_file(route)
        if route == Route.PING:
            return self._answer(
                lambda: {PayloadKey.OK: True, PayloadKey.ADDON: __package__}
            )
        if route == Route.RUNS:
            return self._answer(self.service.kept_runs)
        if route == Route.DESIGNATORS:
            return self._answer(
                lambda: self.service.offered_designators(self._requested_scene())
            )
        if route == Route.RUN:
            return self._answer(
                lambda: self.service.open_run(self._named(""), self._requested_scene())
            )
        if route == Route.VIEW:
            return self._answer(
                lambda: self.service.view(
                    self._named(ViewName.HYPOTHESIS), self._requested_scene()
                )
            )
        return super().do_GET()

    def do_POST(self) -> None:
        """
        Answer the add-on's routes that change something, and leave every other request
        to cramera.
        """
        route = self.path.split("?")[0]
        if route == Route.RESOLVE:
            return self._answer(lambda: self.service.resolve(self._read_request()))
        if route == Route.GENERATE:
            return self._answer(
                lambda: self.service.generate_hypothesis(self._read_request())
            )
        if route == Route.EXECUTE:
            return self._answer(lambda: self.service.execute_run(self._read_request()))
        return super().do_POST()

    @property
    def service(self) -> ResolutionService:
        """
        The operations behind the routes, owned by the server.
        """
        return self.server.service

    def _named(self, default: str) -> str:
        """
        :param default: What to answer when the request names nothing.
        :return: The name the request carries as its ``name`` parameter.
        """
        return (self._query_parameters().get(QueryParameter.NAME) or [default])[0]

    def _read_request(self) -> dict[str, Any]:
        """
        :return: The JSON object the request carries as its body; empty when it has none.
        """
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _answer(self, operation: Callable[[], dict[str, Any]]) -> None:
        """
        Run an operation of the service and send what it answers, or the exception it
        raised, as cramera's panels read an error.

        The operation runs holding the server's query lock, as krrood's symbol graph is not
        safe to use from two requests at once, and cramera's own queries use it too.

        :param operation: What to run.
        """
        if not self._accept_origin():
            return
        try:
            with self.server.query_lock:
                payload = operation()
        except Exception as error:
            logger.exception("%s failed", self.path)
            return self._send_exception(error)
        return self._send_json(payload)

    def _serve_index_with_addon(self) -> None:
        """
        Serve cramera's page with the add-on's scripts inserted.
        """
        if not self._accept_origin():
            return
        page = IndexPage(
            html=(paths.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        )
        if page.boot_script is None:
            return self._send_error(AddOnMessage.NO_BOOT_SCRIPT, 500)
        body = page.with_assets(list(AddOnScript), list(AddOnStylesheet)).encode(
            "utf-8"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_addon_file(self, url_path: str) -> None:
        """
        Serve one file of the add-on's panels, never one outside them.

        :param url_path: The request path, starting with the add-on's prefix.
        """
        if not self._accept_origin():
            return
        base = (AddOnDirectory.WEB / AddOnUrl.PREFIX.strip("/")).resolve()
        target = (base / url_path[len(AddOnUrl.PREFIX) :]).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return self._send_error(AddOnMessage.UNKNOWN_FILE, 404)
        self._send_file(target)


class ResolutionServer(ViewerServer):
    """
    Cramera's viewer server, answering with the add-on's handler.
    """

    service: ResolutionService
    """
    The operations the add-on's routes run, and what they share.
    """

    def __init__(self, port: int) -> None:
        """
        :param port: The port to listen on, or zero to choose a free one.
        """
        super().__init__(port)
        self.RequestHandlerClass = ResolutionHandler
        self.service = ResolutionService()


# %% starting it


def main(arguments: Sequence[str] | None = None) -> None:
    """
    ``cramera-resolution``: serve the viewer with the add-on, as ``cramera`` serves it
    alone.

    :param arguments: The command-line arguments, or ``None`` to read them as given.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    options = parse_arguments(list(sys.argv[1:] if arguments is None else arguments))
    # Everything is imported by now, so the graph knows every class the knowledge
    # vocabulary declares; the world of a scene is read from it.
    SymbolGraph.clear()
    SymbolGraph()
    if EQL_AVAILABLE:
        EpisodeKnowledgeBase.of_active_scene()
    with ResolutionServer(options.port) as server:
        logger.info(
            "cramera with the resolution add-on at http://localhost:%d/", options.port
        )
        logger.info("scene bundles: %s", paths.scenes_directory())
        if options.open_browser:
            webbrowser.open("http://localhost:%d/" % options.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
