"""
The cramera viewer with the resolution add-on: cramera's own handler and server, extended
by subclassing, so nothing of cramera is edited.
"""

from __future__ import annotations

import logging
import re
import sys
import webbrowser
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from cramera import paths
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.server import (
    EQL_AVAILABLE,
    Handler,
    ViewerServer,
    parse_arguments,
)
from typing_extensions import ClassVar, Optional, Sequence

from cramera_resolution.exceptions import IndexPageHasNoBootScript

logger = logging.getLogger(__name__)

ADDON_WEB_ROOT = Path(__file__).parent / "web"
"""
Where the add-on's panels are kept.
"""

ADDON_URL_PREFIX = "/addon/"
"""
The start of the address of every file of the add-on's panels.
"""

ADDON_SCRIPTS = ("addon/hello_panel.js",)
"""
The scripts inserted into the viewer's page, by their address relative to it.
"""

# %% routes


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


# %% the viewer's page


@dataclass(frozen=True)
class IndexPage:
    """
    The viewer's page, as text, that scripts can be inserted into.
    """

    html: str
    """
    The page as cramera ships it.
    """

    BOOT_MARKER: ClassVar[str] = "Panels.boot()"
    """
    What the script that mounts the configured panels is recognised by.
    """

    def with_scripts(self, sources: Sequence[str]) -> str:
        """
        :param sources: The scripts to insert, by their address relative to the page.
        :return: The page with a script tag for each, just before the script that mounts the
            panels, so they define their panels and change the layout in time.
        :raises IndexPageHasNoBootScript: When the page has no such script.
        """
        boot_script = re.search(
            r"<script>(?:(?!</script>).)*" + re.escape(self.BOOT_MARKER),
            self.html,
            re.S,
        )
        if boot_script is None:
            raise IndexPageHasNoBootScript(marker=self.BOOT_MARKER)
        tags = "".join(f'<script src="{source}"></script>\n' for source in sources)
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
        if route.startswith(ADDON_URL_PREFIX):
            return self._serve_addon_file(route)
        if route == Route.PING:
            return self._answer_ping()
        return super().do_GET()

    def _serve_index_with_addon(self) -> None:
        """
        Serve cramera's page with the add-on's scripts inserted.
        """
        if not self._accept_origin():
            return
        page = IndexPage(
            html=(paths.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        )
        try:
            body = page.with_scripts(ADDON_SCRIPTS).encode("utf-8")
        except IndexPageHasNoBootScript as error:
            return self._send_error(str(error), 500)
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
        base = (ADDON_WEB_ROOT / ADDON_URL_PREFIX.strip("/")).resolve()
        target = (base / url_path[len(ADDON_URL_PREFIX) :]).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return self._send_error("unknown add-on file", 404)
        self._send_file(target)

    def _answer_ping(self) -> None:
        """
        Tell that the add-on's server is answering.
        """
        if not self._accept_origin():
            return
        self._send_json({"ok": True, "addon": "cramera_resolution"})


class ResolutionServer(ViewerServer):
    """
    Cramera's viewer server, answering with the add-on's handler.
    """

    def __init__(self, port: int) -> None:
        """
        :param port: The port to listen on, or zero to choose a free one.
        """
        super().__init__(port)
        self.RequestHandlerClass = ResolutionHandler


# %% starting it


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """
    ``cramera-resolution``: serve the viewer with the add-on, as ``cramera`` serves it
    alone.

    :param arguments: The command-line arguments, or ``None`` to read them as given.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    options = parse_arguments(list(sys.argv[1:] if arguments is None else arguments))
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
