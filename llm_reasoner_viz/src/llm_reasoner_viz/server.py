"""
The viewer's HTTP server: the front page, the runs and the events of a run as JSON, and
a stream that follows a run's log as it grows.

The requests it answers:

- ``GET /api/config``: where cram_viz is, for the page to link to.
- ``GET /api/runs``: the runs that have a log, newest first.
- ``GET /api/runs/<run>/events``: the events of a run in one batch; ``?after=<offset>``
  goes on from where a reader stopped, ``?limit=<n>`` takes at most that many.
- ``GET /api/runs/<run>/stream``: the events of a run as server-sent events, those written
  so far and then those written next; ``?after=<offset>`` or the ``Last-Event-ID`` header
  goes on from where a reader stopped.

Started with ``llm-reasoner-viz`` or ``python -m llm_reasoner_viz.server --port 8721``.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from typing_extensions import Any, Callable, Dict, List, Optional

from llm_reasoner_viz import configure_console_logging
from llm_reasoner_viz.paths import cram_viz_url, default_runs_directory
from llm_reasoner_viz.runs import RunLibrary, UnknownRun
from llm_reasoner_viz.stream import EventStream

logger = logging.getLogger(__spec__.name)
"""
This module's logger, named for the module even when ``python -m`` runs it as
``__main__``, so the level :func:`configure_console_logging` sets reaches it.
"""

DEFAULT_PORT = 8721
"""
The port the viewer listens on unless told otherwise.
"""

DEFAULT_HOST = "127.0.0.1"
"""
The address the viewer listens on unless told otherwise: this machine only, since the
events hold prompts in full.
"""

WEB_ROOT = Path(__file__).resolve().parent / "web"
"""
The packaged front end.
"""

FRONT_PAGE = "index.html"
"""
The page served for the root address.
"""

CONFIG_PATH = re.compile(r"/api/config")
RUNS_PATH = re.compile(r"/api/runs")
EVENTS_PATH = re.compile(r"/api/runs/(?P<run>[^/]+)/events")
STREAM_PATH = re.compile(r"/api/runs/(?P<run>[^/]+)/stream")

LAST_EVENT_ID = "Last-Event-ID"
"""
The header in which a reader that lost a stream says where the last event it saw ended.
"""

TEXT_TYPES = ("text/", "application/javascript")
"""
The content types that are sent with a character set.
"""

Query = Dict[str, List[str]]


class BadRequest(ValueError):
    """
    A request the server cannot answer as asked.
    """


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """
    Answers the viewer's requests from a library of runs and the packaged front end.
    """

    def __init__(
        self, *arguments: Any, library: RunLibrary, web_root: Path, **keywords
    ):
        self.library = library
        self.web_root = web_root
        super().__init__(*arguments, **keywords)

    def log_message(self, format: str, *arguments: Any) -> None:
        """
        Route the per-request access log through logging.
        """
        logger.info("  " + format, *arguments)

    # %% routing

    def do_GET(self) -> None:
        """
        Route the API and the front end.
        """
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        routes: List[tuple] = [
            (CONFIG_PATH, self.config),
            (RUNS_PATH, self.runs),
            (EVENTS_PATH, self.events),
            (STREAM_PATH, self.stream),
        ]
        for pattern, route in routes:
            found = pattern.fullmatch(parsed.path)
            if found is not None:
                return self.answering(route, found.groupdict(), query)
        return self.static(unquote(parsed.path))

    def answering(
        self, route: Callable[..., None], arguments: Dict[str, str], query: Query
    ) -> None:
        """
        Answer a request of the API, refusing one for a run that has no log or one the
        server cannot make sense of.
        """
        try:
            route(query=query, **arguments)
        except UnknownRun as error:
            self.send_json({"ok": False, "error": str(error)}, code=404)
        except BadRequest as error:
            self.send_json({"ok": False, "error": str(error)}, code=400)

    # %% the API

    def config(self, query: Query) -> None:
        """
        Send where cram_viz is, for the page to link to.
        """
        self.send_json({"ok": True, "cram_viz_url": cram_viz_url()})

    def runs(self, query: Query) -> None:
        """
        Send the runs that have a log.
        """
        self.send_json(
            {"ok": True, "runs": [run.to_json() for run in self.library.runs()]}
        )

    def events(self, run: str, query: Query) -> None:
        """
        Send the events of a run, from an offset, in one batch.
        """
        log = self.library.log_of(run)
        batch = log.read_from(
            self.number(query, "after", default=0),
            limit=self.number(query, "limit", default=None),
        )
        self.send_json(
            {
                "ok": True,
                "events": [entry.event for entry in batch.entries],
                "next": batch.next_offset,
                "size": batch.size,
            }
        )

    def stream(self, run: str, query: Query) -> None:
        """
        Follow the log of a run, sending each event as it is written, until the reader
        goes away.
        """
        log = self.library.log_of(run)
        offset = self.number(
            query, "after", default=int(self.headers.get(LAST_EVENT_ID) or 0)
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for frame in EventStream(log=log, offset=offset).frames(lambda: False):
                self.wfile.write(frame.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.info("  a reader of %s went away", run)

    @staticmethod
    def number(query: Query, name: str, default: Optional[int]) -> Optional[int]:
        """
        :return: The whole number the query gives for a parameter, the default when it
            gives none.
        :raises BadRequest: When the parameter is not a whole number of zero or more.
        """
        if name not in query:
            return default
        given = query[name][0]
        if not given.isdigit():
            raise BadRequest(f"{name} must be a whole number, not {given!r}")
        return int(given)

    # %% the front end

    def static(self, path: str) -> None:
        """
        Send a file of the front end, and nothing outside it.
        """
        root = self.web_root.resolve()
        found = (root / (path.lstrip("/") or FRONT_PAGE)).resolve()
        if not found.is_relative_to(root) or not found.is_file():
            return self.send_json({"ok": False, "error": "not found"}, code=404)
        content_type = mimetypes.guess_type(found.name)[0] or "application/octet-stream"
        if content_type.startswith(TEXT_TYPES):
            content_type += "; charset=utf-8"
        self.send_bytes(found.read_bytes(), content_type)

    # %% answers

    def send_json(self, payload: Any, code: int = 200) -> None:
        """
        Send a payload as JSON with the given status code.
        """
        self.send_bytes(json.dumps(payload).encode(), "application/json", code)

    def send_bytes(self, body: bytes, content_type: str, code: int = 200) -> None:
        """
        Send a body with its type and the given status code.
        """
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(
    library: RunLibrary,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    web_root: Path = WEB_ROOT,
) -> ThreadingHTTPServer:
    """
    :param library: The runs to show.
    :param host: The address to listen on.
    :param port: The port to listen on; zero for any that is free, as tests ask.
    :param web_root: The directory of the front end.
    :return: A server ready to serve, each request on a thread of its own so a stream does
        not keep the others waiting.
    """
    return ThreadingHTTPServer(
        (host, port), partial(ViewerRequestHandler, library=library, web_root=web_root)
    )


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    """
    :param argv: The command line after the program.
    :return: What the command line asks for: the address and port to listen on and the
        directory of runs to show.
    """
    parser = argparse.ArgumentParser(
        prog="llm-reasoner-viz",
        description="Follow llm_reasoner at work in a browser.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="the address to listen on")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="the port to listen on"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=default_runs_directory(),
        help="the directory of runs to show",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """
    ``llm-reasoner-viz`` — serve the viewer.
    """
    configure_console_logging()
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    with make_server(
        RunLibrary(directory=arguments.runs), arguments.host, arguments.port
    ) as server:
        logger.info(
            "llm_reasoner viewer running at http://%s:%d/",
            arguments.host,
            server.server_address[1],
        )
        logger.info(
            "runs: %s%s",
            arguments.runs,
            (
                ""
                if arguments.runs.is_dir()
                else "  (none yet: resolve a designator in cram-viz)"
            ),
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
