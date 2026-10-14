"""
llm_reasoner_viz — a browser viewer for llm_reasoner at work.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"


def configure_console_logging(level: int = logging.INFO) -> None:
    """
    Make the viewer's own logging visible on the console.

    Importing this package installs no logging configuration of its own; the console
    entry point calls this explicitly. The level is set on this package's own logger, so
    it reaches whatever handler already exists, and a console handler is installed on
    the root logger only when there is none.

    :param level: The level the viewer's own loggers should emit at.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger(__name__).setLevel(level)
