"""
cram_viz — browser-based visualization for the CRAM architecture.
"""

from __future__ import annotations

import logging

__version__ = "1.0.0"


def configure_console_logging(level: int = logging.INFO) -> None:
    """
    Make cram_viz's own logging visible on the console.

    Importing this package installs no logging configuration of its own -- a library
    must not do that as a side effect of import. The console entry points (``cram-viz``,
    ``cram-viz-live``, ``cram-viz-onboard``) call this explicitly instead.

    :func:`logging.basicConfig` is a no-op once the root logger already has a handler,
    which silently drops every message cram_viz's entry points log (the startup banner,
    the scene-bundle path, per-request access logs) whenever a dependency configures
    logging first. Setting the level directly on this package's own logger sidesteps
    that: it reaches whatever handler(s) already exist, installing a console handler on
    the root logger only if none exists yet.

    :param level: The level cram_viz's own loggers should emit at.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger(__name__).setLevel(level)
