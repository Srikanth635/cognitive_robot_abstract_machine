"""
The cramera viewer with the resolution add-on.
"""

from __future__ import annotations

import logging


def configure_console_logging(level: int = logging.INFO) -> None:
    """
    Make the add-on's own logging visible on the console.

    Importing this package installs no logging configuration of its own. The entry
    points call this explicitly, and it sets the level on the add-on's own logger as
    well, which reaches whatever handler a dependency has already installed.

    :param level: The level the add-on's own loggers should emit at.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger(__name__).setLevel(level)
