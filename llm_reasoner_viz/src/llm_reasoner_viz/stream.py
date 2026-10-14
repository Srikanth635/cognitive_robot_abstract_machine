"""
Following a log as it grows: each event sent once, in order, from wherever a reader asks
to start, with a sign of life while nothing happens.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from typing_extensions import Callable, Iterator

from llm_reasoner_viz.runs import EventLog, LoggedEvent

KEEPALIVE = ": keepalive\n\n"
"""
What is sent while the log is quiet: a comment, which a reader ignores, so a connection
nothing has used for a while is not taken for dead.
"""


def frame_of(entry: LoggedEvent) -> str:
    """
    :return: The event as one frame of a server-sent event stream, naming where it ends
        in the log, so a reader that lost the connection can go on from there.
    """
    return f"id: {entry.end}\ndata: {json.dumps(entry.event)}\n\n"


@dataclass
class EventStream:
    """
    Follows one run's log from an offset.
    """

    log: EventLog
    """
    The log followed.
    """

    offset: int = 0
    """
    Where the next look starts, in bytes: after the last event sent.
    """

    poll_seconds: float = 0.25
    """
    How long to wait between looks at a log that has nothing new.
    """

    keepalive_seconds: float = 15.0
    """
    How long a log may be quiet before a sign of life is sent.
    """

    pause: Callable[[float], None] = time.sleep
    """
    Waits between two looks.
    """

    def frames(self, should_stop: Callable[[], bool]) -> Iterator[str]:
        """
        Send each event of the log not sent yet, then wait for more.

        :param should_stop: Asked before each look; the stream ends when it says so.
        :return: The frames to send, in order.
        """
        last_sent = time.monotonic()
        while not should_stop():
            batch = self.log.read_from(self.offset)
            for entry in batch.entries:
                self.offset = entry.end
                last_sent = time.monotonic()
                yield frame_of(entry)
            if batch.entries:
                continue
            if time.monotonic() - last_sent >= self.keepalive_seconds:
                last_sent = time.monotonic()
                yield KEEPALIVE
            self.pause(self.poll_seconds)
