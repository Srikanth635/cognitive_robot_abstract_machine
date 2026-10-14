"""
Where events are reported to: nowhere, memory, or a log shared by every process that
works on a run.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from typing_extensions import List

from llm_reasoner.observation.events import Event

LINE_END = "\n"
"""
What ends each event in a log.
"""


class EventSink(ABC):
    """
    Takes the events a part of the system reports.
    """

    @abstractmethod
    def emit(self, event: Event) -> None:
        """
        :param event: What happened.
        """


class NullSink(EventSink):
    """
    Reports nowhere: the sink of anything that is not being watched.
    """

    def emit(self, event: Event) -> None:
        return None


@dataclass
class MemorySink(EventSink):
    """
    Keeps the events it takes, in order.
    """

    events: List[Event] = field(default_factory=list)
    """
    The events taken so far.
    """

    def emit(self, event: Event) -> None:
        self.events.append(event)


@dataclass
class JsonLinesSink(EventSink):
    """
    Appends each event to a log as one line of JSON, so any number of writers, in this
    process or others, add to the same log and a reader follows it as it grows.
    """

    path: Path
    """
    The log.
    """

    lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)
    """
    Keeps the writers of this process from interleaving.
    """

    def emit(self, event: Event) -> None:
        """
        Append the event as one line, written at once so lines of other processes do not
        mix into it; a value JSON cannot write is written as its text.
        """
        line = json.dumps(event.to_json(), ensure_ascii=False, default=str)
        with self.lock, self.path.open("ab", buffering=0) as log:
            log.write((line + LINE_END).encode())
