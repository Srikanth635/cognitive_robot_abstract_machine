"""
Reading the logs of resolution runs: the events a log holds and where a reader stands in
it, what a run was and how it went, and which runs there are.

A run's log is the file ``events.jsonl`` in the run's directory, one JSON line per
event, which every process working on the run appends to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import Lock

from typing_extensions import Any, Dict, Iterable, List, Optional, Tuple

from llm_reasoner_viz.vocabulary import EventField, KnownEvent

LOG_FILE = "events.jsonl"
"""
The name of a run's log in its directory.
"""

LINE_END = b"\n"
"""
What ends each event in a log.
"""

# %% the events of a log


@dataclass(frozen=True)
class LoggedEvent:
    """
    An event read from a log, with where it ends.
    """

    event: Dict[str, Any]
    """
    The event as written.
    """

    end: int
    """
    The offset in the log, in bytes, just after the event's line.
    """


@dataclass(frozen=True)
class EventBatch:
    """
    What was read of a log from an offset.
    """

    entries: List[LoggedEvent]
    """
    The events read, in order.
    """

    next_offset: int
    """
    Where the next read starts: after the last event read, or where this one started
    when it read none.
    """

    size: int
    """
    How many bytes of the log have been seen, a line still being written included.
    """


@dataclass(frozen=True)
class EventLog:
    """
    The log of one run.
    """

    path: Path
    """
    The file.
    """

    def read_from(self, offset: int = 0, limit: Optional[int] = None) -> EventBatch:
        """
        Read the events written since an offset.

        A line still being written is left for the next read, and a line that is not
        JSON is read as an event saying so.

        :param offset: Where to start, in bytes: zero for the whole log, or the end of
            the last event read before.
        :param limit: The most events to read; all there are when ``None``.
        :return: The events, and where the next read starts.
        """
        with self.path.open("rb") as log:
            log.seek(offset)
            data = log.read()
        complete = data[: data.rfind(LINE_END) + 1]
        entries: List[LoggedEvent] = []
        position = offset
        for line in complete.splitlines(keepends=True):
            if limit is not None and len(entries) >= limit:
                break
            position += len(line)
            if line.strip():
                entries.append(LoggedEvent(event=self.event_of(line), end=position))
        return EventBatch(
            entries=entries,
            next_offset=entries[-1].end if entries else offset,
            size=offset + len(data),
        )

    @staticmethod
    def event_of(line: bytes) -> Dict[str, Any]:
        """
        :return: The event a line of the log holds, or an event saying the line is not
            one.
        """
        text = line.decode(errors="replace").rstrip("\r\n")
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return {EventField.EVENT: KnownEvent.UNREADABLE, EventField.LINE: text}
        if isinstance(event, dict) and EventField.EVENT in event:
            return event
        return {EventField.EVENT: KnownEvent.UNREADABLE, EventField.LINE: text}


# %% what a run was


class Stage(StrEnum):
    """
    How far a stage of a run, resolving or performing, has got.
    """

    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class RunSummary:
    """
    What a run was and how it went, read from its log.
    """

    run: str
    """
    The name of the run, which names its directory.
    """

    query: str = ""
    """
    The match query the run resolved.
    """

    scene: str = ""
    """
    The scene it resolved it in.
    """

    started_at: Optional[float] = None
    """
    When the run started, in seconds since the epoch.
    """

    last_event_at: float = 0.0
    """
    When the latest event happened, in seconds since the epoch.
    """

    resolution: Optional[Stage] = None
    """
    How resolving has gone; none before the run says it started.
    """

    execution: Optional[Stage] = None
    """
    How performing the resolved action in the simulation has gone; none while it has not
    been started.
    """

    events: int = 0
    """
    How many events the log holds.
    """

    model_calls: int = 0
    """
    How many completions the models answered.
    """

    cost: float = 0.0
    """
    What the answered completions cost, in dollars, where the cost was known.
    """

    input_tokens: int = 0
    """
    The input tokens the completions used, read without the cache.
    """

    output_tokens: int = 0
    """
    The tokens the models generated.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The summary as plain JSON.
        """
        return asdict(self)

    @classmethod
    def of(cls, run: str, events: Iterable[Dict[str, Any]]) -> RunSummary:
        """
        :param run: The name of the run.
        :param events: The events of its log, in order.
        :return: What the run was and how it went.
        """
        summary = cls(run=run)
        for event in events:
            summary.take(event)
        return summary

    def take(self, event: Dict[str, Any]) -> None:
        """
        Account for one more event of the log.
        """
        at = event.get(EventField.AT, 0.0)
        self.events += 1
        self.last_event_at = max(self.last_event_at, at)
        if self.started_at is None:
            self.started_at = at
        match event[EventField.EVENT]:
            case KnownEvent.RUN_STARTED:
                self.query = event.get(EventField.QUERY, "")
                self.scene = event.get(EventField.SCENE, "")
                self.started_at = at
                self.resolution = Stage.RUNNING
            case KnownEvent.RUN_FINISHED:
                self.resolution = Stage.FINISHED
            case KnownEvent.RUN_FAILED:
                self.resolution = Stage.FAILED
            case KnownEvent.EXECUTION_STARTED:
                self.execution = Stage.RUNNING
            case KnownEvent.EXECUTION_FINISHED:
                self.execution = Stage.FINISHED
            case KnownEvent.EXECUTION_FAILED:
                self.execution = Stage.FAILED
            case KnownEvent.MODEL_ANSWERED:
                self.model_calls += 1
                self.cost += event.get(EventField.COST) or 0.0
                usage = event.get(EventField.USAGE) or {}
                self.input_tokens += usage.get(EventField.INPUT_TOKENS, 0)
                self.output_tokens += usage.get(EventField.OUTPUT_TOKENS, 0)


# %% which runs there are


class UnknownRun(LookupError):
    """
    No run of the given name has a log here.
    """

    def __init__(self, name: str):
        super().__init__(f"no run named {name!r} has a log")
        self.name = name


@dataclass
class RunLibrary:
    """
    The runs kept in a directory, each in a directory of its own named for it.
    """

    directory: Path
    """
    The directory of runs.
    """

    summaries: Dict[Path, Tuple[Tuple[int, int], RunSummary]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The summary of each log as it was when it last changed, by the log's path, so a log
    that has not changed is not read again.
    """

    lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)
    """
    Keeps the readers of several requests from interleaving.
    """

    def runs(self) -> List[RunSummary]:
        """
        :return: A summary of every run that has a log, the newest first.
        """
        if not self.directory.is_dir():
            return []
        summaries = [
            self.summary_of(log) for log in sorted(self.directory.glob(f"*/{LOG_FILE}"))
        ]
        return sorted(
            summaries,
            key=lambda summary: summary.started_at or summary.last_event_at,
            reverse=True,
        )

    def summary_of(self, log: Path) -> RunSummary:
        """
        :return: A summary of the run whose log it is, read again only if the log
            changed since the last time.
        """
        status = log.stat()
        signature = (status.st_size, status.st_mtime_ns)
        with self.lock:
            cached = self.summaries.get(log)
            if cached is not None and cached[0] == signature:
                return cached[1]
        summary = RunSummary.of(
            log.parent.name,
            (entry.event for entry in EventLog(path=log).read_from(0).entries),
        )
        with self.lock:
            self.summaries[log] = (signature, summary)
        return summary

    def log_of(self, run: str) -> EventLog:
        """
        :param run: The name of a run.
        :return: The run's log.
        :raises UnknownRun: When no run of that name has a log here.
        """
        if run in ("", ".", "..") or Path(run).name != run:
            raise UnknownRun(run)
        log = self.directory / run / LOG_FILE
        if not log.is_file():
            raise UnknownRun(run)
        return EventLog(path=log)
