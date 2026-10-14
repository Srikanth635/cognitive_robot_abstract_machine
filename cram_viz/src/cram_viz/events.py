"""
The events a resolution run in the viewer reports besides those of llm_reasoner's
components: that the run started and how it ended, and each pour tried in the
simulation.

They are written to the run's log (see :mod:`llm_reasoner.observation`).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

from llm_reasoner.observation.events import Event
from llm_reasoner.observation.sinks import EventSink
from typing_extensions import Any, Iterator, Mapping

# %% a run


@dataclass
class RunStarted(Event):
    """
    A resolution run began.
    """

    run: str
    """
    The name of the run.
    """

    query: str
    """
    The designator's match query, as written.
    """

    scene: str
    """
    The scene it is resolved in.
    """


@dataclass
class RunFinished(Event):
    """
    A resolution run resolved its designator.
    """

    run: str
    """
    The name of the run.
    """

    seconds: float
    """
    How long the run took.
    """


@dataclass
class RunFailed(Event):
    """
    A resolution run ended by raising an error.
    """

    run: str
    """
    The name of the run.
    """

    error_type: str
    """
    The name of the error raised.
    """

    reason: str
    """
    What the error said.
    """

    seconds: float
    """
    How long the run ran before it failed.
    """


@contextmanager
def reporting_run(sink: EventSink, run: str, query: str, scene: str) -> Iterator[None]:
    """
    Report a run as started, and as finished or, if it raises, as failed, which then
    goes on.

    :param sink: Where the run is reported.
    :param run: The name of the run.
    :param query: The designator's match query, as written.
    :param scene: The scene it is resolved in.
    """
    sink.emit(RunStarted(run=run, query=query, scene=scene))
    started = perf_counter()
    try:
        yield
    except Exception as error:
        sink.emit(
            RunFailed(
                run=run,
                error_type=type(error).__name__,
                reason=str(error),
                seconds=perf_counter() - started,
            )
        )
        raise
    sink.emit(RunFinished(run=run, seconds=perf_counter() - started))


# %% a pour tried in the simulation


@dataclass
class TrialStarted(Event):
    """
    A resolved pouring was started in the simulation, to see where it ends.
    """

    trial: str
    """
    Identifies the trial, which what it brought about repeats.
    """

    values: Mapping[str, Any]
    """
    The resolved slots' values tried, as data, by field name.
    """


@dataclass
class TrialFinished(Event):
    """
    A pour tried in the simulation ended.
    """

    trial: str
    """
    The trial that ended.
    """

    outcome: Mapping[str, Any]
    """
    What the pour brought about, or why the simulation could not finish it.
    """

    seconds: float
    """
    How long the trial took.
    """
