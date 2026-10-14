"""
The events a resolution run in the viewer reports besides those of llm_reasoner's
components: that the run started and how it ended, and each pour tried in the
simulation.

They are written to the run's log (see :mod:`llm_reasoner.observation`).
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from llm_reasoner.observation.events import Event
from llm_reasoner.observation.sinks import EventSink
from typing_extensions import Any, Callable, Iterator, Mapping

# %% reporting what begins and ends


@contextmanager
def reported(
    sink: EventSink,
    started: Event,
    finished: Callable[[float], Event],
    failed: Callable[[Exception, float], Event],
) -> Iterator[None]:
    """
    Report something as started, and as finished or, if it raises, as failed, which then
    goes on.

    :param sink: Where it is reported.
    :param started: The event saying it began.
    :param finished: The event saying it ended, given how long it took.
    :param failed: The event saying it failed, given the error and how long it ran.
    """
    sink.emit(started)
    began = perf_counter()
    try:
        yield
    except Exception as error:
        sink.emit(failed(error, perf_counter() - began))
        raise
    sink.emit(finished(perf_counter() - began))


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


def reporting_run(
    sink: EventSink, run: str, query: str, scene: str
) -> AbstractContextManager[None]:
    """
    :param sink: Where the run is reported.
    :param run: The name of the run.
    :param query: The designator's match query, as written.
    :param scene: The scene it is resolved in.
    :return: What reports a run as started, and as finished or failed.
    """
    return reported(
        sink,
        RunStarted(run=run, query=query, scene=scene),
        lambda seconds: RunFinished(run=run, seconds=seconds),
        lambda error, seconds: RunFailed(
            run=run,
            error_type=type(error).__name__,
            reason=str(error),
            seconds=seconds,
        ),
    )


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


# %% performing a run in the simulation


@dataclass
class ExecutionStarted(Event):
    """
    A run's resolved action began to be performed in the simulation.
    """

    run: str
    """
    The name of the run.
    """


@dataclass
class ExecutionFinished(Event):
    """
    A run's resolved action was performed.
    """

    run: str
    """
    The name of the run.
    """

    seconds: float
    """
    How long it took.
    """


@dataclass
class ExecutionFailed(Event):
    """
    Performing a run's resolved action ended by raising an error.
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
    How long it ran before it failed.
    """


@dataclass
class StepChanged(Event):
    """
    A step of the plan performing a run's action changed status.
    """

    run: str
    """
    The name of the run.
    """

    step: str
    """
    The action the step performs.
    """

    status: str
    """
    The status it changed to, as the plan names it.
    """


# %% what the simulation performed, kept as a NEEM


class NeemKind(StrEnum):
    """
    What a NEEM is the episode of.
    """

    TRIAL = "trial"
    """
    A pour tried in the simulation before it is accepted.
    """

    EXECUTION = "execution"
    """
    A run's resolved action, performed in the simulation.
    """


@dataclass
class NeemRecorded(Event):
    """
    The plan a simulation performed was kept as a NEEM.
    """

    kind: NeemKind
    """
    What the NEEM is the episode of.
    """

    episode: str
    """
    The trial or the run the NEEM is of: the identifier of the trial, or the name of the run.
    """

    database: str
    """
    The database it is kept in.
    """

    database_id: int
    """
    The identifier the database gave it.
    """

    seconds: float
    """
    How long it took to keep.
    """


@dataclass
class NeemNotRecorded(Event):
    """
    The plan a simulation performed could not be kept as a NEEM.
    """

    kind: NeemKind
    """
    What the NEEM would have been the episode of.
    """

    episode: str
    """
    The trial or the run the NEEM would have been of.
    """

    error_type: str
    """
    The name of the error raised.
    """

    reason: str
    """
    What the error said.
    """


def reporting_execution(sink: EventSink, run: str) -> AbstractContextManager[None]:
    """
    :param sink: Where the execution is reported.
    :param run: The name of the run being performed.
    :return: What reports a run as started to be performed, and as finished or failed.
    """
    return reported(
        sink,
        ExecutionStarted(run=run),
        lambda seconds: ExecutionFinished(run=run, seconds=seconds),
        lambda error, seconds: ExecutionFailed(
            run=run,
            error_type=type(error).__name__,
            reason=str(error),
            seconds=seconds,
        ),
    )
