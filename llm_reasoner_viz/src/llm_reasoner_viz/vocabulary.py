"""
The names the viewer reads in the events of a run's log. The processes writing the log
spell them as the classes of their events are named, so the viewer needs no code of
theirs.
"""

from __future__ import annotations

from enum import StrEnum


class EventField(StrEnum):
    """
    The fields of an event the viewer reads.
    """

    EVENT = "event"
    """
    The kind of the event.
    """

    AT = "at"
    """
    When it happened, in seconds since the epoch.
    """

    QUERY = "query"
    """
    The match query a run resolves.
    """

    SCENE = "scene"
    """
    The scene a run resolves it in.
    """

    COST = "cost"
    """
    What a completion cost, in dollars.
    """

    USAGE = "usage"
    """
    The tokens a completion used.
    """

    INPUT_TOKENS = "input_tokens"
    """
    The input tokens of a completion, read without the cache.
    """

    OUTPUT_TOKENS = "output_tokens"
    """
    The tokens a completion generated.
    """

    CACHE_CREATION_INPUT_TOKENS = "cache_creation_input_tokens"
    """
    The input tokens of a completion that were written to the cache.
    """

    CACHE_READ_INPUT_TOKENS = "cache_read_input_tokens"
    """
    The input tokens of a completion that were read from the cache.
    """

    LINE = "line"
    """
    The text of a line of the log that is not an event.
    """

    KIND = "kind"
    """
    What a NEEM is the episode of.
    """

    EPISODE = "episode"
    """
    The trial or the run a NEEM is of.
    """

    DATABASE = "database"
    """
    The database a NEEM is kept in.
    """

    DATABASE_ID = "database_id"
    """
    The identifier a NEEM was given in its database.
    """

    TRIAL = "trial"
    """
    The trial a pour tried in the simulation is.
    """

    OUTCOME = "outcome"
    """
    What a pour tried in the simulation brought about.
    """

    RUN = "run"
    """
    The run an event is of.
    """


class KnownEvent(StrEnum):
    """
    The kinds of event the viewer sums a run up by.
    """

    RUN_STARTED = "RunStarted"
    RUN_FINISHED = "RunFinished"
    RUN_FAILED = "RunFailed"
    EXECUTION_STARTED = "ExecutionStarted"
    EXECUTION_FINISHED = "ExecutionFinished"
    EXECUTION_FAILED = "ExecutionFailed"
    TRIAL_FINISHED = "TrialFinished"
    MODEL_ANSWERED = "ModelAnswered"
    NEEM_RECORDED = "NeemRecorded"
    NEEM_NOT_RECORDED = "NeemNotRecorded"

    UNREADABLE = "Unreadable"
    """
    Not written by any process: how the viewer shows a line of a log that is not an
    event.
    """


class NeemKind(StrEnum):
    """
    What a NEEM, the plan the simulation performed, is the episode of.
    """

    TRIAL = "trial"
    """
    A pour tried in the simulation before it is accepted.
    """

    EXECUTION = "execution"
    """
    A run's resolved action, performed in the simulation.
    """
