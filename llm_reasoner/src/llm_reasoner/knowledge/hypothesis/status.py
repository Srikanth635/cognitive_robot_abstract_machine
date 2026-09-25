"""The epistemic status of a hypothesis, and what can defeat it.

WHY THE FULL VOCABULARY EXISTS BEFORE ANYTHING WRITES IT
--------------------------------------------------------
Nothing writes ``REFUTED`` yet: recording resolver rejections is deferred work.
The value is declared anyway because adding a status value later is cheap while
retrofitting the *concept* is not -- code written against a two-state world
(worked / did not work) bakes that assumption into every query, and unpicking it
means revisiting all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional


class HypothesisStatus(StrEnum):
    """Where a hypothesis stands.

    Ordered by increasing settledness, but not a lattice: a hypothesis moves
    between these by events, not by accumulating evidence, and a refuted one
    never returns.
    """

    CANDIDATE = "candidate"
    """Generated but not yet checked against the acceptance constraints. Not
    admissible: a candidate is what the repair loop is still working on."""

    VALIDATED = "validated"
    """Passed the acceptance constraints of every module it was generated
    under. Admissible, and the only status the query surface serves by
    default."""

    SUPERSEDED = "superseded"
    """A newer hypothesis for the same action replaced it, typically because a
    module version changed. Retained rather than deleted, so a past resolution
    stays explainable against the knowledge as it stood at the time."""

    REFUTED = "refuted"
    """Contradicted in use: the resolver rejected it, or repair never produced
    an admissible version. Retained because a refuted hypothesis is evidence
    about the ontology as much as about the model."""


@dataclass(frozen=True)
class Defeater:
    """A specific reason a hypothesis stopped being believed.

    Kept as structured data rather than a free-text note so that recurring
    defeaters can be counted. A fragment defeated the same way across many
    episodes is the signal that the stereotype itself is wrong for a class of
    situation -- which is the point at which an exception rule earns its place,
    rather than a regenerated prompt.
    """

    reason: str
    """What went wrong, in the vocabulary of whoever noticed."""

    source: str
    """Who noticed: a validator, the resolver, a human reviewer."""

    fragment_id: Optional[str] = None
    """The ``local_id`` of the fragment at fault, when the defeat is localised
    rather than about the hypothesis as a whole."""

    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


__all__ = ["Defeater", "HypothesisStatus"]