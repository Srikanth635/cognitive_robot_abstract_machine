"""Identity and rigidity vocabulary.

WHY THIS MODULE EXISTS
----------------------
This model was designed in OWL, where a handful of distinctions were enforced by
the formalism itself: whether a class is rigid, whether siblings are disjoint,
whether a term lives at the type level or the token level. Python enforces none
of them. Ordinary inheritance will happily let an anti-rigid role be declared as
a subclass, and nothing will complain.

The metadata declared here is therefore not documentation. It is the load-bearing
part of the port: ``Rigidity.ANTI_RIGID`` is what says a class must be realised as
a ``Role[T]`` rather than a subclass, and ``disjoint_with`` is what a validator
checks. Getting these wrong keeps the shape of the ontology while losing its
content.
"""

from __future__ import annotations

from enum import StrEnum


class Rigidity(StrEnum):
    """Whether class membership is essential to the thing that has it.

    The OntoClean distinction, and the single most consequential piece of
    metadata in the system: it decides subclass versus role.
    """

    RIGID = "rigid"
    """Membership is essential. A thing cannot stop being one without ceasing to
    exist: an ``ActionPhase`` is necessarily an action phase. Realised as an
    ordinary subclass."""

    ANTI_RIGID = "anti_rigid"
    """Membership is contextual and the thing survives losing it. A body that
    stops being the initial support is still that body. Realised as a
    ``Role[T]`` -- pure composition, distinct identity, no inheritance from the
    role taker."""

    SEMI_RIGID = "semi_rigid"
    """Essential to some members and contingent for others. Present for
    completeness; if a class in this model needs it, that is a signal the class
    is conflating two things and should be split."""


class ConceptLevel(StrEnum):
    """Whether a class describes types or occurrences.

    The distinction that keeps stereotypical hypothesis knowledge from
    contaminating the episodic world model. It is recorded here so that the
    scope separation in Phase 5 can be checked mechanically rather than
    maintained by convention.
    """

    DESCRIPTION = "description"
    """A concept: a type that classifies occurrences. Non-indexical, generic,
    defeasible. Everything generated as hypothesis content lives here."""

    OCCURRENCE = "occurrence"
    """A token: something that actually happened or exists, indexical and
    situated. Nothing in the microworlds is at this level; the semantic world
    model owns it."""

    META = "meta"
    """A statement about the knowledge itself rather than about the domain --
    module catalogues, capability profiles, provenance. Neither a domain type
    nor a domain occurrence."""


__all__ = ["Rigidity", "ConceptLevel"]