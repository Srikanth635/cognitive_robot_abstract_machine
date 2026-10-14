"""
Plain enumerations that classify ontology declarations.
"""

from __future__ import annotations

from enum import StrEnum

# %% concepts


class Rigidity(StrEnum):
    """
    Whether membership in a concept is essential to what has it.
    """

    RIGID = "rigid"
    """
    Membership is essential: nothing can stop being a member without ceasing to exist.

    Realised as an ordinary subclass.
    """

    ANTI_RIGID = "anti_rigid"
    """
    Membership is contextual: whatever has it survives losing it.

    Realised as a role held by a separate taker.
    """


class ConceptLevel(StrEnum):
    """
    Whether a concept classifies kinds of thing or actual things.
    """

    DESCRIPTION = "description"
    """
    The concept classifies kinds: generic knowledge that holds independently of any
    episode.
    """

    OCCURRENCE = "occurrence"
    """
    The concept classifies actual, situated things that exist or happen in an episode.
    """


class AlignmentRelation(StrEnum):
    """
    How a concept relates to a term of another ontology it is aligned to.
    """

    SUBCLASS_OF = "subclass_of"
    """
    Every instance of the concept is an instance of the foreign term.
    """

    EQUIVALENT_TO = "equivalent_to"
    """
    The concept and the foreign term have the same instances.
    """

    RELATED_TO = "related_to"
    """
    The concept resembles the foreign term without either containing the other.
    """


class EvidenceStatus(StrEnum):
    """
    How well a source supports a represented claim.
    """

    EMPIRICAL_FINDING = "empirical_finding"
    """
    The source directly reports the represented result.
    """

    SUPPORTED_INTERPRETATION = "supported_interpretation"
    """
    The source infers the claim from converging results without observing it directly.
    """

    AUTHOR_HYPOTHESIS = "author_hypothesis"
    """
    The source proposes the claim without treating it as established.
    """

    OPEN_QUESTION = "open_question"
    """
    The source leaves the relationship explicitly unresolved.
    """

    ENGINEERING_EXTENSION = "engineering_extension"
    """
    The claim was added for application reasoning, and the source does not make it.
    """


# %% properties


class PropertyKind(StrEnum):
    """
    What a property holds.
    """

    RELATION = "relation"
    """
    An instance of another declared concept.
    """

    ATTRIBUTE = "attribute"
    """
    A plain value: a number, a string, a boolean, or a member of an enumeration.
    """

    REFERENCE = "reference"
    """
    A term of another ontology, named by its IRI.
    """


# %% validation


class Severity(StrEnum):
    """
    How seriously a failed check counts.
    """

    ERROR = "error"
    """
    The declaration or hypothesis is inadmissible.
    """

    WARNING = "warning"
    """
    The declaration or hypothesis is legal but probably not what was meant.
    """
