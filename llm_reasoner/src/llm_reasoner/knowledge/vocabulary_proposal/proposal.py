"""
A proposed vocabulary, as the model answers it: classes with the classes they specialise
and the restrictions they state, object properties with their domain and range, and value
partitions with their members. A term of another ontology is named with its prefix, such
as ``core:SourceSlot``; a term the proposal declares, by its name alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any, Dict, Optional, Tuple

from llm_reasoner.knowledge.vocabulary_proposal.catalogue import TermKind


class AnswerKey(StrEnum):
    """
    The keys of an answer.
    """

    SUMMARY = "summary"
    CLASSES = "classes"
    PROPERTIES = "properties"
    VALUE_PARTITIONS = "value_partitions"
    NAME = "name"
    MEANING = "meaning"
    PARENTS = "parents"
    RESTRICTIONS = "restrictions"
    SEE_ALSO = "see_also"
    PROPERTY = "property"
    QUANTIFIER = "quantifier"
    FILLER = "filler"
    DOMAIN = "domain"
    RANGE = "range"
    FUNCTIONAL = "functional"
    MEMBERS = "members"


class RestrictionKind(StrEnum):
    """
    What a restriction requires of the values of a property.
    """

    SOME = "some"
    """
    At least one value belongs to the filler class.
    """

    ONLY = "only"
    """
    Every value belongs to the filler class.
    """

    VALUE = "value"
    """
    The filler individual is one of the values.
    """


@dataclass(frozen=True)
class ProposedRestriction:
    """
    What a class requires of the values of a property.
    """

    on_property: str
    """
    The property.
    """

    kind: RestrictionKind
    """
    What it requires.
    """

    filler: str
    """
    The class the values belong to, or the individual that is one of them.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ProposedRestriction:
        """
        :return: The restriction an answer states.
        """
        return cls(
            on_property=data[AnswerKey.PROPERTY],
            kind=RestrictionKind(data[AnswerKey.QUANTIFIER]),
            filler=data[AnswerKey.FILLER],
        )


@dataclass(frozen=True)
class ProposedClass:
    """
    A class the proposal declares.
    """

    name: str
    """
    Its name, in capitals, such as ``FillerExpectation``.
    """

    meaning: str
    """
    Its definition.
    """

    parents: Tuple[str, ...]
    """
    The classes it specialises.
    """

    restrictions: Tuple[ProposedRestriction, ...]
    """
    What it requires of the values of properties.
    """

    see_also: Tuple[str, ...]
    """
    The IRIs of the sources it is taken from, such as a FrameNet frame.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ProposedClass:
        """
        :return: The class an answer declares.
        """
        return cls(
            name=data[AnswerKey.NAME],
            meaning=data[AnswerKey.MEANING],
            parents=tuple(data[AnswerKey.PARENTS]),
            restrictions=tuple(
                map(ProposedRestriction.from_json, data[AnswerKey.RESTRICTIONS])
            ),
            see_also=tuple(data[AnswerKey.SEE_ALSO]),
        )


@dataclass(frozen=True)
class ProposedProperty:
    """
    An object property the proposal declares.
    """

    name: str
    """
    Its name, starting in lower case, such as ``expectsOfSlot``.
    """

    meaning: str
    """
    Its definition.
    """

    domain: str
    """
    The class its subjects belong to.
    """

    value_class: Optional[str]
    """
    The class its values belong to, its range; ``None`` when its values name a CRAM-AICOR
    concept, such as an action category or a state type, by its IRI.
    """

    parents: Tuple[str, ...]
    """
    The properties it specialises.
    """

    functional: bool
    """
    Whether a subject has at most one value of it.
    """

    see_also: Tuple[str, ...]
    """
    The IRIs of the sources it is taken from.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ProposedProperty:
        """
        :return: The property an answer declares.
        """
        return cls(
            name=data[AnswerKey.NAME],
            meaning=data[AnswerKey.MEANING],
            domain=data[AnswerKey.DOMAIN],
            value_class=data[AnswerKey.RANGE],
            parents=tuple(data[AnswerKey.PARENTS]),
            functional=data[AnswerKey.FUNCTIONAL],
            see_also=tuple(data[AnswerKey.SEE_ALSO]),
        )


@dataclass(frozen=True)
class ProposedMember:
    """
    A member of a value partition.
    """

    name: str
    """
    Its name, in capitals, such as ``HasOpening``.
    """

    meaning: str
    """
    Its definition.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ProposedMember:
        """
        :return: The member an answer declares.
        """
        return cls(name=data[AnswerKey.NAME], meaning=data[AnswerKey.MEANING])


@dataclass(frozen=True)
class ProposedValuePartition:
    """
    A class whose individuals are exactly the members listed, such as the features a
    filler can have.
    """

    name: str
    """
    Its name, in capitals.
    """

    meaning: str
    """
    Its definition.
    """

    parents: Tuple[str, ...]
    """
    The classes it specialises.
    """

    members: Tuple[ProposedMember, ...]
    """
    Its members, in order.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ProposedValuePartition:
        """
        :return: The value partition an answer declares.
        """
        return cls(
            name=data[AnswerKey.NAME],
            meaning=data[AnswerKey.MEANING],
            parents=tuple(data[AnswerKey.PARENTS]),
            members=tuple(map(ProposedMember.from_json, data[AnswerKey.MEMBERS])),
        )


@dataclass(frozen=True)
class VocabularyProposal:
    """
    The vocabulary the model proposes for a perspective.
    """

    summary: str
    """
    What the vocabulary lets a knowledge base say, which becomes the module's comment.
    """

    classes: Tuple[ProposedClass, ...]
    """
    The classes it declares.
    """

    properties: Tuple[ProposedProperty, ...]
    """
    The object properties it declares.
    """

    value_partitions: Tuple[ProposedValuePartition, ...]
    """
    The value partitions it declares.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> VocabularyProposal:
        """
        :return: The proposal an answer holds.
        """
        return cls(
            summary=data[AnswerKey.SUMMARY],
            classes=tuple(map(ProposedClass.from_json, data[AnswerKey.CLASSES])),
            properties=tuple(
                map(ProposedProperty.from_json, data[AnswerKey.PROPERTIES])
            ),
            value_partitions=tuple(
                map(ProposedValuePartition.from_json, data[AnswerKey.VALUE_PARTITIONS])
            ),
        )

    def declared(self) -> Dict[str, TermKind]:
        """
        :return: The kind of every term the proposal declares, by name; a name declared
            twice keeps its first kind.
        """
        declared: Dict[str, TermKind] = {}
        for name, kind in [
            *((proposed.name, TermKind.CLASS) for proposed in self.classes),
            *((partition.name, TermKind.CLASS) for partition in self.value_partitions),
            *((proposed.name, TermKind.PROPERTY) for proposed in self.properties),
            *(
                (member.name, TermKind.INDIVIDUAL)
                for partition in self.value_partitions
                for member in partition.members
            ),
        ]:
            declared.setdefault(name, kind)
        return declared

    def names(self) -> Tuple[str, ...]:
        """
        :return: The name of every term the proposal declares, in order, repeated ones
            included.
        """
        return (
            *(proposed.name for proposed in self.classes),
            *(partition.name for partition in self.value_partitions),
            *(proposed.name for proposed in self.properties),
            *(
                member.name
                for partition in self.value_partitions
                for member in partition.members
            ),
        )
