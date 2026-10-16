"""
Which fields of an action designator fill which parts of the action core's frame: a
participant slot, described by its class, or a parameter dimension, described by its
kind of quantity and the slots it concerns, so that a mapping kept as data holds for
every core hypothesis of the action, whose fragments are named anew each time. A mapping
is checked against the code ontology, which says what each field of a designator holds.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

from rdflib import OWL, RDF, RDFS, Graph, URIRef
from typing_extensions import TYPE_CHECKING, Any, Mapping, Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontologies import (
    ONTOLOGIES_DIRECTORY,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary

if TYPE_CHECKING:
    from llm_reasoner.knowledge.ontologies.described_actions import DescribedAction

CODE_ONTOLOGY_DIRECTORY = ONTOLOGIES_DIRECTORY / "cram_classes"
"""
Where the ontology of the code's classes is kept.
"""

CODE_ONTOLOGY_FILE = CODE_ONTOLOGY_DIRECTORY / "cram-classes.ttl"
"""
The ontology of the code's classes, one class per designator and one property per field.
"""

CODE_ONTOLOGY_SELECTION_FILE = CODE_ONTOLOGY_DIRECTORY / "cram-classes.json"
"""
What the ontology of the code's classes was generated from, its namespace included.
"""

CODE_ONTOLOGY_NAMESPACE_KEY = "namespace"
"""
The key of the namespace in the selection file.
"""

FIELD_SEPARATOR = "."
"""
What separates a class's name from a field's name in the IRI of the field's property.
"""

# %% what a field of a designator holds


class CodeFieldKind(StrEnum):
    """
    What a field of a designator holds, as the code ontology says.
    """

    OBJECT = "object"
    """
    An object of a class, such as a container.
    """

    ENUMERATION = "enumeration"
    """
    A member of an enumeration, such as an arm.
    """

    VALUE = "value"
    """
    A plain value, such as a number.
    """


@dataclass(frozen=True)
class CodeOntology:
    """
    The ontology of the code's classes, read for the fields of a designator.
    """

    graph: Graph
    """
    The ontology.
    """

    namespace: str
    """
    The namespace its classes and properties are named in.
    """

    @classmethod
    @cache
    def shipped(cls) -> CodeOntology:
        """
        :return: The code ontology shipped with the package, read once.
        """
        return cls(
            graph=Graph().parse(CODE_ONTOLOGY_FILE, format="turtle"),
            namespace=json.loads(CODE_ONTOLOGY_SELECTION_FILE.read_text())[
                CODE_ONTOLOGY_NAMESPACE_KEY
            ],
        )

    def fields_of(self, class_name: str) -> dict[str, CodeFieldKind]:
        """
        :param class_name: The module and class name of a designator.
        :return: What each of its fields holds, those of its base classes included, by
            field name; none for a class the ontology lacks.
        """
        fields: dict[str, CodeFieldKind] = {}
        for owner in self.class_and_bases(URIRef(self.namespace + class_name)):
            for prop in self.graph.subjects(RDFS.domain, owner):
                name = str(prop)[len(str(owner) + FIELD_SEPARATOR) :]
                fields.setdefault(name, self.kind_of(prop))
        return fields

    def class_and_bases(self, owner: URIRef) -> list[URIRef]:
        """
        :return: A class and every class it specialises, the class first.
        """
        found = [owner]
        for current in found:
            found.extend(
                base
                for base in self.graph.objects(current, RDFS.subClassOf)
                if isinstance(base, URIRef) and base not in found
            )
        return found

    def kind_of(self, prop: URIRef) -> CodeFieldKind:
        """
        :return: What the field a property stands for holds.
        """
        if (prop, RDF.type, OWL.DatatypeProperty) in self.graph:
            return CodeFieldKind.VALUE
        value_class = self.graph.value(prop, RDFS.range)
        if value_class is not None and any(self.graph.subjects(RDF.type, value_class)):
            return CodeFieldKind.ENUMERATION
        return CodeFieldKind.OBJECT


class FrameFieldProblemKind(StrEnum):
    """
    Why a field may not fill the part of the frame a mapping says it fills.
    """

    UNKNOWN_FIELD = "unknown_field"
    """
    The designator has no field of that name.
    """

    SLOT_FOR_VALUE = "slot_for_value"
    """
    A participant slot is filled by a field holding no object.
    """

    DIMENSION_FOR_OBJECT = "dimension_for_object"
    """
    A parameter dimension is filled by a field holding an object.
    """


# %% the parts of a frame a field fills


class FramePartKey(StrEnum):
    """
    The keys of a part of the frame, as a mapping kept as data writes it.
    """

    SLOT = "slot"
    QUANTITY = "quantity"
    ABOUT = "about"
    RELATIVE_TO = "relative_to"


@dataclass
class UnknownFrameTerm(LLMReasonerError):
    """
    A mapping names a slot class or a kind of quantity the action core does not have.
    """

    term: str
    """
    The name the mapping uses.
    """

    def __post_init__(self):
        super().__init__(
            f"the action core has no slot class or quantity kind '{self.term}'"
        )


def slot_or_part_of(slot: Optional[DescriptionEntity], kind: type) -> bool:
    """
    :return: Whether a slot is of a class, or is a part of a slot of that class.
    """
    if isinstance(slot, kind):
        return True
    return isinstance(slot, action_core.PartSlot) and isinstance(
        slot.is_part_of_slot, kind
    )


class FramePartDescription(ABC):
    """
    A part of the action core's frame, described in the core's vocabulary.
    """

    @abstractmethod
    def fragments_in(self, frame: action_core.ActionFrame) -> list[DescriptionEntity]:
        """
        :return: The fragments of a frame this description describes.
        """

    @property
    @abstractmethod
    def accepted_field_kinds(self) -> frozenset[CodeFieldKind]:
        """
        What a field filling this part may hold.
        """

    @property
    @abstractmethod
    def mismatch(self) -> FrameFieldProblemKind:
        """
        Why a field holding anything else may not fill this part.
        """


@dataclass(frozen=True)
class SlotDescription(FramePartDescription):
    """
    A participant slot of the frame, by its class.
    """

    slot: type[action_core.FrameSlot]
    """
    The class of the slot.
    """

    def fragments_in(self, frame: action_core.ActionFrame) -> list[DescriptionEntity]:
        return [slot for slot in frame.defines_slot if isinstance(slot, self.slot)]

    @property
    def accepted_field_kinds(self) -> frozenset[CodeFieldKind]:
        return frozenset({CodeFieldKind.OBJECT})

    @property
    def mismatch(self) -> FrameFieldProblemKind:
        return FrameFieldProblemKind.SLOT_FOR_VALUE


@dataclass(frozen=True)
class DimensionDescription(FramePartDescription):
    """
    A parameter dimension of the frame, by the kind of quantity it sets, the slot it is
    about and, when it is measured from one, the slot it is measured from; a part of a
    slot counts as the slot.
    """

    quantity: action_core.QuantityKind
    """
    The kind of quantity the choice sets.
    """

    about: type[action_core.FrameSlot]
    """
    The class of the slot whose filler the choice concerns.
    """

    relative_to: Optional[type[action_core.FrameSlot]] = None
    """
    The class of the slot the choice is measured from, when it is measured from one.
    """

    def fragments_in(self, frame: action_core.ActionFrame) -> list[DescriptionEntity]:
        return [
            dimension
            for dimension in frame.leaves_open
            if dimension.has_quantity_kind is self.quantity
            and slot_or_part_of(dimension.about_slot, self.about)
            and (
                self.relative_to is None
                or slot_or_part_of(dimension.relative_to_slot, self.relative_to)
            )
        ]

    @property
    def accepted_field_kinds(self) -> frozenset[CodeFieldKind]:
        return frozenset({CodeFieldKind.VALUE, CodeFieldKind.ENUMERATION})

    @property
    def mismatch(self) -> FrameFieldProblemKind:
        return FrameFieldProblemKind.DIMENSION_FOR_OBJECT


def frame_part_from_json(stored: Mapping[str, Any]) -> FramePartDescription:
    """
    :return: The part of the frame a mapping kept as data describes.
    :raises UnknownFrameTerm: When it names a slot class or a quantity kind the core
        lacks.
    """
    if FramePartKey.SLOT in stored:
        return SlotDescription(slot=core_slot_class(stored[FramePartKey.SLOT]))
    quantity = stored[FramePartKey.QUANTITY]
    if quantity not in action_core.QuantityKind.__members__:
        raise UnknownFrameTerm(term=quantity)
    return DimensionDescription(
        quantity=action_core.QuantityKind[quantity],
        about=core_slot_class(stored[FramePartKey.ABOUT]),
        relative_to=(
            core_slot_class(stored[FramePartKey.RELATIVE_TO])
            if FramePartKey.RELATIVE_TO in stored
            else None
        ),
    )


def core_slot_class(name: str) -> type[action_core.FrameSlot]:
    """
    :return: The slot class of the action core of the given name.
    :raises UnknownFrameTerm: When the core has no slot class of that name.
    """
    vocabulary = Vocabulary()
    named = vocabulary.class_named(name)
    if (
        named is None
        or vocabulary.module_file_of(named) is not ActionPerspectiveFile.ACTION_CORE
        or not issubclass(named, action_core.FrameSlot)
    ):
        raise UnknownFrameTerm(term=name)
    return named


# %% checking a mapping


@dataclass(frozen=True)
class FrameFieldProblem:
    """
    A field that may not fill the part of the frame a mapping says it fills.
    """

    field_name: str
    """
    The field.
    """

    kind: FrameFieldProblemKind
    """
    Why it may not.
    """


@dataclass(frozen=True)
class FrameFieldChecks:
    """
    Checks a designator's mapping to the frame against what its fields hold.
    """

    code_ontology: CodeOntology
    """
    Says what each field of a designator holds.
    """

    def problems(self, described: DescribedAction) -> list[FrameFieldProblem]:
        """
        :return: The fields that may not fill the parts of the frame the mapping says
            they fill, in the mapping's order.
        """
        fields = self.code_ontology.fields_of(described.designator)
        problems = []
        for field_name, part in described.frame_fields.items():
            if field_name not in fields:
                problems.append(
                    FrameFieldProblem(field_name, FrameFieldProblemKind.UNKNOWN_FIELD)
                )
            elif fields[field_name] not in part.accepted_field_kinds:
                problems.append(FrameFieldProblem(field_name, part.mismatch))
        return problems
