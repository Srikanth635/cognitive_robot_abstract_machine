"""
The action designator the Designator tab shows: the slots of coraplex's equivalent action,
followed by the generated hypothesis about the action, as one tree of descriptions in the
manner of a CRAM designator -- ``(an action (type pouring) (source-container ?x) ...)``.

The hypothesis is described by walking its fragments along their ontology properties,
starting at its structure: the roles first, then the phases in the order they run, then
everything else. Each fragment is described where it is first reached and referred to
by its identifier everywhere after, so the tree holds every fragment once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from enum import StrEnum
from operator import attrgetter

from llm_reasoner.knowledge.access import (
    ConceptReference,
    DescriptionEntity,
    HypothesisView,
    is_inferred,
    PropertyTerm,
)
from llm_reasoner.resolver.reflection import SlotKind
from typing_extensions import Any, Optional, Union

from cram_viz.designator_knowledge import DesignatorKnowledge, DesignatorSlot
from cram_viz.hypothesis_knowledge.ontology_backing import OntologyBacking

ACTION_FAMILY = "action"
"""
The family the described action belongs to.
"""

ACTION_SUFFIX = "Action"
"""
The suffix coraplex action class names end in, left out of the action's type.
"""

LEADING_FIELDS = ("defines_role", "defines_phase")
"""
The fields of a structure described before all others: who takes part, then the phases.
"""

# %% tree


class ValueKind(StrEnum):
    """
    What a leaf of the designator stands for.
    """

    LITERAL = "literal"
    """
    A number, a truth value or a text.
    """

    VARIABLE = "variable"
    """
    A slot of the action left open, to be bound when the action is resolved.
    """

    OBJECT = "object"
    """
    A scene object a slot of the action is bound to.
    """

    TERM = "term"
    """
    A term of an ontology: an enumerated individual or a cited concept.
    """

    REFERENCE = "reference"
    """
    A fragment of the hypothesis described elsewhere in the designator.
    """


@dataclass(frozen=True)
class DesignatorValue:
    """
    A leaf of the designator.
    """

    kind: ValueKind
    """
    What the value stands for.
    """

    text: str
    """
    The value as written in Lisp.
    """

    data: Any
    """
    The value as written in JSON.
    """

    reference: str = ""
    """
    The identifier of the description the value refers to, if any.
    """

    note: str = ""
    """
    What the value is, written beside it as a comment.
    """

    def payload(self) -> dict[str, Any]:
        """
        :return: The value for the viewer.
        """
        return {
            "node": "value",
            "kind": self.kind.value,
            "text": self.text,
            "data": self.data,
            "reference": self.reference,
            "note": self.note,
        }


@dataclass
class DesignatorProperty:
    """
    One property of a description, with its values.
    """

    key: str
    """
    The property's name in Lisp spelling, such as ``aims-for``.
    """

    values: list[Union[DesignatorValue, DesignatorDescription]]
    """
    The values: leaves, or descriptions of fragments first reached here.
    """

    many: bool = False
    """
    Whether the property may hold several values, so JSON writes its values as a list.
    """

    def payload(self) -> dict[str, Any]:
        """
        :return: The property for the viewer.
        """
        return {
            "key": self.key,
            "many": self.many,
            "values": [value.payload() for value in self.values],
        }


@dataclass
class DesignatorDescription:
    """
    A description in the designator, such as ``(an action-phase ...)``.
    """

    kind: str
    """
    The family the described thing belongs to, such as ``action-phase``.
    """

    identifier: str
    """
    The identifier references to the description use.
    """

    type: str
    """
    The class of the described thing: an ontology class as a prefixed term, or the
    action's type.
    """

    properties: list[DesignatorProperty] = field(default_factory=list)
    """
    What the description states.
    """

    note: str = ""
    """
    What the description is, written beside it as a comment.
    """

    @property
    def article(self) -> str:
        """
        The indefinite article a description of the family begins with.
        """
        return "an" if self.kind[:1] in "aeiou" else "a"

    def payload(self) -> dict[str, Any]:
        """
        :return: The description for the viewer.
        """
        return {
            "node": "description",
            "article": self.article,
            "kind": self.kind,
            "id": self.identifier,
            "type": self.type,
            "note": self.note,
            "properties": [entry.payload() for entry in self.properties],
        }


def lisp_name(name: str) -> str:
    """
    :return: A Python or class name in Lisp spelling, such as ``aims-for`` or
        ``action-phase``.
    """
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return words.replace("_", "-").lower()


# %% building


@dataclass
class DesignatorDescriber:
    """
    Describes an action: the slots of coraplex's equivalent action, then the generated
    hypothesis about it.
    """

    view: Optional[HypothesisView]
    """
    The active hypothesis, or ``None`` when nothing has been generated.
    """

    backing: OntologyBacking = field(default_factory=OntologyBacking)
    """
    Writes ontology terms with their prefixes.
    """

    described: set[int] = field(default_factory=set)
    """
    The identities of the fragments already described.
    """

    def describe(self, knowledge: DesignatorKnowledge) -> DesignatorDescription:
        """
        :return: The action designator: its slots, then the hypothesis's instruction
            and structure when there is a hypothesis.
        """
        properties = [self.slot_property(slot) for slot in knowledge.slots]
        properties.extend(self.resolution_properties(knowledge))
        if self.view is not None:
            properties.append(
                DesignatorProperty(
                    key="instruction",
                    values=[self.literal(self.view.record.instruction)],
                )
            )
            properties.append(
                DesignatorProperty(
                    key="hypothesis", values=[self.description(self.view.root)]
                )
            )
        return DesignatorDescription(
            kind=ACTION_FAMILY,
            identifier=knowledge.designator_name,
            type=lisp_name(knowledge.designator_name.removesuffix(ACTION_SUFFIX)),
            properties=properties,
            note=f"coraplex {knowledge.designator_name}",
        )

    # %% the action's slots

    def slot_property(self, slot: DesignatorSlot) -> DesignatorProperty:
        """
        :return: One slot of the action with its value, noting the role it fills and
            the hypothesis's filler of that role.
        """
        text = slot.lisp_value
        notes = [slot.type_name]
        reference = ""
        filler = self.filler_of(slot)
        if filler is not None:
            reference = self.view.citation_of(filler).fragment_name
            notes.append(
                f"fills {self.backing.prefixed(slot.role.term.iri)}: "
                f"{filler.label or reference}"
            )
        if slot.resolved_candidate is not None:
            notes.append(f"scene object {slot.resolved_candidate.label}, illustrative")
        if slot.resolved_value is not None:
            notes.append("resolved by the resolver")
            kind = (
                ValueKind.OBJECT
                if slot.kind is SlotKind.REFERENT
                else ValueKind.LITERAL
            )
        elif text.startswith("?"):
            kind = ValueKind.VARIABLE
        elif slot.resolved_candidate is not None:
            kind = ValueKind.OBJECT
        else:
            kind = ValueKind.LITERAL
        return DesignatorProperty(
            key=slot.lisp_keyword.removeprefix(":"),
            values=[
                DesignatorValue(
                    kind=kind,
                    text=text,
                    data=(
                        slot.default
                        if kind is ValueKind.LITERAL and slot.resolved_value is None
                        else text
                    ),
                    reference=reference,
                    note="; ".join(notes),
                )
            ],
        )

    def resolution_properties(
        self, knowledge: DesignatorKnowledge
    ) -> list[DesignatorProperty]:
        """
        :return: What the resolver's resolution states beyond the slots: the
            designator resolved, why the values were chosen and what they rest on;
            nothing without a resolution.
        """
        resolution = knowledge.resolution
        if resolution is None or not resolution.actions:
            return []
        action, *_ = resolution.actions
        return [
            DesignatorProperty(
                key="resolved-from", values=[self.literal(resolution.query_on_one_line)]
            ),
            DesignatorProperty(key="why", values=[self.literal(action.why)]),
            DesignatorProperty(
                key="rests-on",
                values=[
                    self.literal(f"{ground.source}: {ground.name}")
                    for ground in action.grounds
                ],
                many=True,
            ),
        ]

    def filler_of(self, slot: DesignatorSlot) -> Optional[DescriptionEntity]:
        """
        :return: The hypothesis's filler of the role a slot fills, if any.
        """
        if self.view is None or slot.role is None:
            return None
        return next(
            (role for role in self.view.root.defines_role if type(role) is slot.role),
            None,
        )

    # %% the hypothesis

    def description(self, fragment: DescriptionEntity) -> DesignatorDescription:
        """
        :return: The description of a fragment, describing the fragments it reaches
            first here and referring to the others.
        """
        self.described.add(id(fragment))
        properties = []
        if fragment.label:
            properties.append(
                DesignatorProperty(key="label", values=[self.literal(fragment.label)])
            )
        for name in self.field_order(fragment):
            held: Any = attrgetter(name)(fragment)
            values = [
                self.value(value)
                for value in self.ordered(
                    fragment, name, held if isinstance(held, list) else [held]
                )
                if value is not None
            ]
            if values:
                properties.append(
                    DesignatorProperty(
                        key=lisp_name(name), values=values, many=isinstance(held, list)
                    )
                )
        return DesignatorDescription(
            kind=lisp_name(self.family_of(type(fragment)).__name__),
            identifier=self.view.citation_of(fragment).fragment_name,
            type=self.backing.prefixed(type(fragment).term.iri),
            properties=properties,
        )

    def field_order(self, fragment: DescriptionEntity) -> list[str]:
        """
        :return: The fields stating the fragment's ontology properties, those a
            structure leads with first.
        """
        names = [
            declared.name
            for declared in fields(fragment)
            if PropertyTerm.of_field(type(fragment), declared.name) is not None
            and not is_inferred(type(fragment), declared.name)
        ]
        leading = [name for name in LEADING_FIELDS if name in names]
        return leading + [name for name in names if name not in leading]

    def ordered(
        self, fragment: DescriptionEntity, name: str, values: list[Any]
    ) -> list[Any]:
        """
        :return: The values of a field, the phases of the hypothesis's structure in the
            order they run.
        """
        if fragment is not self.view.root or name != "defines_phase":
            return values
        running = self.view.phases()
        return sorted(
            values,
            key=lambda phase: next(
                index for index, ran in enumerate(running) if ran is phase
            ),
        )

    def value(self, value: Any) -> Union[DesignatorValue, DesignatorDescription]:
        """
        :return: A value of a fragment: the description of a fragment reached for the
            first time, a reference to one described before, an ontology term, or a
            literal.
        """
        if isinstance(value, DescriptionEntity):
            if id(value) not in self.described:
                return self.description(value)
            identifier = self.view.citation_of(value).fragment_name
            return DesignatorValue(
                kind=ValueKind.REFERENCE,
                text="#" + identifier,
                data={"$ref": identifier},
                reference=identifier,
                note=value.label or type(value).__name__,
            )
        if isinstance(value, ConceptReference):
            return self.term(value.iri)
        if isinstance(value, StrEnum):
            return self.term(value.value)
        return self.literal(value)

    def term(self, iri: str) -> DesignatorValue:
        """
        :return: An ontology term as a value.
        """
        prefixed = self.backing.prefixed(iri)
        return DesignatorValue(kind=ValueKind.TERM, text=prefixed, data=prefixed)

    @staticmethod
    def literal(value: Any) -> DesignatorValue:
        """
        :return: A literal in Lisp and JSON spelling.
        """
        if isinstance(value, bool):
            text = "T" if value else "NIL"
        elif isinstance(value, str):
            text = '"' + value.replace('"', '\\"') + '"'
        else:
            text = str(value)
        return DesignatorValue(kind=ValueKind.LITERAL, text=text, data=value)

    @staticmethod
    def family_of(described: type[DescriptionEntity]) -> type[DescriptionEntity]:
        """
        :return: The most general ontology class of a vocabulary class: the first class
            in its order of inheritance that specialises nothing but the base of every
            description.
        """
        return next(
            ancestor
            for ancestor in described.__mro__
            if DescriptionEntity in ancestor.__bases__
        )
