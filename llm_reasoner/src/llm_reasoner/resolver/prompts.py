"""
The prompts a model resolves a query from: the system prompt stating its role, what each
section of a request means and the rules it answers by, and the request itself, made of
tagged sections whose data is JSON.
"""

from __future__ import annotations

import enum
import inspect
import json
from dataclasses import dataclass, field
from enum import StrEnum

from krrood.entity_query_language.backends import QueryBackend
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from typing_extensions import Any, Dict, List, Optional, Sequence

from llm_reasoner.resolver.context import CandidateRenderer
from llm_reasoner.resolver.exceptions import RetryableModelFailure
from llm_reasoner.resolver.prompt_format import (
    SECTION_SEPARATOR,
    AcceptedType,
    AcceptKey,
    BlockAttribute,
    CandidateKey,
    PromptBlock,
    PromptRule,
    PromptSection,
    RuleName,
    SchemaKey,
    SchemaType,
    SlotKey,
    render_blocks,
)
from llm_reasoner.resolver.referents import IntegerTokens, ReferentNaming
from llm_reasoner.resolver.reflection import Slot, SlotKind
from llm_reasoner.world.access import is_individual

JSON_INDENT = 2
"""
The indentation of the JSON data a prompt holds.
"""

REJECTED_ANSWER_LENGTH = 2000
"""
The most characters of a rejected answer a repair repeats.
"""

# %% answer keys


class RequestAnswerKey(StrEnum):
    """
    The keys of the answers that speak for the whole request rather than fill slots.
    """

    UNSATISFIABLE = "unsatisfiable"
    """
    Says that the request's conditions cannot all hold together, and why.
    """

    WHY = "why"
    """
    Says, in one sentence, why an element's values were chosen; it is kept with the
    answer for people to read and never decoded.
    """


# %% system prompt

ROLE = "Fill the open slots of a partially specified object."
"""
What the model is asked to do.
"""

INPUT_MEANINGS = {
    PromptSection.REQUEST: 'the query in English; "Ellipsis" marks an open slot, never a '
    "value.",
    PromptSection.TASK: "what the caller is trying to achieve: intent to serve, not fact "
    "about the world.",
    PromptSection.SITUATION: "what is currently true: fact.",
    PromptSection.ACTION: "the action the requested values are for, as its own "
    f"<{PromptSection.REQUEST}> and <{PromptSection.GIVEN}>: context to read, not slots "
    "to answer.",
    PromptSection.SOURCE: "the definition of a type holding slots, showing what it does "
    "with each value.",
    PromptSection.DESCRIPTION: "the docstring of a type holding slots.",
    PromptSection.SLOTS: "one entry per open slot: its path, type, meaning, what it "
    "accepts, its default and its candidates.",
    PromptSection.GIVEN: "the values the query fixes.",
    PromptSection.OUTPUT: "the JSON schema the answer follows.",
    PromptSection.REJECTED_ANSWER: "an earlier answer that was refused.",
    PromptSection.REJECTION: "why that answer was refused.",
}
"""
What each section of a request means, in the order a request may hold them.
"""

ANSWER_FORMAT_RULE = PromptRule(
    name=RuleName.ANSWER_FORMAT,
    text=f"reply with one JSON array following <{PromptSection.OUTPUT}>; no prose, no "
    "markdown fences, no trailing commas.",
)
"""
The rule stating how the answer is written as a whole.
"""

ELEMENT_KEYS_RULE = PromptRule(
    name=RuleName.ELEMENT_KEYS,
    text=f'each element holds every "{SlotKey.PATH}" of <{PromptSection.SLOTS}> and '
    f'"{RequestAnswerKey.WHY}", and no other key.',
)
"""
The rule stating which keys each element holds.
"""

VALUE_SLOT_RULE = PromptRule(
    name=RuleName.VALUE_SLOT,
    text=f'answer a slot without "{SlotKey.CANDIDATES}" with one value its '
    f'"{SlotKey.ACCEPTS}" lists: an enum member by its name, a {AcceptedType.NUMBER}, '
    f"{AcceptedType.STRING} or {AcceptedType.BOOLEAN} as JSON, and "
    f"{AcceptedType.NULL} only where listed; never a JSON object. When several values "
    f'fit equally, give one and say so in "{RequestAnswerKey.WHY}".',
)
"""
The rule stating how a slot taking a plain value is answered.
"""

ESTABLISHED_VALUES_RULE = PromptRule(
    name=RuleName.ESTABLISHED_VALUES,
    text="the values asked for are facts, not choices: answer each value slot with what "
    f"<{PromptSection.ACTION}>, <{PromptSection.GIVEN}>, <{PromptSection.TASK}>, "
    f"<{PromptSection.SITUATION}> or the world establishes, and {AcceptedType.NULL} "
    "where nothing does; never fill one by convention or as the usual case.",
)
"""
The rule stating that value slots ask for what is established, and nothing else.
"""

DEFAULT_RULE = PromptRule(
    name=RuleName.DEFAULT,
    text=f'a slot\'s "{SlotKey.DEFAULT}" is the value its class uses when none is given.',
)
"""
The rule stating what a slot's default stands for.
"""

UNION_RULE = PromptRule(
    name=RuleName.UNION,
    text=f'a slot whose "{SlotKey.ACCEPTS}" lists several types takes a value of any one '
    "of them; choose the one that fits the situation.",
)
"""
The rule stating how a slot accepting several types is answered.
"""

KIND_SLOT_RULE = PromptRule(
    name=RuleName.KIND_SLOT,
    text=f'a slot whose "{SlotKey.TYPE}" is a kind takes the "{CandidateKey.TOKEN}" of '
    f'one of its "{CandidateKey.LISTED}" kinds, as a JSON {AcceptedType.NUMBER}.',
)
"""
The rule stating how a slot asking for a kind is answered.
"""

UNSATISFIABLE_RULE = PromptRule(
    name=RuleName.UNSATISFIABLE,
    text=f"when the conditions of <{PromptSection.REQUEST}> cannot all hold together, "
    f'answer [{{"{RequestAnswerKey.UNSATISFIABLE}": "<why they cannot all hold>"}}] '
    "instead of breaking one.",
)
"""
The rule letting the model say that the request's conditions cannot all hold.
"""

CONDITIONS_RULE = PromptRule(
    name=RuleName.CONDITIONS,
    text=f"respect every condition of <{PromptSection.REQUEST}>; a condition forcing a "
    "slot's value decides it.",
)
"""
The rule stating how the request's conditions bind the answer.
"""

GIVEN_RULE = PromptRule(
    name=RuleName.GIVEN,
    text=f"the values in <{PromptSection.GIVEN}> are fixed; let them inform your choices "
    "and never restate them.",
)
"""
The rule stating how the values the query fixes are used.
"""

FEWER_CANDIDATES_RULE = PromptRule(
    name=RuleName.CANDIDATES,
    text="give the elements best first, each a good answer on its own; never vary a slot "
    "only to make the elements differ, and when fewer good answers exist than asked "
    "for, give only those.",
)
"""
The rule asking for distinct candidates while letting the model give fewer than asked.
"""

WHY_RULE = PromptRule(
    name=RuleName.WHY,
    text=f'"{RequestAnswerKey.WHY}" holds one sentence on why you chose what the request '
    "left open.",
)
"""
The rule stating what the explanation of an element says.
"""

REPAIR_RULE = PromptRule(
    name=RuleName.REPAIR,
    text=f"when <{PromptSection.REJECTED_ANSWER}> and <{PromptSection.REJECTION}> "
    "follow the request, answer again with what the rejection names corrected.",
)
"""
The rule stating how a refused answer is answered again.
"""

# %% type descriptions


class OpenValueMeaning(enum.Enum):
    """What the model's answer to an open value slot stands for."""

    CHOICE = enum.auto()
    """A value chosen to serve the request."""

    FACT = enum.auto()
    """
    A value the request, its context or the world establishes, nothing where none does.
    """

    @property
    def rules(self) -> List[PromptRule]:
        """:return: The rules stating how slots of this meaning are answered."""
        return OPEN_VALUE_RULES[self]


OPEN_VALUE_RULES: Dict[OpenValueMeaning, List[PromptRule]] = {
    OpenValueMeaning.CHOICE: [],
    OpenValueMeaning.FACT: [ESTABLISHED_VALUES_RULE],
}
"""
The rules stating how open value slots of each meaning are answered, beyond the rule
for every value slot.
"""


class SourceDetail(enum.Enum):
    """How much of the target type's definition is put in the prompt."""

    NONE = enum.auto()
    """Nothing. The slots section alone describes what must be produced."""

    SIGNATURE = enum.auto()
    """
    The docstring of every class contributing a slot.

    The slots section already carries each field's name, type, docstring and default, so
    the full definition would repeat all of it and bury the repetition in method bodies
    describing how the type is *used* rather than how to choose its values. What the
    slots cannot carry is what each class *is*, which matters most for nested
    descriptions where the outer type alone explains little.
    """

    FULL = enum.auto()
    """
    The complete source of every type contributing a slot, which shows what the type does
    with each value: a field's docstring may name the value without saying how the type
    uses it.
    """


# %% prompt assembly


@dataclass
class PromptBuilder:
    """
    Assembles the prompts from three layers.

    The request and the slots are derived from the query alone; the context sections
    carry everything from outside it, each in its own section so that intent and world
    state are not read as the same kind of claim.
    """

    renderer: CandidateRenderer
    """Describes the candidates of referent slots."""

    referent_naming: ReferentNaming = field(default_factory=IntegerTokens)
    """How the candidates of referent slots are named and answered."""

    source_detail: SourceDetail = SourceDetail.SIGNATURE
    """How much of the target type's definition to include."""

    open_values: OpenValueMeaning = OpenValueMeaning.CHOICE
    """What an answer to an open value slot stands for."""

    # %% the system prompt

    def rules(self, further_rules: Sequence[PromptRule] = ()) -> List[PromptRule]:
        """
        :param further_rules: Rules of how the model is asked, such as where its answer
            goes when it may call tools.
        :return: Every rule the model answers by, the referent rules matching how
            candidates are named.
        """
        return [
            ANSWER_FORMAT_RULE,
            ELEMENT_KEYS_RULE,
            VALUE_SLOT_RULE,
            *self.open_values.rules,
            DEFAULT_RULE,
            UNION_RULE,
            KIND_SLOT_RULE,
            *self.referent_naming.rules,
            UNSATISFIABLE_RULE,
            CONDITIONS_RULE,
            GIVEN_RULE,
            FEWER_CANDIDATES_RULE,
            WHY_RULE,
            REPAIR_RULE,
            *further_rules,
        ]

    def system_prompt(self, further_rules: Sequence[PromptRule] = ()) -> str:
        """
        :param further_rules: Rules of how the model is asked.
        :return: The system prompt: the model's role, what each section of a request
            means, and the rules it answers by.
        """
        return render_blocks(
            [
                PromptBlock(section=PromptSection.ROLE, body=ROLE),
                PromptBlock(
                    section=PromptSection.INPUTS,
                    body="\n".join(
                        f"- <{section}>: {meaning}"
                        for section, meaning in INPUT_MEANINGS.items()
                    ),
                ),
                PromptBlock(
                    section=PromptSection.RULES,
                    body="\n".join(rule.render() for rule in self.rules(further_rules)),
                ),
            ]
        )

    # %% the request

    def build(
        self,
        expression: Match,
        slots: Sequence[Slot],
        context_sections: Sequence[PromptBlock],
        number_of_candidates: int,
        backend: Optional[QueryBackend] = None,
    ) -> str:
        """
        :param expression: The match being resolved.
        :param slots: Every reflected slot, bound and open.
        :param context_sections: The sections from the context providers, in the order
            they should be read.
        :param number_of_candidates: How many alternatives to request at most.
        :param backend: Decides the verb the request is verbalized with.
        :return: The user prompt.
        """
        open_slots = [slot for slot in slots if slot.kind is not SlotKind.BOUND]
        bound_slots = [slot for slot in slots if slot.kind is SlotKind.BOUND]
        blocks = [
            *self.describe_types(expression.type, slots),
            PromptBlock(
                section=PromptSection.REQUEST,
                body=self.verbalize_request(expression, backend),
            ),
            *context_sections,
            PromptBlock(
                section=PromptSection.SLOTS,
                body=self.as_json(
                    [self.slot_entry(slot, expression) for slot in open_slots]
                ),
            ),
        ]
        if bound_slots:
            blocks.append(
                PromptBlock(
                    section=PromptSection.GIVEN,
                    body=self.as_json(
                        [self.given_entry(slot, expression) for slot in bound_slots]
                    ),
                )
            )
        blocks.append(
            PromptBlock(
                section=PromptSection.OUTPUT,
                body=self.as_json(self.output_schema(open_slots, number_of_candidates)),
            )
        )
        return render_blocks(blocks)

    def describe_types(
        self, target_type: type, slots: Sequence[Slot]
    ) -> List[PromptBlock]:
        """
        Account for every type the model has to fill fields on.

        A nested description contributes slots owned by classes other than the one being
        constructed, and those classes explain themselves in their own docstrings. The
        outer type is described first, then each nested owner in the order its slots
        appear.

        :param target_type: The type being constructed.
        :param slots: Every reflected slot.
        :return: One section per type described, none when no account is wanted.
        """
        if self.source_detail is SourceDetail.NONE:
            return []
        if self.source_detail is SourceDetail.FULL:
            return [
                PromptBlock(
                    section=PromptSection.SOURCE,
                    body="```python\n"
                    + inspect.getsource(described).rstrip()
                    + "\n```",
                    attributes={BlockAttribute.TYPE: described.__name__},
                )
                for described in self.types_with_their_family(target_type, slots)
            ]
        return [
            PromptBlock(
                section=PromptSection.DESCRIPTION,
                body=inspect.getdoc(described).strip(),
                attributes={BlockAttribute.TYPE: described.__name__},
            )
            for described in self.types_to_describe(target_type, slots)
            if inspect.getdoc(described)
        ]

    @staticmethod
    def types_to_describe(target_type: type, slots: Sequence[Slot]) -> List[type]:
        """
        :param target_type: The type being constructed.
        :param slots: Every reflected slot.
        :return: The constructed type followed by every other class declaring a slot,
            without repetition, in the order their slots appear.
        """
        described: List[type] = [target_type]
        for slot in slots:
            if slot.owner not in described:
                described.append(slot.owner)
        return described

    @staticmethod
    def types_with_their_family(target_type: type, slots: Sequence[Slot]) -> List[type]:
        """
        :param target_type: The type being constructed.
        :param slots: Every reflected slot.
        :return: The types to describe, each followed by the classes it specialises that
            are declared in its own module, without repetition. A class declared beside
            a type is part of its family, and may define what the type does with its
            values; the shared foundations it rests on are declared elsewhere.
        """
        family: List[type] = []
        for described in PromptBuilder.types_to_describe(target_type, slots):
            for member in described.__mro__:
                if member.__module__ == described.__module__ and member not in family:
                    family.append(member)
        return family

    # %% the slots

    def slot_entry(self, slot: Slot, expression: Match) -> Dict[str, Any]:
        """
        :param slot: An open slot.
        :param expression: The match being resolved, passed to the renderer.
        :return: The slot as data: its path, type, meaning, what it accepts, its default
            and its candidates.
        """
        entry: Dict[str, Any] = {SlotKey.PATH: slot.path, SlotKey.TYPE: slot.type_name}
        if slot.docstring:
            entry[SlotKey.MEANING] = slot.docstring
        if slot.group_path is not None:
            entry[SlotKey.WITHIN] = self.group_entry(slot)
        if slot.kind is SlotKind.SORTAL:
            entry[SlotKey.ACCEPTS] = [IntegerTokens().accepted]
            entry[SlotKey.CANDIDATES] = IntegerTokens().candidates(
                slot, expression, self.renderer
            )
            return entry
        if slot.selects_from_domain:
            entry[SlotKey.ACCEPTS] = [
                self.referent_naming.accepted,
                *self.emptiness(slot),
            ]
            entry[SlotKey.CANDIDATES] = self.referent_naming.candidates(
                slot, expression, self.renderer
            )
            return entry
        entry[SlotKey.ACCEPTS] = self.accepts(slot)
        if slot.has_default:
            entry[SlotKey.DEFAULT] = self.answer_value(slot.default)
        return entry

    @staticmethod
    def group_entry(slot: Slot) -> Dict[str, Any]:
        """
        :param slot: A slot sitting inside a nested description.
        :return: The nested description it is filled through: its path, type and
            meaning.
        """
        group: Dict[str, Any] = {
            SlotKey.PATH: slot.group_path,
            SlotKey.TYPE: slot.group_type.__name__,
        }
        if slot.group_docstring:
            group[SlotKey.MEANING] = slot.group_docstring
        return group

    @staticmethod
    def accepts(slot: Slot) -> List[Dict[str, Any]]:
        """
        :param slot: A slot whose value is to be generated.
        :return: What the slot may be answered with, one entry per branch of a union,
            and nothing when the slot may be left empty.
        """
        return [
            PromptBuilder.accepted(candidate_type)
            for candidate_type in slot.decodable_types
        ] + PromptBuilder.emptiness(slot)

    @staticmethod
    def accepted(candidate_type: type) -> Dict[str, Any]:
        """
        :param candidate_type: One type a value may take.
        :return: What a value of that type is answered with.
        """
        if issubclass(candidate_type, enum.Enum):
            return {AcceptKey.ENUM: [member.name for member in candidate_type]}
        if candidate_type is bool:
            return {AcceptKey.TYPE: AcceptedType.BOOLEAN}
        if candidate_type in (int, float):
            return {AcceptKey.TYPE: AcceptedType.NUMBER}
        return {AcceptKey.TYPE: AcceptedType.STRING}

    @staticmethod
    def emptiness(slot: Slot) -> List[Dict[str, Any]]:
        """
        :param slot: A slot.
        :return: The entry accepting nothing when the slot may be left empty, else
            none.
        """
        if not slot.may_be_empty:
            return []
        return [{AcceptKey.TYPE: AcceptedType.NULL}]

    @staticmethod
    def value_hint(slot: Slot) -> str:
        """
        :param slot: A slot whose value is to be generated.
        :return: What the slot accepts, written as in the slots section.
        """
        return json.dumps(PromptBuilder.accepts(slot))

    def given_entry(self, slot: Slot, expression: Match) -> Dict[str, Any]:
        """
        :param slot: A slot the query fixes.
        :param expression: The match being resolved, passed to the renderer.
        :return: The slot's path and value as data: an individual or a kind described
            line by line, any other value as an answer writes it.
        """
        value = slot.bound_value
        if isinstance(value, type) or is_individual(value):
            given = self.renderer.render(value, expression).splitlines()
        else:
            given = self.answer_value(value)
        return {SlotKey.PATH: slot.path, SlotKey.VALUE: given}

    @staticmethod
    def answer_value(value: Any) -> Any:
        """
        :param value: A value a slot may take, such as its default.
        :return: The value as an answer gives it: an enum member by its name, a value
            JSON cannot hold by its representation, any other value as it is.
        """
        return json.loads(PromptBuilder.answer_form(value))

    @staticmethod
    def answer_form(value: Any) -> str:
        """
        :param value: A value a slot may take, such as its default.
        :return: The value as an answer writes it: an enum member by its name, any other
            value as JSON.
        """
        return json.dumps(
            value.name if isinstance(value, enum.Enum) else value, default=repr
        )

    # %% the output

    @staticmethod
    def output_schema(
        open_slots: Sequence[Slot], number_of_candidates: int
    ) -> Dict[str, Any]:
        """
        :param open_slots: The slots the model must answer for.
        :param number_of_candidates: How many alternatives to request at most.
        :return: The JSON schema of the answer: an array of at most that many objects,
            each holding every slot path and the explanation.
        """
        return {
            SchemaKey.TYPE: SchemaType.ARRAY,
            SchemaKey.MIN_ITEMS: 1,
            SchemaKey.MAX_ITEMS: number_of_candidates,
            SchemaKey.ITEMS: {
                SchemaKey.TYPE: SchemaType.OBJECT,
                SchemaKey.PROPERTIES: {
                    **{slot.path: {} for slot in open_slots},
                    RequestAnswerKey.WHY: {SchemaKey.TYPE: SchemaType.STRING},
                },
                SchemaKey.REQUIRED: [slot.path for slot in open_slots]
                + [RequestAnswerKey.WHY],
                SchemaKey.ADDITIONAL_PROPERTIES: False,
            },
        }

    @staticmethod
    def as_json(data: Any) -> str:
        """
        :return: Data as the JSON a prompt holds.
        """
        return json.dumps(data, indent=JSON_INDENT)

    # %% the request in English, and repairs

    @staticmethod
    def verbalize_request(
        expression: Match, backend: Optional[QueryBackend] = None
    ) -> str:
        """
        Render the whole query in English, constraints included.

        Unfilled leaves are rendered by krrood as the literal word "Ellipsis". The
        system prompt glosses that word as an open slot rather than a value, so the
        request can be shown in full instead of stripped down to its conditions.

        :param expression: The match being resolved.
        :param backend: Decides the opening verb.
        :return: The English rendering.
        """
        return verbalize_expression(expression.expression, backend=backend)

    @staticmethod
    def build_repair(
        previous_prompt: str, raw_response: str, error: RetryableModelFailure
    ) -> str:
        """
        :param previous_prompt: The prompt that produced the rejected response.
        :param raw_response: The rejected response.
        :param error: Why it was rejected.
        :return: The prompt followed by the rejected answer and, as data, why it was
            rejected.
        """
        repair = render_blocks(
            [
                PromptBlock(
                    section=PromptSection.REJECTED_ANSWER,
                    body=raw_response[:REJECTED_ANSWER_LENGTH],
                ),
                PromptBlock(
                    section=PromptSection.REJECTION,
                    body=PromptBuilder.as_json(error.details()),
                    attributes={BlockAttribute.KIND: type(error).__name__},
                ),
            ]
        )
        return previous_prompt + SECTION_SEPARATOR + repair
