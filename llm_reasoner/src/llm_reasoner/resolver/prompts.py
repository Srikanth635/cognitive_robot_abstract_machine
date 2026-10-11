from __future__ import annotations

import enum
import inspect
import json
from dataclasses import dataclass, field

from krrood.entity_query_language.backends import QueryBackend
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from typing_extensions import Any, List, Optional, Sequence, Tuple

from llm_reasoner.resolver.context import CandidateRenderer
from llm_reasoner.resolver.referents import (
    SLOT_NOTE_INDENT,
    IntegerTokens,
    ReferentNaming,
)
from llm_reasoner.resolver.reflection import Slot, SlotKind

# %% system prompt

SYSTEM_PROMPT_TEMPLATE = """\
You fill in the missing parameters of a partially specified object.

Reading the request:
- The request is stated in English, followed by a table of the slots you must fill.
- The word "Ellipsis" in the request marks a blank you must fill. It is not a value,
  and you must never answer with it.
- A section headed "The task" states what the caller is trying to achieve. Treat it as
  intent to be served, not as fact about the world.
- A section headed "Situation" states what is currently true. Treat it as fact.

Rules:
- Answer with a JSON array only. No prose, no markdown fences, no trailing commas.
- Each array element is an object whose keys are exactly the slot paths given to you.
- Fill every listed slot in every element. Add no other keys.
- Enum slots take the member name as a string.
- Kind slots take the integer token of one of the listed kinds.
{mode_rules}
- A slot offering several types accepts a value of any one of them; choose the type
  that fits the situation.
- Respect every stated constraint. When a constraint forces a slot's value, use the
  forced value rather than something merely plausible.
- Make the elements genuinely different from one another, best first.
"""


class SourceDetail(enum.Enum):
    """How much of the target type's definition is put in the prompt."""

    NONE = enum.auto()
    """Nothing. The slot table alone describes what must be produced."""

    SIGNATURE = enum.auto()
    """
    The docstring of every class contributing a slot.

    The slot table already carries each field's name, type, docstring and default, so
    the full definition would repeat all of it and bury the repetition in method bodies
    describing how the type is *used* rather than how to choose its values. What the
    table cannot carry is what each class *is*, which matters most for nested
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
    Assembles the prompt from three layers.

    The request and the schema are derived from the query alone; the context sections
    carry everything from outside it, each under its own heading so that intent and
    world state are not read as the same kind of claim.
    """

    renderer: CandidateRenderer
    """Describes the candidates of referent slots."""

    referent_naming: ReferentNaming = field(default_factory=IntegerTokens)
    """How the candidates of referent slots are named and answered."""

    source_detail: SourceDetail = SourceDetail.SIGNATURE
    """How much of the target type's definition to include."""

    def build(
        self,
        expression: Match,
        slots: Sequence[Slot],
        context_sections: Sequence[Tuple[str, str]],
        number_of_candidates: int,
        backend: Optional[QueryBackend] = None,
    ) -> str:
        """
        :param expression: The match being resolved.
        :param slots: Every reflected slot, bound and open.
        :param context_sections: Title and text pairs from the context providers, in
            the order they should be read.
        :param number_of_candidates: How many alternatives to request.
        :param backend: Decides the verb the request is verbalized with.
        :return: The user prompt.
        """
        target_type = expression.type
        open_slots = [slot for slot in slots if slot.kind is not SlotKind.BOUND]
        bound_slots = [slot for slot in slots if slot.kind is SlotKind.BOUND]

        sections: List[str] = [
            f"Generate {number_of_candidates} candidate "
            f"{target_type.__name__} object(s)."
        ]
        description = self.describe_types(target_type, slots)
        if description:
            sections.append(description)
        request = self.verbalize_request(expression, backend)
        if request:
            sections.append("## The request\n\n" + request)
        sections.extend(f"## {title}\n\n{text}" for title, text in context_sections)
        sections.append(
            "## Slots to fill\n\n" + self.render_slots(open_slots, expression)
        )
        if bound_slots:
            sections.append(
                "## Already known\n\n"
                + "\n".join(
                    self.hanging(
                        f"{slot.path} = ",
                        self.renderer.render(slot.bound_value, expression),
                    )
                    for slot in bound_slots
                )
                + "\n\nThese are given. Let them inform your choices; do not restate "
                "them."
            )
        sections.append(
            "## Output\n\n"
            f"A JSON array of {number_of_candidates} object(s). "
            "Keys of each object, exactly: "
            + ", ".join(f'"{slot.path}"' for slot in open_slots)
            + "."
        )
        return "\n\n".join(sections)

    def describe_types(self, target_type: type, slots: Sequence[Slot]) -> Optional[str]:
        """
        Account for every type the model has to fill fields on.

        A nested description contributes slots owned by classes other than the one being
        constructed, and those classes explain themselves in their own docstrings. The
        outer type is described first, then each nested owner in the order its slots
        appear.

        :param target_type: The type being constructed.
        :param slots: Every reflected slot.
        :return: The prompt's account of the types, or ``None`` when none is wanted.
        """
        if self.source_detail is SourceDetail.NONE:
            return None
        if self.source_detail is SourceDetail.FULL:
            return "\n\n".join(
                f"## Source of {described.__name__}\n\n```python\n"
                + inspect.getsource(described).rstrip()
                + "\n```"
                for described in self.types_with_their_family(target_type, slots)
            )
        described_types = [
            (described, inspect.getdoc(described))
            for described in self.types_to_describe(target_type, slots)
        ]
        paragraphs = [
            f"**{described.__name__}** -- {docstring.strip()}"
            for described, docstring in described_types
            if docstring
        ]
        if not paragraphs:
            return None
        return "## What these are\n\n" + "\n\n".join(paragraphs)

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

    def render_slots(self, slots: Sequence[Slot], expression: Match) -> str:
        """
        Render the slot table, gathering the slots of each nested description under a
        heading of their own.

        The nesting attribute contributes no slot, so its type and docstring would
        otherwise never reach the prompt -- which is where the sentence explaining what
        the nested description is *for* usually lives.

        :param slots: The slots the model must answer for, in reflection order.
        :param expression: The match being resolved, passed to the renderer.
        :return: The rendered table.
        """
        blocks: List[str] = []
        current_group: Optional[str] = ""
        for slot in slots:
            if slot.group_path != current_group:
                current_group = slot.group_path
                heading = self.render_group_heading(slot)
                if heading:
                    blocks.append(heading)
            blocks.append(self.render_slot(slot, expression))
        return "\n\n".join(blocks)

    @staticmethod
    def render_group_heading(slot: Slot) -> Optional[str]:
        """
        :param slot: The first slot of a nested description.
        :return: The heading introducing that description, or ``None`` for slots
            sitting directly on the constructed type.
        """
        if slot.group_path is None:
            return None
        lines = [f"{slot.group_path}   ({slot.group_type.__name__})"]
        if slot.group_docstring:
            lines.append(f'    "{slot.group_docstring}"')
        lines.append("    filled through the slots below:")
        return "\n".join(lines)

    def render_slot(self, slot: Slot, expression: Match) -> str:
        """
        :param slot: The slot to render.
        :param expression: The match being resolved, passed to the renderer.
        :return: The slot's prompt block.
        """
        lines = [f"{slot.path}   ({slot.type_name})"]
        if slot.docstring:
            lines.append(f'    "{slot.docstring}"')
        if slot.kind is SlotKind.SORTAL:
            lines.append(f"{SLOT_NOTE_INDENT}choose one kind by its integer token:")
            lines.extend(
                IntegerTokens().candidate_lines(slot, expression, self.renderer)
            )
            return "\n".join(lines)
        if slot.selects_from_domain:
            lines.append(self.referent_naming.candidate_heading)
            lines.extend(
                self.referent_naming.candidate_lines(slot, expression, self.renderer)
            )
            return "\n".join(lines)
        hint = self.value_hint(slot)
        if slot.has_default:
            hint += f"    (conventionally {self.answer_form(slot.default)})"
        lines.append(f"    {hint}")
        return "\n".join(lines)

    @staticmethod
    def candidate_prefix(token: int) -> str:
        """
        :param token: The integer token a candidate is chosen by.
        :return: What precedes the description of that candidate.
        """
        return IntegerTokens.candidate_prefix(token)

    @staticmethod
    def hanging(prefix: str, text: str) -> str:
        """
        :param prefix: What precedes the first line.
        :param text: Text that may span several lines.
        :return: The text after the prefix, its later lines aligned with the first.
        """
        return IntegerTokens.hanging(prefix, text)

    def system_prompt(self, further_rules: Sequence[str] = ()) -> str:
        """
        :param further_rules: Rules of how the model is asked, such as where its answer
            goes when it may call tools.
        :return: The system prompt, its rule for referent slots matching how their
            candidates are named.
        """
        return SYSTEM_PROMPT_TEMPLATE.format(
            mode_rules="\n".join([*self.referent_naming.rules, *further_rules])
        )

    @staticmethod
    def value_hint(slot: Slot) -> str:
        """
        :param slot: A slot whose value is to be generated.
        :return: The expected JSON shape, listing every branch of a union.
        """
        return " or ".join(
            PromptBuilder.hint_for_type(candidate_type)
            for candidate_type in slot.decodable_types
        )

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

    @staticmethod
    def hint_for_type(candidate_type: type) -> str:
        """
        :param candidate_type: One type a value may take.
        :return: The expected JSON shape for it.
        """
        if issubclass(candidate_type, enum.Enum):
            return "one of: " + " | ".join(member.name for member in candidate_type)
        if candidate_type is bool:
            return "a JSON boolean"
        if candidate_type in (int, float):
            return "a JSON number"
        return "a JSON string"

    @staticmethod
    def verbalize_request(
        expression: Match, backend: Optional[QueryBackend] = None
    ) -> Optional[str]:
        """
        Render the whole query in English, constraints included.

        Unfilled leaves are rendered by krrood as the literal word "Ellipsis". The
        system prompt glosses that word as a blank rather than a value, so the request
        can be shown in full instead of stripped down to its conditions.

        :param expression: The match being resolved.
        :param backend: Decides the opening verb.
        :return: The English rendering.
        """
        return verbalize_expression(expression.expression, backend=backend)

    @staticmethod
    def build_repair(previous_prompt: str, raw_response: str, error: Exception) -> str:
        """
        :param previous_prompt: The prompt that produced the rejected response.
        :param raw_response: The rejected response.
        :param error: Why it was rejected.
        :return: A prompt asking for a corrected response.
        """
        return (
            previous_prompt
            + "\n\n## Your previous answer was rejected\n\n```\n"
            + raw_response[:2000]
            + "\n```\n\nReason: "
            + str(error)
            + "\n\nAnswer again, correcting this. JSON array only."
        )
