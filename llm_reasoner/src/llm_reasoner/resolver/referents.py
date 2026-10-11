"""
How the candidates of a referent slot are named to the model and how its answer names
one: by an integer token into a listed extension, or by a handle of the world injector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List, Mapping, Tuple

from llm_reasoner.resolver.context import CandidateRenderer
from llm_reasoner.resolver.exceptions import (
    AmbiguousReferent,
    NoFittingReferent,
    SlotValueRejected,
)
from llm_reasoner.resolver.reflection import Slot
from llm_reasoner.world.classes import identifier_of
from llm_reasoner.world.tools import WorldToolName
from llm_reasoner.world.view import WorldView

LISTED_CANDIDATES = 10
"""
The most candidates a slot lists by handle; a slot with more states how many it has and
leaves finding them to the world tools.
"""

CANDIDATE_INDENT = "      "
"""
The indentation of a candidate under its slot.
"""

SLOT_NOTE_INDENT = "    "
"""
The indentation of a note under its slot.
"""


class ReferentAnswerKey(StrEnum):
    """
    The keys of the answers a referent slot takes besides a single candidate.
    """

    NO_FITTING_CANDIDATE = "no_fitting_candidate"
    """
    Says that no candidate is what the request asks for, and why.
    """

    EQUALLY_FITTING = "equally_fitting"
    """
    Names the candidates that fit equally when nothing in the request prefers one.
    """

    WHY = "why"
    """
    Says why none of the equally fitting candidates is preferred.
    """


NO_FITTING_CANDIDATE_RULE = (
    "- When no candidate of a referent slot is what the request asks for, or none can do "
    "what the slot is for in this object, never substitute a near miss: answer the slot "
    f'with {{"{ReferentAnswerKey.NO_FITTING_CANDIDATE}": "<why none fits>"}}.'
)
"""
The rule letting the model say that no candidate fits.
"""

EQUALLY_FITTING_RULE = (
    "- When several candidates of a referent slot fit equally and nothing in the request "
    "prefers one, do not pick: answer the slot with "
    f'{{"{ReferentAnswerKey.EQUALLY_FITTING}": [<each of them>], '
    f'"{ReferentAnswerKey.WHY}": "<why none is preferred>"}}.'
)
"""
The rule letting the model say that several candidates fit equally.
"""


class ReferentNaming(ABC):
    """
    Names the candidates of referent slots in the prompt and reads the answer for one:
    a candidate, or the verdict that none fits or that several fit equally.
    """

    @property
    def rules(self) -> Tuple[str, ...]:
        """
        The rules telling the model how to answer a referent slot.
        """
        return (self.answer_rule, NO_FITTING_CANDIDATE_RULE, EQUALLY_FITTING_RULE)

    @property
    @abstractmethod
    def answer_rule(self) -> str:
        """
        The rule telling the model how to answer a referent slot.
        """

    @property
    @abstractmethod
    def candidate_heading(self) -> str:
        """
        The line introducing the candidates of a referent slot.
        """

    @abstractmethod
    def candidate_lines(
        self, slot: Slot, expression: Match, renderer: CandidateRenderer
    ) -> List[str]:
        """
        :param slot: A referent slot.
        :param expression: The match being resolved.
        :param renderer: Describes a candidate.
        :return: The lines presenting the slot's candidates.
        """

    def decode(self, slot: Slot, value: Any) -> Any:
        """
        :param slot: A referent slot.
        :param value: What the answer gives for it.
        :return: The candidate it names.
        :raises NoFittingReferent: When the answer says no candidate fits.
        :raises AmbiguousReferent: When the answer names candidates fitting equally.
        :raises SlotValueRejected: When it is neither a candidate nor a verdict.
        """
        if not isinstance(value, Mapping):
            return self.decode_candidate(slot, value)
        if isinstance(value.get(ReferentAnswerKey.NO_FITTING_CANDIDATE), str):
            raise NoFittingReferent(
                slot_path=slot.path,
                reason=value[ReferentAnswerKey.NO_FITTING_CANDIDATE],
            )
        named = value.get(ReferentAnswerKey.EQUALLY_FITTING)
        if not isinstance(named, list) or len(named) < 2:
            raise SlotValueRejected(
                slot_path=slot.path,
                value=value,
                reason="Expected a candidate, a verdict that none fits, or at least "
                "two equally fitting candidates.",
            )
        raise AmbiguousReferent(
            slot_path=slot.path,
            candidates=[self.decode_candidate(slot, candidate) for candidate in named],
            reason=str(value.get(ReferentAnswerKey.WHY, "")),
        )

    @abstractmethod
    def decode_candidate(self, slot: Slot, value: Any) -> Any:
        """
        :param slot: A referent slot.
        :param value: What the answer gives for one candidate.
        :return: The candidate it names.
        :raises SlotValueRejected: When it names none of the slot's candidates.
        """


@dataclass(frozen=True)
class IntegerTokens(ReferentNaming):
    """
    Lists every candidate under an integer token, described by the prompt's candidate
    renderer, and reads the token back.
    """

    @property
    def answer_rule(self) -> str:
        return (
            "- Referent slots take the integer token of one of the listed candidates."
        )

    @property
    def candidate_heading(self) -> str:
        return f"{SLOT_NOTE_INDENT}choose one candidate by its integer token:"

    def candidate_lines(
        self, slot: Slot, expression: Match, renderer: CandidateRenderer
    ) -> List[str]:
        return [
            self.hanging(
                self.candidate_prefix(token), renderer.render(candidate, expression)
            )
            for token, candidate in enumerate(slot.domain)
        ]

    @staticmethod
    def candidate_prefix(token: int) -> str:
        """
        :param token: The integer token a candidate is chosen by.
        :return: What precedes the description of that candidate.
        """
        return f"{CANDIDATE_INDENT}[{token}] "

    @staticmethod
    def hanging(prefix: str, text: str) -> str:
        """
        :param prefix: What precedes the first line.
        :param text: Text that may span several lines.
        :return: The text after the prefix, its later lines indented to align with
            the first.
        """
        first, *rest = text.splitlines() or [""]
        return "\n".join([prefix + first] + [" " * len(prefix) + line for line in rest])

    def decode_candidate(self, slot: Slot, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SlotValueRejected(
                slot_path=slot.path,
                value=value,
                reason="Expected an integer candidate token.",
            )
        if not 0 <= value < len(slot.domain):
            raise SlotValueRejected(
                slot_path=slot.path,
                value=value,
                reason=f"Token out of range; expected 0..{len(slot.domain) - 1}.",
            )
        return slot.domain[value]


@dataclass(frozen=True)
class WorldHandles(ReferentNaming):
    """
    Names candidates by the world injector's handles: a few are listed with what holds
    them, more are left for the model to find with the world tools.
    """

    view: WorldView
    """
    The world view giving out the handles the model answers with.
    """

    @property
    def answer_rule(self) -> str:
        return (
            "- Referent slots take the handle of one of the slot's candidates, such as "
            '"Body#3". Handles exist only once a tool has returned them, so never name '
            "one you have not been given. When the candidates are not listed, search "
            f"the slot's kind with {WorldToolName.FIND_INDIVIDUALS} first, and use "
            f"{WorldToolName.LIST_KINDS} only to explore what else the world has. An "
            "individual that gives a candidate its meaning, such as a Cup giving a body "
            "its meaning, holds the candidate in one of its fields: follow that field to "
            "the candidate's handle."
        )

    @property
    def candidate_heading(self) -> str:
        return f"{SLOT_NOTE_INDENT}answer with the handle of one candidate:"

    def candidate_lines(
        self, slot: Slot, expression: Match, renderer: CandidateRenderer
    ) -> List[str]:
        if len(slot.domain) > LISTED_CANDIDATES:
            return [self.unlisted_line(slot)]
        return [self.candidate_line(candidate) for candidate in slot.domain]

    def candidate_line(self, candidate: Any) -> str:
        """
        :return: The line naming a listed candidate by its handle, with its kind and the
            individuals holding it.
        """
        handle = self.view.handles.handle_of(candidate)
        holdings = [
            f"held as {holder.field_name} of a {holder.type_name}"
            for holder in self.view.holders(handle)
        ]
        return f"{CANDIDATE_INDENT}{handle} ({type(candidate).__name__})" + (
            " - " + ", ".join(holdings) if holdings else ""
        )

    @staticmethod
    def unlisted_line(slot: Slot) -> str:
        """
        :return: The line stating how many candidates a slot has, too many to list.
        """
        return (
            f"{SLOT_NOTE_INDENT}{len(slot.domain)} candidates of the kind "
            f"{identifier_of(slot.declared_type)}; find the one meant with the world "
            "tools and answer with its handle."
        )

    def decode_candidate(self, slot: Slot, value: Any) -> Any:
        if not isinstance(value, str) or not self.view.handles.is_live(value):
            raise SlotValueRejected(
                slot_path=slot.path,
                value=value,
                reason="Expected the handle of a live individual, as the world tools "
                "give them out.",
            )
        individual = self.view.handles.individual_of(value)
        if not any(individual is candidate for candidate in slot.domain):
            raise SlotValueRejected(
                slot_path=slot.path,
                value=value,
                reason=f"{value} is not one of the {len(slot.domain)} candidates of "
                f"the kind {slot.type_name}.",
            )
        return individual
