"""
How the candidates of a referent slot are named to the model and how its answer names
one: by an integer token into a listed extension, or by a handle of the world injector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, Dict, List, Mapping, Tuple

from llm_reasoner.resolver.context import CandidateRenderer
from llm_reasoner.resolver.exceptions import (
    AmbiguousReferent,
    NoFittingReferent,
    SlotValueRejected,
)
from llm_reasoner.resolver.prompt_format import (
    AcceptedType,
    AcceptKey,
    CandidateKey,
    CandidateReference,
    PromptRule,
    RuleName,
    SlotKey,
)
from llm_reasoner.resolver.reflection import Slot
from llm_reasoner.world.classes import identifier_of
from llm_reasoner.world.naming import HANDLE_SEPARATOR
from llm_reasoner.world.tools import WorldToolName
from llm_reasoner.world.view import WorldView

LISTED_CANDIDATES = 10
"""
The most candidates a slot lists by handle; a slot with more states how many it has and
leaves finding them to the world tools.
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


NO_FITTING_CANDIDATE_RULE = PromptRule(
    name=RuleName.NO_FITTING_CANDIDATE,
    text=f'when no candidate of a slot with "{SlotKey.CANDIDATES}" is what the request '
    "asks for, or none can do what the slot is for in this object, answer the slot with "
    f'{{"{ReferentAnswerKey.NO_FITTING_CANDIDATE}": "<why none fits>"}} instead of a '
    "near miss.",
)
"""
The rule letting the model say that no candidate fits.
"""

EQUALLY_FITTING_RULE = PromptRule(
    name=RuleName.EQUALLY_FITTING,
    text=f'when several "{SlotKey.CANDIDATES}" of a slot fit equally and nothing in the '
    "request prefers one, answer the slot with "
    f'{{"{ReferentAnswerKey.EQUALLY_FITTING}": [<each of them, as "{SlotKey.ACCEPTS}" '
    f'names one>], "{ReferentAnswerKey.WHY}": "<why none is preferred>"}} instead of '
    "picking.",
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
    def rules(self) -> Tuple[PromptRule, ...]:
        """
        The rules telling the model how to answer a referent slot.
        """
        return (
            self.answer_rule,
            *self.finding_rules,
            NO_FITTING_CANDIDATE_RULE,
            EQUALLY_FITTING_RULE,
        )

    @property
    @abstractmethod
    def answer_rule(self) -> PromptRule:
        """
        The rule telling the model how to answer a referent slot.
        """

    @property
    def finding_rules(self) -> Tuple[PromptRule, ...]:
        """
        The rules telling the model how to find candidates the prompt does not list.
        """
        return ()

    @property
    @abstractmethod
    def accepted(self) -> Dict[str, Any]:
        """
        What a referent slot is answered with: the JSON type of the value and what it
        names.
        """

    @abstractmethod
    def candidates(
        self, slot: Slot, expression: Match, renderer: CandidateRenderer
    ) -> Dict[str, Any]:
        """
        :param slot: A referent slot.
        :param expression: The match being resolved.
        :param renderer: Describes a candidate.
        :return: The slot's candidates, as data.
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
    def answer_rule(self) -> PromptRule:
        return PromptRule(
            name=RuleName.REFERENT_SLOT,
            text=f'a slot whose "{SlotKey.ACCEPTS}" {AcceptKey.HOLDS} '
            f'"{CandidateReference.TOKEN}" takes the "{CandidateKey.TOKEN}" of one of its '
            f'"{CandidateKey.LISTED}" candidates, as a JSON {AcceptedType.NUMBER}.',
        )

    @property
    def accepted(self) -> Dict[str, Any]:
        return {
            AcceptKey.TYPE: AcceptedType.NUMBER,
            AcceptKey.HOLDS: CandidateReference.TOKEN,
        }

    def candidates(
        self, slot: Slot, expression: Match, renderer: CandidateRenderer
    ) -> Dict[str, Any]:
        return {
            CandidateKey.COUNT: len(slot.domain),
            CandidateKey.LISTED: [
                {
                    CandidateKey.TOKEN: token,
                    CandidateKey.DESCRIPTION: renderer.render(
                        candidate, expression
                    ).splitlines(),
                }
                for token, candidate in enumerate(slot.domain)
            ],
        }

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
    def answer_rule(self) -> PromptRule:
        return PromptRule(
            name=RuleName.REFERENT_SLOT,
            text=f'a slot whose "{SlotKey.ACCEPTS}" {AcceptKey.HOLDS} '
            f'"{CandidateReference.HANDLE}" takes the handle of one of its candidates as a '
            f'JSON {AcceptedType.STRING}, written "<Kind>{HANDLE_SEPARATOR}<n>"; a handle '
            "exists only once a tool has returned it, so never write one you were not "
            "given.",
        )

    @property
    def finding_rules(self) -> Tuple[PromptRule, ...]:
        return (
            PromptRule(
                name=RuleName.FINDING_CANDIDATES,
                text=f'when a slot\'s candidates are counted but not "{CandidateKey.LISTED}", '
                f"search their kind with {WorldToolName.FIND_INDIVIDUALS} first; use "
                f"{WorldToolName.LIST_KINDS} only to explore what else the world has.",
            ),
            PromptRule(
                name=RuleName.MEANING_HOLDER,
                text="an individual giving a candidate its meaning holds the candidate in "
                "one of its fields; follow that field to the candidate's handle.",
            ),
        )

    @property
    def accepted(self) -> Dict[str, Any]:
        return {
            AcceptKey.TYPE: AcceptedType.STRING,
            AcceptKey.HOLDS: CandidateReference.HANDLE,
        }

    def candidates(
        self, slot: Slot, expression: Match, renderer: CandidateRenderer
    ) -> Dict[str, Any]:
        counted = {
            CandidateKey.COUNT: len(slot.domain),
            CandidateKey.KIND: identifier_of(slot.declared_type),
        }
        if len(slot.domain) > LISTED_CANDIDATES:
            return counted
        return {
            **counted,
            CandidateKey.LISTED: [
                self.candidate_entry(candidate) for candidate in slot.domain
            ],
        }

    def candidate_entry(self, candidate: Any) -> Dict[str, Any]:
        """
        :return: A listed candidate: its handle, its kind and the fields of other
            individuals holding it.
        """
        handle = self.view.handles.handle_of(candidate)
        entry = {
            CandidateKey.HANDLE: handle,
            CandidateKey.KIND: type(candidate).__name__,
        }
        holdings = [
            f"{holder.field_name} of a {holder.type_name}"
            for holder in self.view.holders(handle)
        ]
        return {**entry, CandidateKey.HELD_AS: holdings} if holdings else entry

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
