from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field, replace

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.factories import entity
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.quantifiers import An
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from typing_extensions import Any, Dict, Iterable, List, Optional, Sequence, TypeVar

from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.clients.tools import Tool, ToolConversation, ToolUsingClient
from llm_reasoner.resolver.action_context import ActionContextProvider
from llm_reasoner.resolver.completion import Completion, PlainCompletion, ToolCompletion
from llm_reasoner.resolver.context import (
    CandidateRenderer,
    CompositeContextProvider,
    PromptContextProvider,
    SchemaOnlyContextProvider,
    StructuralCandidateRenderer,
    context_sections,
)
from llm_reasoner.resolver.exceptions import (
    ClientCannotUseTools,
    MissingSlotInResponse,
    ModelResponseUnparseable,
    NoCandidateReferents,
    NoValidCandidates,
    ReferentSlotHasNoDomain,
    RepairBudgetExhausted,
    RetryableModelFailure,
    SlotValueRejected,
    UnknownSlotInResponse,
    UnsatisfiableRequest,
)
from llm_reasoner.resolver.knowledge import KnowledgeConsultation, KnowledgeRequester
from llm_reasoner.resolver.prompt_format import PromptRule
from llm_reasoner.resolver.prompts import (
    OpenValueMeaning,
    PromptBuilder,
    RequestAnswerKey,
    SourceDetail,
)
from llm_reasoner.resolver.referents import IntegerTokens, ReferentNaming, WorldHandles
from llm_reasoner.resolver.reflection import Slot, SlotKind, reflect_slots
from llm_reasoner.world.access import WorldAccess

T = TypeVar("T")

JSON_ARRAY = re.compile(r"\[.*]", re.DOTALL)
"""
Locates the JSON array inside a response that may carry surrounding text.
"""

# %% backend


@dataclass
class LanguageModelBackend(GenerativeBackend):
    """
    Resolves an underspecified match by asking a language model.

    Values are generated and individuals are selected from an extension, never invented.
    Because a model is an unsound generator, the match's where-conditions are stated in
    the prompt and also applied afterwards as a filter.
    """

    client: LanguageModelClient
    """
    The model seam.
    """

    context_provider: PromptContextProvider = field(
        default_factory=SchemaOnlyContextProvider
    )
    """
    The situational-knowledge seam.
    """

    candidate_renderer: CandidateRenderer = field(
        default_factory=StructuralCandidateRenderer
    )
    """
    Describes the candidates of referent slots.
    """

    number_of_candidates: int = field(kw_only=True, default=3)
    """
    How many alternatives to request when the query sets no limit.
    """

    max_repairs: int = field(kw_only=True, default=1)
    """
    How often an unusable response is fed back before giving up.
    """

    source_detail: SourceDetail = field(kw_only=True, default=SourceDetail.FULL)
    """
    How much of the target type's definition is put in the prompt: by default its whole
    source, which shows what the type does with each value its docstrings may leave
    unsaid.
    """

    open_values: OpenValueMeaning = field(kw_only=True, default=OpenValueMeaning.CHOICE)
    """
    What an answer to an open value slot stands for: by default a value chosen to serve
    the request.
    """

    tools: Sequence[Tool] = field(kw_only=True, default=())
    """
    The tools the model may call before it answers; none asks it in one completion.
    """

    referent_naming: ReferentNaming = field(kw_only=True, default_factory=IntegerTokens)
    """
    How the candidates of referent slots are named and answered.
    """

    knowledge: Optional[KnowledgeRequester] = field(kw_only=True, default=None)
    """
    Consulted about the queried action before the model is asked; no consultation when
    absent.
    """

    completion: Completion = field(init=False)
    """
    How the model is asked, with the tools when there are any.
    """

    log: List[Exchange] = field(default_factory=list, init=False)
    """
    Every exchange, in order.
    """

    conversations: List[ToolConversation] = field(default_factory=list, init=False)
    """
    The conversation of every exchange, in the same order, with the tool calls made in
    it and the tokens it consumed.
    """

    consultation: Optional[KnowledgeConsultation] = field(default=None, init=False)
    """
    The latest consultation of the knowledge component, ``None`` before the model was
    first asked or without a knowledge requester.
    """

    situation_backend: Optional[LanguageModelBackend] = field(default=None, init=False)
    """
    The backend that chose the situation features in the latest consultation, with its
    own log and conversations.
    """

    def __post_init__(self) -> None:
        if not self.tools:
            self.completion = PlainCompletion(client=self.client)
            return
        if not isinstance(self.client, ToolUsingClient):
            raise ClientCannotUseTools(client_identity=self.client.identity)
        self.completion = ToolCompletion(client=self.client, tools=self.tools)

    @classmethod
    def with_world_tools(
        cls, client: LanguageModelClient, world: WorldAccess, **settings: Any
    ) -> LanguageModelBackend:
        """
        :param client: A client that lets the model call tools.
        :param world: The world the model asks about.
        :param settings: Further settings of the backend.
        :return: A backend whose model finds the candidates of referent slots with the
            world tools and answers with their handles.
        """
        return cls(
            client=client,
            tools=world.tools(),
            referent_naming=WorldHandles(world=world),
            **settings,
        )

    @property
    def system_prompt(self) -> str:
        """
        The system prompt, its rules matching how the candidates of referent slots are
        named and how the model is asked.
        """
        return self._prompt_builder().system_prompt(self.completion.answer_rules)

    @property
    def rules(self) -> List[PromptRule]:
        """
        The rules the system prompt states, matching how the candidates of referent
        slots are named and how the model is asked.
        """
        return self._prompt_builder().rules(self.completion.answer_rules)

    # %% evaluation

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        slots = reflect_slots(expression)
        self._reject_unresolvable(slots)

        open_slots = [slot for slot in slots if slot.kind is not SlotKind.BOUND]
        if not open_slots:
            yield from self._validate(expression, [expression.construct_instance()])
            return

        number_of_candidates = (
            expression.expression._limit_ or self.number_of_candidates
        )
        yield from self._ask(expression, slots, open_slots, number_of_candidates)

    @staticmethod
    def _reject_unresolvable(slots: Sequence[Slot]) -> None:
        """
        Refuse slots this backend cannot honestly fill.

        :param slots: Every reflected slot.
        """
        for slot in slots:
            if not slot.selects_from_domain:
                continue
            if slot.domain is None:
                raise ReferentSlotHasNoDomain(
                    slot_path=slot.path,
                    declared_type=slot.declared_type,
                    hint=(
                        "A language model can generate values but cannot mint "
                        f"individuals. Write this slot as variable({slot.type_name}) "
                        "to take the extension from the symbol graph, or "
                        f"variable({slot.type_name}, domain) to supply it explicitly."
                    ),
                )
            if not slot.domain:
                raise NoCandidateReferents(
                    slot_path=slot.path, declared_type=slot.declared_type
                )

    # %% asking

    def _ask(
        self,
        expression: Match[T],
        slots: Sequence[Slot],
        open_slots: Sequence[Slot],
        number_of_candidates: int,
    ) -> List[T]:
        """
        Request every slot in a single completion, repairing on rejection, including an
        answer whose instances break the match's where-conditions.

        Slots are asked about jointly because they are correlated, and per-slot
        questions would discard that structure.

        :param expression: The match being resolved.
        :param slots: Every reflected slot.
        :param open_slots: The slots the model must answer for.
        :param number_of_candidates: How many alternatives to request.
        :return: The instances built from the answer that meet the where-conditions.
        """
        self._consult(expression)
        prompt = self._prompt_builder().build(
            expression=expression,
            slots=slots,
            context_sections=context_sections(self.context_provider, expression),
            number_of_candidates=number_of_candidates,
            backend=self,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_repairs + 1):
            conversation = self.completion.complete(self.system_prompt, prompt)
            raw_response = conversation.answer
            self.conversations.append(conversation)
            self.log.append(
                Exchange(
                    system=self.system_prompt,
                    user=prompt,
                    raw_response=raw_response,
                    client_identity=self.completion.identity,
                    attempt=attempt,
                )
            )
            try:
                return self._accept(expression, raw_response, open_slots)
            except RetryableModelFailure as error:
                last_error = error
                prompt = PromptBuilder.build_repair(prompt, raw_response, error)

        raise RepairBudgetExhausted(
            attempts=self.max_repairs + 1, last_error=last_error
        )

    def _consult(self, expression: Match[T]) -> None:
        """
        Consult the knowledge component about the queried action, when there is a
        knowledge requester.

        :param expression: The match being resolved.
        """
        if self.knowledge is None:
            return
        self.situation_backend = self.situation_backend_for(expression)
        self.consultation = self.knowledge.consult(expression, self.situation_backend)

    def situation_backend_for(self, action: Match) -> LanguageModelBackend:
        """
        :param action: The query for an action.
        :return: A backend choosing the situation features of the knowledge describing
            the action: this backend's model, tools and context with the action stated,
            one structure described by its docstrings, its values facts rather than
            choices, and no consultation of its own.
        """
        return replace(
            self,
            context_provider=CompositeContextProvider(
                providers=[
                    ActionContextProvider(
                        action=action, builder=self._prompt_builder()
                    ),
                    self.context_provider,
                ]
            ),
            source_detail=SourceDetail.SIGNATURE,
            number_of_candidates=1,
            open_values=OpenValueMeaning.FACT,
            knowledge=None,
        )

    def _accept(
        self, expression: Match[T], raw_response: str, open_slots: Sequence[Slot]
    ) -> List[T]:
        """
        :param expression: The match being resolved.
        :param raw_response: The response text.
        :param open_slots: The slots the model was asked to fill.
        :return: The instances built from the response that meet the where-conditions.
        :raises NoValidCandidates: When none of them does.
        """
        constructed = [
            self._construct(expression, assignment)
            for assignment in self._decode(raw_response, open_slots)
        ]
        return self._validate(expression, constructed)

    def _prompt_builder(self) -> PromptBuilder:
        """:return: A builder configured from this backend's seams."""
        return PromptBuilder(
            renderer=self.candidate_renderer,
            referent_naming=self.referent_naming,
            source_detail=self.source_detail,
            open_values=self.open_values,
        )

    # %% decoding

    def _decode(
        self, raw_response: str, open_slots: Sequence[Slot]
    ) -> List[Dict[str, Any]]:
        """
        Turn a response into assignments, checked against the reflected schema.

        :param raw_response: The response text.
        :param open_slots: The slots the model was asked to fill.
        :return: One assignment per candidate.
        """
        payload = self._parse_array(raw_response)
        slots_by_path = {slot.path: slot for slot in open_slots}
        return [
            self._decode_element(element, slots_by_path, raw_response)
            for element in payload
        ]

    @staticmethod
    def _parse_array(raw_response: str) -> List[Any]:
        """
        :param raw_response: The response text.
        :return: The JSON array it contains.
        """
        located = JSON_ARRAY.search(raw_response)
        if located is None:
            raise ModelResponseUnparseable(
                raw_response=raw_response, reason="No JSON array found."
            )
        try:
            payload = json.loads(located.group(0))
        except json.JSONDecodeError as error:
            raise ModelResponseUnparseable(
                raw_response=raw_response, reason=f"Invalid JSON: {error}"
            ) from error
        if not isinstance(payload, list) or not payload:
            raise ModelResponseUnparseable(
                raw_response=raw_response, reason="Expected a non-empty JSON array."
            )
        return payload

    def _decode_element(
        self,
        element: Any,
        slots_by_path: Dict[str, Slot],
        raw_response: str,
    ) -> Dict[str, Any]:
        """
        :param element: One element of the response array.
        :param slots_by_path: The slots the model was asked to fill.
        :param raw_response: The response text, for error reporting.
        :return: The decoded assignment.
        """
        if not isinstance(element, dict):
            raise ModelResponseUnparseable(
                raw_response=raw_response, reason="Array elements must be objects."
            )
        if RequestAnswerKey.UNSATISFIABLE in element:
            raise UnsatisfiableRequest(reason=element[RequestAnswerKey.UNSATISFIABLE])
        for key in element:
            if key not in slots_by_path and key != RequestAnswerKey.WHY:
                raise UnknownSlotInResponse(
                    slot_path=key, known_paths=list(slots_by_path)
                )
        for path in slots_by_path:
            if path not in element:
                raise MissingSlotInResponse(slot_path=path)
        return {
            path: self._decode_value(slot, element[path])
            for path, slot in slots_by_path.items()
        }

    def _decode_value(self, slot: Slot, value: Any) -> Any:
        """
        Decode one JSON value into its slot's declared type.

        Nothing is accepted for a slot that may be left empty. A referent is decoded by
        the naming its candidates were offered under, which recovers the live individual
        the answer names; a kind by its integer token. A slot declaring a union is
        decoded against its branches in declaration order, so the model may choose the
        branch that fits the situation.

        :param slot: The slot the value was offered for.
        :param value: The offered JSON value.
        :return: The decoded value.
        """
        if value is None and slot.may_be_empty:
            return None
        if slot.kind is SlotKind.SORTAL:
            return IntegerTokens().decode(slot, value)
        if slot.selects_from_domain:
            return self.referent_naming.decode(slot, value)
        for candidate_type in slot.decodable_types:
            if LanguageModelBackend._json_fits(value, candidate_type):
                return LanguageModelBackend._coerce(value, candidate_type)
        raise SlotValueRejected(
            slot_path=slot.path,
            value=value,
            reason=f"Expected {PromptBuilder.value_hint(slot)}.",
        )

    @staticmethod
    def _json_fits(value: Any, candidate_type: type) -> bool:
        """
        :param value: An offered JSON value.
        :param candidate_type: One type the slot accepts.
        :return: Whether the value can become a value of that type.
        """
        if issubclass(candidate_type, enum.Enum):
            if isinstance(value, candidate_type):
                return True
            names = [member.name for member in candidate_type]
            values = [member.value for member in candidate_type]
            return value in names or value in values
        if candidate_type is bool:
            return isinstance(value, bool)
        if candidate_type in (int, float):
            return not isinstance(value, bool) and isinstance(value, (int, float))
        if candidate_type is str:
            return isinstance(value, str)
        return False

    @staticmethod
    def _coerce(value: Any, candidate_type: type) -> Any:
        """
        :param value: An offered JSON value known to fit.
        :param candidate_type: The type to produce.
        :return: The value as an instance of that type.
        """
        if issubclass(candidate_type, enum.Enum):
            if isinstance(value, candidate_type):
                return value
            by_name = {member.name: member for member in candidate_type}
            if value in by_name:
                return by_name[value]
            return {member.value: member for member in candidate_type}[value]
        if candidate_type is bool:
            return value
        return candidate_type(value)

    # %% construction and validation

    @staticmethod
    def _construct(expression: Match[T], assignment: Dict[str, Any]) -> T:
        """
        Write an assignment back into the match and build the instance.

        The instance is built by the factory the caller supplied, so this package never
        imports the types it constructs.

        :param expression: The match being resolved.
        :param assignment: Values keyed by access path.
        :return: The constructed instance.
        """
        for path, value in assignment.items():
            expression._get_mapped_variable_by_name(path)._value_ = value
        expression._update_kwargs_from_literal_values()
        return expression.construct_instance()

    @staticmethod
    def _validate(expression: Match[T], instances: List[T]) -> List[T]:
        """
        Keep only the instances satisfying the match's where-conditions.

        .. note::
            Conditions must be bound through ``match.variable``. Binding them through
            ``match.expression`` yields a condition that matches nothing.

        :param expression: The match being resolved.
        :param instances: The constructed candidates.
        :return: The candidates that satisfy the conditions, quantified as the match
            asks.
        :raises NoValidCandidates: When none of them does, naming the conditions none
            of them meets.
        """
        conditions = expression._where_conditions_
        meeting = LanguageModelBackend._meeting(expression, instances, conditions)
        if not meeting:
            raise NoValidCandidates(
                constructed=len(instances),
                broken_conditions=[
                    verbalize_expression(condition)
                    for condition in conditions
                    if not LanguageModelBackend._meeting(
                        expression, instances, [condition]
                    )
                ],
            )
        expression.variable._update_domain_(meeting)
        return list(
            entity(expression.variable)
            ._quantify_(expression._quantifier_type_)
            ._evaluate_natively_()
        )

    @staticmethod
    def _meeting(
        expression: Match[T], instances: List[T], conditions: Sequence[Any]
    ) -> List[T]:
        """
        :param expression: The match being resolved.
        :param instances: The constructed candidates.
        :param conditions: Conditions bound through the match's variable.
        :return: The candidates meeting every one of the conditions.
        """
        expression.variable._update_domain_(instances)
        query = entity(expression.variable)._quantify_(An)
        if conditions:
            query = query.where(*conditions)
        return list(query._evaluate_natively_())

    # %% introspection

    def describe(self, expression: Match[T]) -> str:
        """
        Assemble the prompt without calling the model.

        :param expression: The match to describe.
        :return: The user prompt that would be sent.
        """
        return self._prompt_builder().build(
            expression=expression,
            slots=reflect_slots(expression),
            context_sections=context_sections(self.context_provider, expression),
            number_of_candidates=expression.expression._limit_
            or self.number_of_candidates,
            backend=self,
        )
