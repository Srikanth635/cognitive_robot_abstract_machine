from __future__ import annotations

import enum
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.factories import entity
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.quantifiers import An
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from typing_extensions import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.clients.tools import (
    ResponseTruncated,
    Tool,
    ToolConversation,
    ToolUsingClient,
)
from llm_reasoner.observation.events import (
    AttemptRejected,
    AttemptStarted,
    GroundSummary,
    JustificationSummary,
    KnowledgeConsulted,
    ResolutionFailed,
    ResolutionFinished,
    ResolutionPurpose,
    ResolutionStarted,
    SlotSummary,
    new_call,
)
from llm_reasoner.observation.sinks import EventSink, NullSink
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
    AnswerTruncated,
    ClientCannotUseTools,
    UnknownGround,
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
from llm_reasoner.resolver.grounds import (
    Ground,
    GroundKey,
    GroundSource,
    Justification,
)
from llm_reasoner.resolver.knowledge import KnowledgeConsultation, KnowledgeRequester
from llm_reasoner.resolver.outcomes import OutcomeCheck, OutcomePrediction
from llm_reasoner.resolver.prompt_format import PromptRule
from llm_reasoner.resolver.prompts import (
    OpenValueMeaning,
    PromptBuilder,
    RequestAnswerKey,
    SourceDetail,
)
from llm_reasoner.resolver.referents import IntegerTokens, ReferentNaming, WorldHandles
from llm_reasoner.resolver.reflection import Slot, SlotKind, reflect_slots
from llm_reasoner.world.access import WorldAccess, identifier_of

T = TypeVar("T")

JSON_ARRAY = re.compile(r"\[.*]", re.DOTALL)
"""
Locates the JSON array inside a response that may carry surrounding text.
"""

# %% decoded answers


@dataclass(frozen=True)
class DecodedElement:
    """
    One element of an answer, decoded and checked.
    """

    assignment: Dict[str, Any]
    """
    The values for the open slots, by access path.
    """

    why: str
    """
    The model's explanation of the element.
    """

    grounds: List[Ground]
    """
    What the element's choices rest on, each checked against its source.
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

    outcome_check: Optional[OutcomeCheck] = field(kw_only=True, default=None)
    """
    Tries the instances an answer constructs before they are accepted, refusing the
    answer with its outcome when they fall short; none tries nothing.
    """

    outcome_prediction: Optional[OutcomePrediction] = field(kw_only=True, default=None)
    """
    Offers the model tools predicting what an answer would bring about, beside the
    backend's own tools when it asks with tools; none offers nothing.
    """

    observer: EventSink = field(kw_only=True, default_factory=NullSink)
    """
    Where the resolution is reported to as it goes: each attempt at the model's answer,
    why one is rejected and how the resolution ends; nowhere by default.
    """

    purpose: ResolutionPurpose = field(kw_only=True, default=ResolutionPurpose.ACTION)
    """
    What the resolution is for, which tells the resolution of the action from the choice
    of the situation features of the knowledge describing it in what is reported.
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

    knowledge_tools: List[Tool] = field(default_factory=list, init=False)
    """
    The knowledge tools reading the hypothesis the latest consultation obtained; none
    before one, or when no knowledge describes the queried action.
    """

    frame_parts: Dict[str, List[str]] = field(default_factory=dict, init=False)
    """
    The names of the parts of the action's frame each field of the queried action fills,
    from the latest consultation; none when no frame was consulted.
    """

    justifications: List[Justification] = field(default_factory=list, init=False)
    """
    Why each instance of the latest resolution was chosen and what its choices rest on.
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
    def offers_knowledge(self) -> bool:
        """
        Whether the model is offered the knowledge tools: when the backend asks with
        tools and the latest consultation obtained knowledge.
        """
        return bool(self.tools) and bool(self.knowledge_tools)

    @property
    def offers_prediction(self) -> bool:
        """
        Whether the model is offered tools predicting what an answer would bring about:
        when the backend asks with tools and has a prediction.
        """
        return bool(self.tools) and self.outcome_prediction is not None

    @property
    def request_completion(self) -> Completion:
        """
        How the model is asked for the current request: with the knowledge tools and the
        prediction's tools beside the backend's own when they are offered.
        """
        further_tools = [
            *(self.knowledge_tools if self.offers_knowledge else []),
            *(self.outcome_prediction.tools() if self.offers_prediction else []),
        ]
        if not further_tools:
            return self.completion
        return ToolCompletion(client=self.client, tools=[*self.tools, *further_tools])

    @property
    def system_prompt(self) -> str:
        """
        The system prompt, its rules matching how the candidates of referent slots are
        named, how the model is asked and whether it is offered knowledge.
        """
        return self._prompt_builder().system_prompt(self._further_rules())

    @property
    def rules(self) -> List[PromptRule]:
        """
        The rules the system prompt states, matching how the candidates of referent
        slots are named, how the model is asked and whether it is offered knowledge.
        """
        return self._prompt_builder().rules(self._further_rules())

    @property
    def ground_sources(self) -> List[GroundSource]:
        """
        The sources an answer to the current request may name what it rests on in: the
        world when candidates are named by its handles, and the consulted knowledge when
        it is offered.
        """
        sources = list(self.referent_naming.ground_sources)
        if self.offers_knowledge:
            sources.extend(self.knowledge.grounds_in(self.consultation))
        return sources

    def _further_rules(self) -> List[PromptRule]:
        """
        :return: The rules of how the model is asked for the current request, of
            reading the knowledge when it is offered, and of naming what an answer rests
            on when there is a source to name it in.
        """
        rules = list(self.request_completion.answer_rules)
        if self.offers_knowledge:
            rules.extend(self.knowledge.rules_for(self.consultation))
        if self.offers_prediction:
            rules.append(self.outcome_prediction.rule)
        if self.ground_sources:
            rules.append(PromptBuilder.grounds_rule(self.ground_sources))
        return rules

    # %% evaluation

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        self.justifications = []
        slots = reflect_slots(expression)
        with self._reporting(expression, slots) as resolution:
            self._reject_unresolvable(slots)

            open_slots = [slot for slot in slots if slot.kind is not SlotKind.BOUND]
            if not open_slots:
                accepted = self._validate(expression, [expression.construct_instance()])
                self._finished(resolution, 0, accepted)
                yield from accepted
                return

            number_of_candidates = (
                expression.expression._limit_ or self.number_of_candidates
            )
            yield from self._ask(
                expression, slots, open_slots, number_of_candidates, resolution
            )

    @contextmanager
    def _reporting(self, expression: Match[T], slots: Sequence[Slot]) -> Iterator[str]:
        """
        Report a resolution as started, and as failed if it ends by raising, which then
        goes on.

        :param expression: The match being resolved.
        :param slots: Every reflected slot.
        :return: The identifier of the resolution, which its events repeat.
        """
        resolution = new_call()
        self.observer.emit(
            ResolutionStarted(
                resolution=resolution,
                purpose=self.purpose,
                action=identifier_of(expression.type),
                slots=[
                    SlotSummary(
                        path=slot.path,
                        kind=slot.kind.name,
                        type_name=slot.type_name,
                        candidates=(
                            len(slot.domain)
                            if slot.selects_from_domain and slot.domain is not None
                            else None
                        ),
                    )
                    for slot in slots
                ],
                max_repairs=self.max_repairs,
            )
        )
        try:
            yield resolution
        except Exception as error:
            self.observer.emit(
                ResolutionFailed(
                    resolution=resolution,
                    error_type=type(error).__name__,
                    reason=str(error),
                )
            )
            raise

    def _finished(self, resolution: str, attempts: int, accepted: Sequence[T]) -> None:
        """
        Report a resolution as finished, with why each instance was chosen.

        :param resolution: The identifier of the resolution.
        :param attempts: How many answers it took.
        :param accepted: The instances resolved.
        """
        self.observer.emit(
            ResolutionFinished(
                resolution=resolution,
                attempts=attempts,
                instances=[type(instance).__name__ for instance in accepted],
                justifications=[
                    JustificationSummary(
                        why=justification.why,
                        grounds=[
                            GroundSummary(
                                source=ground.source, identifier=ground.identifier
                            )
                            for ground in justification.grounds
                        ],
                    )
                    for justification in self.justifications
                ],
            )
        )

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
        resolution: str,
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
        :param resolution: The identifier of the resolution, which what is reported
            repeats.
        :return: The instances built from the answer that meet the where-conditions.
        """
        self._consult(expression, resolution)
        prompt = self._prompt_builder().build(
            expression=expression,
            slots=slots,
            context_sections=context_sections(self.context_provider, expression),
            number_of_candidates=number_of_candidates,
            backend=self,
            ground_sources=self.ground_sources,
        )

        completion = self.request_completion
        last_error: Optional[Exception] = None
        for attempt in range(self.max_repairs + 1):
            self.observer.emit(AttemptStarted(resolution=resolution, attempt=attempt))
            try:
                conversation = completion.complete(self.system_prompt, prompt)
            except ResponseTruncated as truncated:
                last_error = AnswerTruncated(
                    output_tokens=truncated.output_tokens,
                    token_limit=truncated.token_limit,
                )
                prompt = self._sent_back_unfinished(
                    resolution,
                    attempt,
                    prompt,
                    truncated.partial_answer,
                    last_error,
                    completion,
                )
                continue
            raw_response = conversation.answer
            self.conversations.append(conversation)
            self.log.append(
                Exchange(
                    system=self.system_prompt,
                    user=prompt,
                    raw_response=raw_response,
                    client_identity=completion.identity,
                    attempt=attempt,
                )
            )
            try:
                accepted = self._accept(expression, raw_response, open_slots)
            except RetryableModelFailure as error:
                last_error = error
                self.observer.emit(
                    AttemptRejected(
                        resolution=resolution,
                        attempt=attempt,
                        kind=type(error).__name__,
                        details=error.details(),
                        answer=raw_response,
                    )
                )
                prompt = PromptBuilder.build_repair(prompt, raw_response, error)
                continue
            self._finished(resolution, attempt + 1, accepted)
            return accepted

        raise RepairBudgetExhausted(
            attempts=self.max_repairs + 1, last_error=last_error
        )

    def _sent_back_unfinished(
        self,
        resolution: str,
        attempt: int,
        prompt: str,
        partial_answer: str,
        error: AnswerTruncated,
        completion: Completion,
    ) -> str:
        """
        Report an answer the model was cut off in as rejected and keep it in the log.

        :param prompt: The prompt that was asked.
        :param partial_answer: What the model had written when it was cut off.
        :param error: That it was cut off, and where.
        :return: The prompt asking again, showing the model the answer it did not
            finish.
        """
        self.log.append(
            Exchange(
                system=self.system_prompt,
                user=prompt,
                raw_response=partial_answer,
                client_identity=completion.identity,
                attempt=attempt,
            )
        )
        self.observer.emit(
            AttemptRejected(
                resolution=resolution,
                attempt=attempt,
                kind=type(error).__name__,
                details=error.details(),
                answer=partial_answer,
            )
        )
        return PromptBuilder.build_repair(prompt, partial_answer, error)

    def _consult(self, expression: Match[T], resolution: str) -> None:
        """
        Consult the knowledge component about the queried action, when there is a
        knowledge requester, and keep the tools reading what it obtained.

        :param expression: The match being resolved.
        :param resolution: The identifier of the resolution, which the consultation is
            reported with.
        """
        if self.knowledge is None:
            return
        self.situation_backend = self.situation_backend_for(expression)
        self.consultation = self.knowledge.consult(expression, self.situation_backend)
        self.knowledge_tools = self.knowledge.tools_for(self.consultation)
        self.frame_parts = self.knowledge.frame_parts_for(self.consultation)
        self.observer.emit(
            KnowledgeConsulted(
                resolution=resolution,
                designator=self.consultation.designator,
                situation={
                    name: feature.value
                    for name, feature in self.consultation.situation.items()
                },
                record_id=self.consultation.record_id,
                perspective_record_ids=list(self.consultation.perspective_record_ids),
            )
        )

    def situation_backend_for(self, action: Match) -> LanguageModelBackend:
        """
        :param action: The query for an action.
        :return: A backend choosing the situation features of the knowledge describing
            the action: this backend's model, tools and context with the action stated,
            one structure described by its docstrings, its values facts rather than
            choices, and neither a consultation, an outcome check nor a prediction of
            its own.
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
            outcome_check=None,
            outcome_prediction=None,
            purpose=ResolutionPurpose.SITUATION,
        )

    def _accept(
        self, expression: Match[T], raw_response: str, open_slots: Sequence[Slot]
    ) -> List[T]:
        """
        :param expression: The match being resolved.
        :param raw_response: The response text.
        :param open_slots: The slots the model was asked to fill.
        :return: The instances built from the response that meet the where-conditions,
            each kept with its justification.
        :raises NoValidCandidates: When none of them does.
        :raises OutcomeFellShort: When trying them falls short.
        """
        decoded = self._decode(raw_response, open_slots)
        constructed = [
            self._construct(expression, element.assignment) for element in decoded
        ]
        accepted = self._validate(expression, constructed)
        if self.outcome_check is not None:
            self.outcome_check.check(accepted)
        self.justifications = [
            Justification(instance=instance, why=element.why, grounds=element.grounds)
            for instance, element in zip(constructed, decoded)
            if any(instance is kept for kept in accepted)
        ]
        return accepted

    def _prompt_builder(self) -> PromptBuilder:
        """:return: A builder configured from this backend's seams."""
        return PromptBuilder(
            renderer=self.candidate_renderer,
            referent_naming=self.referent_naming,
            source_detail=self.source_detail,
            open_values=self.open_values,
            frame_parts=self.frame_parts if self.offers_knowledge else {},
        )

    # %% decoding

    def _decode(
        self, raw_response: str, open_slots: Sequence[Slot]
    ) -> List[DecodedElement]:
        """
        Turn a response into decoded elements, checked against the reflected schema and
        against the sources of their grounds.

        :param raw_response: The response text.
        :param open_slots: The slots the model was asked to fill.
        :return: One decoded element per candidate.
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
    ) -> DecodedElement:
        """
        :param element: One element of the response array.
        :param slots_by_path: The slots the model was asked to fill.
        :param raw_response: The response text, for error reporting.
        :return: The decoded element.
        """
        if not isinstance(element, dict):
            raise ModelResponseUnparseable(
                raw_response=raw_response, reason="Array elements must be objects."
            )
        if RequestAnswerKey.UNSATISFIABLE in element:
            raise UnsatisfiableRequest(reason=element[RequestAnswerKey.UNSATISFIABLE])
        for key in element:
            if key not in slots_by_path and key not in (
                RequestAnswerKey.WHY,
                RequestAnswerKey.GROUNDS,
            ):
                raise UnknownSlotInResponse(
                    slot_path=key, known_paths=list(slots_by_path)
                )
        for path in slots_by_path:
            if path not in element:
                raise MissingSlotInResponse(slot_path=path)
        return DecodedElement(
            assignment={
                path: self._decode_value(slot, element[path])
                for path, slot in slots_by_path.items()
            },
            why=element.get(RequestAnswerKey.WHY, ""),
            grounds=[
                self._decode_ground(stated, raw_response)
                for stated in element.get(RequestAnswerKey.GROUNDS, [])
            ],
        )

    def _decode_ground(self, stated: Any, raw_response: str) -> Ground:
        """
        :param stated: One entry of an element's grounds.
        :param raw_response: The response text, for error reporting.
        :return: What the entry names, identified by its source.
        :raises UnknownGround: When no source offered has what it names.
        """
        if not isinstance(stated, dict) or len(stated) != 1:
            raise ModelResponseUnparseable(
                raw_response=raw_response,
                reason="Each ground must be an object with exactly one key, its "
                "source.",
            )
        ((key, name),) = stated.items()
        sources = {source.key: source for source in self.ground_sources}
        identifier = (
            sources[key].identify(name)
            if key in sources and isinstance(name, str)
            else None
        )
        if identifier is None:
            raise UnknownGround(source=str(key), name=str(name))
        return Ground(source=GroundKey(key), identifier=identifier)

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
            ground_sources=self.ground_sources,
        )
