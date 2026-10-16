"""
Tests of the resolver consulting the knowledge component about the action a query asks
for: the action stated to a resolution serving it, the situation features filled in a
first pass, and the hypothesis served for them.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from krrood.entity_query_language.factories import the, variable

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ScriptedToolRequest, ToolBox
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    ActionPerspectiveFile,
    ConceptReference,
    CRAM_AICOR_NAMESPACE,
    DescribedAction,
    DescribedActions,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
    KnowledgeToolName,
    SituationFeature,
    SituationFeatures,
    action_core,
    action_frame,
    pouring,
)
from llm_reasoner.resolver.action_context import ActionContextProvider
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.context import StructuralCandidateRenderer
from llm_reasoner.resolver.exceptions import UnknownGround
from llm_reasoner.resolver.grounds import Ground, GroundKey
from llm_reasoner.knowledge.ontologies.frame_fields import DimensionDescription
from llm_reasoner.observation.events import KnowledgeConsulted
from llm_reasoner.observation.sinks import MemorySink
from llm_reasoner.resolver.knowledge import (
    FRAME_KNOWLEDGE_RULE,
    KNOWLEDGE_RULE,
    KnowledgeRequester,
)
from llm_reasoner.resolver.prompt_format import PromptSection, SlotKey
from llm_reasoner.resolver.prompts import (
    ESTABLISHED_VALUES_RULE,
    INPUT_MEANINGS,
    OpenValueMeaning,
    PromptBuilder,
    RequestAnswerKey,
    SourceDetail,
)
from llm_reasoner.resolver.reflection import SlotKind, reflect_slots
from llm_reasoner.world.access import identifier_of
from llm_reasoner.world.view import WorldView

from .prompt_sections import bodies, text_outside_sections
from .test_generation import GenerationCase
from .world_mimics import ActionWithBodyAndValueSlots, WorldBody, body_with_geometry

CASES = Path(__file__).parent / "resolver_cases"
"""
The scripted model answers the tests replay.
"""

TILT_DESCRIBED_AS_POURING = DescribedAction(
    designator=identifier_of(ActionWithBodyAndValueSlots),
    concept=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring"),
    structure=pouring.PouringPhaseStructure,
)
"""
The tilting action, described by the knowledge about pouring.
"""

# %% helpers


def tilt_of(body) -> object:
    """
    :return: A query for tilting the given body, its angle left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=body, angle=...)


def builder() -> PromptBuilder:
    """
    :return: A builder describing individuals by their fields.
    """
    return PromptBuilder(renderer=StructuralCandidateRenderer())


def answer(name: str) -> str:
    return (CASES / name).read_text()


def situation_backend(*answers: str) -> LanguageModelBackend:
    """
    :return: A backend filling situation features with the scripted answers in turn.
    """
    return LanguageModelBackend(
        client=ScriptedClient(responses=[answer(name) for name in answers]),
        source_detail=SourceDetail.SIGNATURE,
        number_of_candidates=1,
    )


def requester_generating(
    store: HypothesisStore, *cases: GenerationCase
) -> KnowledgeRequester:
    """
    :return: A requester for the tilting action described as pouring, whose knowledge
        component generates the scripted hypotheses in turn, admitted without acceptance
        rules.
    """
    return KnowledgeRequester(
        service=KnowledgeService(
            backend=HypothesisBackend(
                client=ScriptedClient(responses=[case.text for case in cases]),
                suite=AcceptanceSuite(rules=[]),
                store=store,
            )
        ),
        described_actions=DescribedActions(entries=[TILT_DESCRIBED_AS_POURING]),
    )


def pouring_feature_taking(value) -> SituationFeature:
    """
    :return: The situation feature of a pouring structure that may take the value.
    """
    (feature,) = [
        feature
        for feature in SituationFeatures().of(pouring.PouringPhaseStructure)
        if value in feature.values
    ]
    return feature


# %% the action a resolution serves


class TestActionContext:
    def test_the_action_is_stated_as_its_own_request(self):
        action = tilt_of(body_with_geometry())
        (stated,) = bodies(
            ActionContextProvider(action=action, builder=builder()).context_for(action),
            PromptSection.REQUEST,
        )
        assert stated == PromptBuilder.verbalize_request(action)

    def test_the_action_states_the_values_its_query_fixes(self):
        body = body_with_geometry()
        action = tilt_of(body)
        (given,) = bodies(
            ActionContextProvider(action=action, builder=builder()).context_for(action),
            PromptSection.GIVEN,
        )
        assert json.loads(given) == [
            builder().given_entry(slot, action)
            for slot in reflect_slots(action)
            if slot.kind is SlotKind.BOUND
        ]

    def test_the_action_states_its_open_slots_as_its_own_resolution_writes_them(
        self,
    ):
        body = body_with_geometry()
        action = the(ActionWithBodyAndValueSlots)(
            source=variable(WorldBody, domain=[body]), angle=...
        )
        (slots,) = bodies(
            ActionContextProvider(action=action, builder=builder()).context_for(action),
            PromptSection.SLOTS,
        )
        assert json.loads(slots) == [
            builder().slot_entry(slot, action)
            for slot in reflect_slots(action)
            if slot.kind is not SlotKind.BOUND
        ]

    def test_an_action_whose_query_fixes_every_slot_states_no_slots(self):
        action = the(ActionWithBodyAndValueSlots)(
            source=body_with_geometry(), angle=1.0
        )
        assert (
            bodies(
                ActionContextProvider(action=action, builder=builder()).context_for(
                    action
                ),
                PromptSection.SLOTS,
            )
            == []
        )

    def test_the_action_is_a_section_of_the_prompt_of_the_resolution_it_serves(self):
        body = body_with_geometry()
        action = tilt_of(body)
        backend = LanguageModelBackend(
            client=ScriptedClient(),
            context_provider=ActionContextProvider(action=action, builder=builder()),
        )
        prompt = backend.describe(tilt_of(body))
        (section,) = bodies(prompt, PromptSection.ACTION)
        assert text_outside_sections(section) == ""
        assert text_outside_sections(prompt) == ""

    def test_the_system_prompt_says_what_the_action_section_means(self):
        backend = LanguageModelBackend(client=ScriptedClient())
        assert INPUT_MEANINGS[PromptSection.ACTION] in backend.system_prompt


# %% consulting the knowledge component


class TestKnowledgeRequester:
    def test_an_action_no_knowledge_describes_is_consulted_without_a_model(
        self, tmp_path
    ):
        action = tilt_of(body_with_geometry())
        requester = requester_generating(HypothesisStore(directory=tmp_path))
        requester.described_actions = DescribedActions(entries=[])
        situation = situation_backend()
        consultation = requester.consult(action, situation)
        assert consultation.designator == identifier_of(ActionWithBodyAndValueSlots)
        assert consultation.record_id is None
        assert situation.log == []
        assert requester.service.backend.client.calls == []

    def test_the_situation_is_what_the_backend_serving_the_action_chose(self, tmp_path):
        action = tilt_of(body_with_geometry())
        requester = requester_generating(
            HypothesisStore(directory=tmp_path), GenerationCase.GRANULAR_POURING
        )
        consultation = requester.consult(
            action, situation_backend("situation_granular.json")
        )
        granular = pouring.MaterialRegime.GRANULAR
        assert consultation.situation == {
            pouring_feature_taking(granular).field_name: granular
        }

    def test_the_hypothesis_for_the_situation_is_served_by_its_identifier(
        self, tmp_path
    ):
        store = HypothesisStore(directory=tmp_path)
        action = tilt_of(body_with_geometry())
        requester = requester_generating(store, GenerationCase.GRANULAR_POURING)
        consultation = requester.consult(
            action, situation_backend("situation_granular.json")
        )
        assert [record.record_id for record in store.records()] == [
            consultation.record_id
        ]

    def test_a_feature_left_unknown_is_not_fixed_in_the_request(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        action = tilt_of(body_with_geometry())
        requester = requester_generating(store, GenerationCase.GRANULAR_POURING)
        requester.consult(action, situation_backend("situation_granular.json"))
        (record,) = store.records()
        unknown = pouring_feature_taking(
            pouring.TransferTerminationReason.SOURCE_EXHAUSTED
        )
        assert unknown.field_name not in record.fixed_values

    def test_a_situation_met_before_is_served_without_generating(self, tmp_path):
        action = tilt_of(body_with_geometry())
        requester = requester_generating(
            HypothesisStore(directory=tmp_path), GenerationCase.GRANULAR_POURING
        )
        first = requester.consult(action, situation_backend("situation_granular.json"))
        second = requester.consult(action, situation_backend("situation_granular.json"))
        assert second.record_id == first.record_id
        assert len(requester.service.backend.client.calls) == 1

    def test_another_situation_is_generated_anew(self, tmp_path):
        action = tilt_of(body_with_geometry())
        requester = requester_generating(
            HypothesisStore(directory=tmp_path),
            GenerationCase.GRANULAR_POURING,
            GenerationCase.LIQUID_POURING,
        )
        granular = requester.consult(
            action, situation_backend("situation_granular.json")
        )
        liquid = requester.consult(action, situation_backend("situation_liquid.json"))
        assert liquid.record_id != granular.record_id
        assert len(requester.service.backend.client.calls) == 2


# %% consulting before every request to the model


def backend_consulting(store: HypothesisStore, *answers: str) -> LanguageModelBackend:
    """
    :return: A backend consulting the knowledge component about the tilting action
        described as pouring, whose model gives the scripted answers in turn.
    """
    return LanguageModelBackend(
        client=ScriptedClient(responses=[answer(name) for name in answers]),
        knowledge=requester_generating(store, GenerationCase.GRANULAR_POURING),
    )


class TestConsultingBackend:
    def test_the_knowledge_component_is_consulted_before_the_model_is_asked(
        self, tmp_path
    ):
        store = HypothesisStore(directory=tmp_path)
        backend = backend_consulting(
            store, "situation_granular.json", "angle_chosen.json"
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert [record.record_id for record in store.records()] == [
            backend.consultation.record_id
        ]

    def test_the_situation_is_chosen_by_a_backend_stating_the_action(self, tmp_path):
        body = body_with_geometry()
        backend = backend_consulting(
            HypothesisStore(directory=tmp_path),
            "situation_granular.json",
            "angle_chosen.json",
        )
        list(tilt_of(body).evaluate(backend=backend))
        (exchange,) = backend.situation_backend.log
        (action,) = bodies(exchange.user, PromptSection.ACTION)
        assert bodies(action, PromptSection.REQUEST) == [
            PromptBuilder.verbalize_request(tilt_of(body))
        ]

    def test_the_situation_is_one_structure_described_by_its_docstrings(self, tmp_path):
        backend = backend_consulting(
            HypothesisStore(directory=tmp_path),
            "situation_granular.json",
            "angle_chosen.json",
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert backend.situation_backend.number_of_candidates == 1
        assert backend.situation_backend.source_detail is SourceDetail.SIGNATURE

    def test_the_situation_is_answered_with_facts(self, tmp_path):
        backend = backend_consulting(
            HypothesisStore(directory=tmp_path),
            "situation_granular.json",
            "angle_chosen.json",
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert backend.situation_backend.open_values is OpenValueMeaning.FACT

    def test_choosing_the_situation_consults_nothing_itself(self, tmp_path):
        backend = backend_consulting(
            HypothesisStore(directory=tmp_path),
            "situation_granular.json",
            "angle_chosen.json",
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert backend.situation_backend.knowledge is None

    def test_the_request_to_the_model_follows_the_consultation(self, tmp_path):
        backend = backend_consulting(
            HypothesisStore(directory=tmp_path),
            "situation_granular.json",
            "angle_chosen.json",
        )
        (action,) = tilt_of(body_with_geometry()).evaluate(backend=backend)
        assert (
            action.angle
            == json.loads(answer("angle_chosen.json"))[0][
                "ActionWithBodyAndValueSlots.angle"
            ]
        )
        assert len(backend.log) == 1

    def test_a_query_leaving_nothing_open_consults_nothing(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        backend = backend_consulting(store)
        query = the(ActionWithBodyAndValueSlots)(source=body_with_geometry(), angle=0.8)
        list(query.evaluate(backend=backend))
        assert backend.consultation is None
        assert store.records() == []

    def test_a_backend_without_a_requester_consults_nothing(self):
        backend = LanguageModelBackend(
            client=ScriptedClient(responses=[answer("angle_chosen.json")])
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert backend.consultation is None
        assert backend.situation_backend is None


# %% values that are facts rather than choices


class TestOpenValueMeaning:
    def test_a_backend_answering_with_facts_states_the_rule_for_them(self):
        backend = LanguageModelBackend(
            client=ScriptedClient(), open_values=OpenValueMeaning.FACT
        )
        assert ESTABLISHED_VALUES_RULE in backend.rules

    def test_a_backend_answers_with_choices_by_default(self):
        backend = LanguageModelBackend(client=ScriptedClient())
        assert backend.open_values is OpenValueMeaning.CHOICE
        assert ESTABLISHED_VALUES_RULE not in backend.rules


# %% reading the knowledge while resolving


def world_backend_consulting(
    store: HypothesisStore, *turns: object
) -> LanguageModelBackend:
    """
    :return: A backend asking with the world tools that consults the knowledge
        component about the tilting action described as pouring, its model taking the
        scripted turns in order: the situation's answer first, then the request's.
    """
    return LanguageModelBackend.with_world_tools(
        client=ScriptedClient(tool_turns=list(turns)),
        world=WorldView(),
        knowledge=requester_generating(store, GenerationCase.GRANULAR_POURING),
    )


class TestReadingKnowledge:
    def test_the_request_reads_the_consulted_hypothesis_with_the_knowledge_tools(
        self, tmp_path
    ):
        body = body_with_geometry()
        backend = world_backend_consulting(
            HypothesisStore(directory=tmp_path),
            answer("situation_granular.json"),
            [ScriptedToolRequest(KnowledgeToolName.KNOWLEDGE_OVERVIEW, {})],
            answer("angle_chosen.json"),
        )
        list(tilt_of(body).evaluate(backend=backend))
        (call,) = backend.conversations[0].calls
        overview = json.loads(call.outcome.content)
        assert overview["situation"] == {
            name: value.name for name, value in backend.consultation.situation.items()
        }

    def test_the_rules_say_how_to_read_the_knowledge_when_it_is_offered(self, tmp_path):
        backend = world_backend_consulting(
            HypothesisStore(directory=tmp_path),
            answer("situation_granular.json"),
            answer("angle_chosen.json"),
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert KNOWLEDGE_RULE in backend.rules

    def test_a_backend_asking_without_tools_is_offered_no_knowledge(self, tmp_path):
        backend = backend_consulting(
            HypothesisStore(directory=tmp_path),
            "situation_granular.json",
            "angle_chosen.json",
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert KNOWLEDGE_RULE not in backend.rules

    def test_no_knowledge_is_offered_before_a_consultation(self):
        backend = LanguageModelBackend.with_world_tools(
            client=ScriptedClient(), world=WorldView()
        )
        assert KNOWLEDGE_RULE not in backend.rules


# %% what an answer rests on


def grounds_schema(backend: LanguageModelBackend) -> dict:
    """
    :return: The schema the latest request's output gives the grounds of an element.
    """
    (output,) = bodies(backend.log[-1].user, PromptSection.OUTPUT)
    return json.loads(output)["items"]["properties"][RequestAnswerKey.GROUNDS]


class TestGrounds:
    def test_an_answer_grounded_on_knowledge_is_justified_by_its_citation(
        self, tmp_path
    ):
        backend = world_backend_consulting(
            HypothesisStore(directory=tmp_path),
            answer("situation_granular.json"),
            answer("angle_grounded_on_knowledge.json"),
        )
        (action,) = tilt_of(body_with_geometry()).evaluate(backend=backend)
        view = backend.knowledge.service.view(backend.consultation.record_id)
        (justification,) = backend.justifications
        assert justification.instance is action
        assert justification.grounds == [
            Ground(
                source=GroundKey.KNOWLEDGE, identifier=view.citation_of(view.root).iri
            )
        ]

    def test_an_answer_grounded_on_a_world_individual_is_justified_by_its_handle(
        self,
    ):
        view = WorldView()
        body = body_with_geometry()
        handle = view.handle_of(body)
        stated = json.loads(answer("angle_grounded_on_the_body.json"))[0]
        backend = LanguageModelBackend.with_world_tools(
            client=ScriptedClient(
                tool_turns=[answer("angle_grounded_on_the_body.json")]
            ),
            world=view,
        )
        list(tilt_of(body).evaluate(backend=backend))
        (justification,) = backend.justifications
        assert stated[RequestAnswerKey.GROUNDS] == [{GroundKey.WORLD: handle}]
        assert justification.grounds == [
            Ground(source=GroundKey.WORLD, identifier=handle)
        ]
        assert justification.why == stated[RequestAnswerKey.WHY]

    def test_a_ground_naming_no_fragment_is_sent_back_for_repair(self, tmp_path):
        backend = world_backend_consulting(
            HypothesisStore(directory=tmp_path),
            answer("situation_granular.json"),
            answer("angle_grounded_on_no_fragment.json"),
            answer("angle_grounded_on_knowledge.json"),
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert len(backend.log) == 2
        assert f'kind="{UnknownGround.__name__}"' in backend.log[1].user

    def test_the_answer_may_rest_on_the_knowledge_when_it_is_offered(self, tmp_path):
        backend = world_backend_consulting(
            HypothesisStore(directory=tmp_path),
            answer("situation_granular.json"),
            answer("angle_chosen.json"),
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        Draft202012Validator(grounds_schema(backend)).validate(
            [{GroundKey.KNOWLEDGE: "pour"}, {GroundKey.WORLD: "WorldBody#1"}]
        )

    def test_an_answer_without_a_source_to_rest_on_is_asked_for_no_grounds(self):
        backend = LanguageModelBackend(
            client=ScriptedClient(responses=[answer("angle_chosen.json")])
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        (output,) = bodies(backend.log[0].user, PromptSection.OUTPUT)
        assert RequestAnswerKey.GROUNDS not in json.loads(output)["items"]["properties"]


# %% consulting the action core and its perspectives beside the phase structure

PERSPECTIVE_CASES = Path(__file__).parent / "action_perspective_cases"
"""
The scripted answers describing pouring in the action perspectives.
"""

TILT_FRAMED_AS_POURING = DescribedAction(
    designator=identifier_of(ActionWithBodyAndValueSlots),
    concept=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring"),
    structure=pouring.PouringPhaseStructure,
    frame=action_core.PouringFrame,
    frame_fields={
        "angle": DimensionDescription(
            quantity=action_core.QuantityKind.POSITION,
            about=action_core.SourceSlot,
            relative_to=action_core.GoalSlot,
        )
    },
)
"""
The tilting action, described by the knowledge about pouring and framed by the action
core's pouring frame.
"""


def requester_offering_the_frame(store: HypothesisStore) -> KnowledgeRequester:
    """
    :return: A requester for the tilting action framed as pouring that consults the
        action frame, whose knowledge component generates the granular phase structure,
        the core and the frame in turn.
    """
    return KnowledgeRequester(
        service=KnowledgeService(
            backend=HypothesisBackend(
                client=ScriptedClient(
                    responses=[
                        GenerationCase.GRANULAR_POURING.text,
                        (PERSPECTIVE_CASES / "pouring_core.json").read_text(),
                        (PERSPECTIVE_CASES / "pouring_frame.json").read_text(),
                    ]
                ),
                suite=AcceptanceSuite(rules=[]),
                store=store,
            )
        ),
        described_actions=DescribedActions(entries=[TILT_FRAMED_AS_POURING]),
        perspectives=(ActionPerspectiveFile.ACTION_FRAME,),
    )


def consulted_with_the_frame(store: HypothesisStore):
    """
    :return: The requester offering the frame, and its consultation about tilting.
    """
    requester = requester_offering_the_frame(store)
    consultation = requester.consult(
        tilt_of(body_with_geometry()), situation_backend("situation_granular.json")
    )
    return requester, consultation


class TestConsultingPerspectives:
    def test_by_default_neither_the_core_nor_a_perspective_is_consulted(self, tmp_path):
        requester = requester_generating(
            HypothesisStore(directory=tmp_path), GenerationCase.GRANULAR_POURING
        )
        requester.described_actions = DescribedActions(entries=[TILT_FRAMED_AS_POURING])
        consultation = requester.consult(
            tilt_of(body_with_geometry()), situation_backend("situation_granular.json")
        )
        assert consultation.perspective_record_ids == []
        assert len(requester.service.backend.client.calls) == 1

    def test_the_phase_structure_is_served_as_before(self, tmp_path):
        requester, consultation = consulted_with_the_frame(
            HypothesisStore(directory=tmp_path)
        )
        assert isinstance(
            requester.service.view(consultation.record_id).root,
            pouring.PouringPhaseStructure,
        )

    def test_offering_the_frame_serves_the_core_and_the_frame_on_it(self, tmp_path):
        requester, consultation = consulted_with_the_frame(
            HypothesisStore(directory=tmp_path)
        )
        core_record, frame_record = consultation.perspective_record_ids
        core_view = requester.service.view(core_record)
        frame_view = requester.service.view(frame_record)
        assert isinstance(core_view.root, action_core.PouringFrame)
        assert frame_view.root.stereotype_of is core_view.root

    def test_the_tools_read_the_phases_the_core_and_the_frame_together(self, tmp_path):
        requester, consultation = consulted_with_the_frame(
            HypothesisStore(directory=tmp_path)
        )
        overview = json.loads(
            ToolBox(tools=requester.tools_for(consultation))
            .call(KnowledgeToolName.KNOWLEDGE_OVERVIEW, {})
            .content
        )
        phase_structure = requester.service.view(consultation.record_id).root
        assert overview["structure"]["name"] == type(phase_structure).__name__
        assert [perspective["module"] for perspective in overview["perspectives"]] == [
            ActionPerspectiveFile.ACTION_FRAME.value
        ]

    def test_a_ground_may_name_a_fragment_of_the_frame(self, tmp_path):
        requester, consultation = consulted_with_the_frame(
            HypothesisStore(directory=tmp_path)
        )
        found = json.loads(
            ToolBox(tools=requester.tools_for(consultation))
            .call(
                KnowledgeToolName.FIND_FRAGMENTS,
                {"kind": action_frame.SlotExpectation.__name__},
            )
            .content
        )
        name = found["fragments"][0]["name"]
        (grounds,) = requester.grounds_in(consultation)
        frame_record = consultation.perspective_record_ids[-1]
        assert grounds.identify(name) == requester.service.citation(frame_record, name)
        assert grounds.identify(name) is not None

    def test_the_rules_say_how_to_read_the_frame_only_when_it_is_offered(
        self, tmp_path
    ):
        requester, consultation = consulted_with_the_frame(
            HypothesisStore(directory=tmp_path / "with_frame")
        )
        assert FRAME_KNOWLEDGE_RULE in requester.rules_for(consultation)
        plain = requester_generating(
            HypothesisStore(directory=tmp_path / "plain"),
            GenerationCase.GRANULAR_POURING,
        )
        plain_consultation = plain.consult(
            tilt_of(body_with_geometry()), situation_backend("situation_granular.json")
        )
        assert plain.rules_for(plain_consultation) == [KNOWLEDGE_RULE]

    def test_a_resolution_offered_the_frame_states_both_knowledge_rules(self, tmp_path):
        backend = LanguageModelBackend.with_world_tools(
            client=ScriptedClient(
                tool_turns=[
                    answer("situation_granular.json"),
                    answer("angle_chosen.json"),
                ]
            ),
            world=WorldView(),
            knowledge=requester_offering_the_frame(HypothesisStore(directory=tmp_path)),
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert {KNOWLEDGE_RULE, FRAME_KNOWLEDGE_RULE} <= set(backend.rules)

    def test_the_consultation_reports_the_core_and_the_perspectives_served(
        self, tmp_path
    ):
        sink = MemorySink()
        backend = LanguageModelBackend.with_world_tools(
            client=ScriptedClient(
                tool_turns=[
                    answer("situation_granular.json"),
                    answer("angle_chosen.json"),
                ]
            ),
            world=WorldView(),
            knowledge=requester_offering_the_frame(HypothesisStore(directory=tmp_path)),
            observer=sink,
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        (consulted,) = [
            event for event in sink.events if isinstance(event, KnowledgeConsulted)
        ]
        assert list(consulted.perspective_record_ids) == (
            backend.consultation.perspective_record_ids
        )


def slot_entries_of_request(backend: LanguageModelBackend) -> dict:
    """
    :return: The open slots' entries in the latest request to the model, by field
        name.
    """
    (slots,) = bodies(backend.log[-1].user, PromptSection.SLOTS)
    return {
        entry[SlotKey.PATH].rpartition(".")[2]: entry for entry in json.loads(slots)
    }


def world_backend_offering_the_frame(store: HypothesisStore) -> LanguageModelBackend:
    """
    :return: A backend asking with the world tools that consults the core and the
        action frame about the tilting action framed as pouring.
    """
    return LanguageModelBackend.with_world_tools(
        client=ScriptedClient(
            tool_turns=[answer("situation_granular.json"), answer("angle_chosen.json")]
        ),
        world=WorldView(),
        knowledge=requester_offering_the_frame(store),
    )


class TestSlotsNamingTheFrame:
    def test_an_open_slot_names_the_frame_parts_it_fills(self, tmp_path):
        backend = world_backend_offering_the_frame(HypothesisStore(directory=tmp_path))
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        core_record = backend.consultation.perspective_record_ids[0]
        service = backend.knowledge.service
        view = service.view(core_record)
        (part,) = TILT_FRAMED_AS_POURING.frame_fields.values()
        filled = part.fragments_in(view.root)
        names = slot_entries_of_request(backend)["angle"][SlotKey.FRAME]
        assert [service.citation(core_record, name) for name in names] == [
            view.citation_of(fragment).iri for fragment in filled
        ]

    def test_without_perspectives_no_slot_names_the_frame(self, tmp_path):
        requester = requester_generating(
            HypothesisStore(directory=tmp_path), GenerationCase.GRANULAR_POURING
        )
        requester.described_actions = DescribedActions(entries=[TILT_FRAMED_AS_POURING])
        backend = LanguageModelBackend.with_world_tools(
            client=ScriptedClient(
                tool_turns=[
                    answer("situation_granular.json"),
                    answer("angle_chosen.json"),
                ]
            ),
            world=WorldView(),
            knowledge=requester,
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        assert SlotKey.FRAME not in slot_entries_of_request(backend)["angle"]
