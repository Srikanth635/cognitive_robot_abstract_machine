"""
Tests of the resolver consulting the knowledge component about the action a query asks
for: the action stated to a resolution serving it, the situation features filled in a
first pass, and the hypothesis served for them.
"""

from __future__ import annotations

import json
from pathlib import Path

from krrood.entity_query_language.factories import the

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    ConceptReference,
    CRAM_AICOR_NAMESPACE,
    DescribedAction,
    DescribedActions,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
    SituationFeature,
    SituationFeatures,
    pouring,
)
from llm_reasoner.resolver.action_context import ActionContextProvider
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.context import StructuralCandidateRenderer
from llm_reasoner.resolver.knowledge import KnowledgeRequester
from llm_reasoner.resolver.prompt_format import PromptSection
from llm_reasoner.resolver.prompts import (
    ESTABLISHED_VALUES_RULE,
    INPUT_MEANINGS,
    OpenValueMeaning,
    PromptBuilder,
    SourceDetail,
)
from llm_reasoner.resolver.reflection import SlotKind, reflect_slots
from llm_reasoner.world.access import identifier_of

from .prompt_sections import bodies, text_outside_sections
from .test_generation import GenerationCase
from .world_mimics import ActionWithBodyAndValueSlots, body_with_geometry

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
