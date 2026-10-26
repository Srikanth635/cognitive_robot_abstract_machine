"""
Tests that a condition a world can check about the filler of a slot is one term of the
action core: the action frame requires and expects such conditions, the failures observe
them, and each perspective keeps only the kinds of condition that are its own.
"""

from __future__ import annotations

import pytest
from rdflib import RDFS, Graph, URIRef
from typing_extensions import List, Optional, get_type_hints

from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.ontologies import (
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.acceptance.derived import RestrictionsHold
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
    failures,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import (
    DescriptionEntity,
    PropertyRestriction,
    PropertyTerm,
)
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.storage.store import HypothesisStore

from .test_action_perspective_knowledge import (
    AnswerCase,
    answer_text,
    core_query,
    frame_query_on,
    service_answering,
)

CONDITION_FIELDS = [
    (action_frame.FrameStereotype, "requires_before_start"),
    (action_frame.SlotExpectation, "expects_filler_to_satisfy"),
    (action_frame.DimensionRationale, "decides_whether"),
    (failures.Failure, "has_failed_condition"),
    (failures.ParameterChange, "corrects_condition"),
    (failures.ParameterChange, "requires_of_choice"),
]
"""
The fields of each perspective that hold conditions on slots.
"""

FRAME_CONDITIONS = [
    action_frame.FeatureCondition,
    action_frame.AffordanceCondition,
    action_frame.PostureCondition,
]
"""
The kinds of condition only the action frame has.
"""

PREDICTIVE_CONDITION_SLOTS = [
    (
        prehension.EffectorObjectContact,
        prehension.GraspingEffector,
        (core.Manipulandum, prehension.GraspSite),
    ),
    (
        prehension.EffectorPositioned,
        prehension.GraspingEffector,
        (prehension.GraspSite,),
    ),
    (prehension.GraspStability, core.Manipulandum, (prehension.GraspingEffector,)),
    (
        prehension.ObjectSupportContact,
        core.Manipulandum,
        (prehension.InitialSupport, prehension.FinalSupport),
    ),
    (
        prehension.ObjectClearOfSupport,
        core.Manipulandum,
        (prehension.InitialSupport, prehension.FinalSupport),
    ),
    (prehension.SafeContactForce, prehension.GraspingEffector, (core.Manipulandum,)),
    (prehension.ObjectIntact, core.Manipulandum, ()),
    (pouring.LipOverReceptacle, pouring.PouringLip, (pouring.TargetReceptacle,)),
    (pouring.SourceUpright, pouring.SourceContainer, ()),
    (
        pouring.SubstanceContainedInSource,
        pouring.PouredSubstance,
        (pouring.SourceContainer,),
    ),
    (
        pouring.SubstanceAccumulatedInTarget,
        pouring.PouredSubstance,
        (pouring.TargetReceptacle,),
    ),
    (pouring.SubstanceFlowing, pouring.PouredSubstance, (pouring.PouringLip,)),
]
"""
Each condition of the predictive manipulation microworld, with the class of the slot it
is on and the classes of the slots it may involve.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    Every module of both families in one graph.
    """
    union = Graph()
    for module in KNOWLEDGE_ONTOLOGY_MODULES:
        union.parse(module.path, format="turtle")
    return union


# %% the term


class TestTheTerm:
    def test_the_action_core_declares_the_condition_and_the_state_condition(self):
        assert action_core.SlotCondition.term.iri.startswith(
            ActionPerspectiveFile.ACTION_CORE.namespace
        )
        assert issubclass(action_core.StateCondition, action_core.SlotCondition)

    def test_a_condition_names_its_slot_another_slot_and_its_polarity(self):
        hints = get_type_hints(action_core.SlotCondition)
        assert hints["condition_on_slot"] == Optional[action_core.FrameSlot]
        assert hints["condition_involves_slot"] == List[action_core.FrameSlot]
        assert hints["has_polarity"] == Optional[action_core.ConditionPolarity]

    def test_a_state_condition_names_its_state(self):
        assert "condition_has_state" in get_type_hints(action_core.StateCondition)

    def test_no_perspective_keeps_a_general_condition_of_its_own(self):
        assert "SlotCondition" not in vars(action_frame)
        assert "StateCondition" not in vars(action_frame)
        assert "ObservableCondition" not in vars(failures)

    @pytest.mark.parametrize("owner, field_name", CONDITION_FIELDS)
    def test_every_condition_property_ranges_over_it(self, graph, owner, field_name):
        prop = URIRef(PropertyTerm.of_field(owner, field_name).iri)
        assert set(graph.objects(prop, RDFS.range)) == {
            URIRef(action_core.SlotCondition.term.iri)
        }


# %% the conditions of the predictive manipulation microworld


def violated_on(
    fragment: DescriptionEntity, property_term: PropertyTerm
) -> list[PropertyRestriction]:
    """
    :return: The restrictions on a property that a fragment violates.
    """
    return [
        violation.expression
        for violation in RestrictionsHold().violations([fragment])
        if isinstance(violation.expression, PropertyRestriction)
        and violation.expression.property_iri == property_term.iri
    ]


ON_SLOT = PropertyTerm.of_field(action_core.SlotConditionType, "condition_on_slot")
"""
The property naming the slot a condition is on.
"""

INVOLVES_SLOT = PropertyTerm.of_field(
    action_core.SlotConditionType, "condition_involves_slot"
)
"""
The property naming the other slots a condition relates the first to.
"""


class TestPredictiveManipulationConditions:
    def test_a_condition_type_of_the_microworld_is_one_of_the_core(self):
        assert issubclass(core.PhaseConditionType, action_core.SlotConditionType)
        assert not issubclass(core.PhaseConditionType, action_core.SlotCondition)

    def test_a_core_condition_is_a_condition_type_stated_with_a_polarity(self):
        assert issubclass(action_core.SlotCondition, action_core.SlotConditionType)
        assert "has_polarity" not in get_type_hints(action_core.SlotConditionType)
        assert "has_polarity" in get_type_hints(action_core.SlotCondition)

    def test_no_condition_of_the_microworld_lists_roles_or_states_a_polarity(self):
        names = {
            name
            for generated in Vocabulary().classes
            if issubclass(generated, core.PhaseConditionType)
            for name in get_type_hints(generated)
        }
        assert {"condition_on_role", "has_polarity"} & names == set()

    @pytest.mark.parametrize("condition, on, involved", PREDICTIVE_CONDITION_SLOTS)
    def test_a_condition_is_on_the_slot_its_class_names(self, condition, on, involved):
        assert violated_on(condition(condition_on_slot=on()), ON_SLOT) == []
        assert violated_on(condition(condition_on_slot=core.Actor()), ON_SLOT)

    @pytest.mark.parametrize(
        "condition, on, involved",
        [case for case in PREDICTIVE_CONDITION_SLOTS if case[2]],
    )
    def test_a_condition_involves_only_slots_its_class_names(
        self, condition, on, involved
    ):
        assert (
            violated_on(
                condition(
                    condition_on_slot=on(),
                    condition_involves_slot=[kind() for kind in involved],
                ),
                INVOLVES_SLOT,
            )
            == []
        )
        assert violated_on(
            condition(condition_on_slot=on(), condition_involves_slot=[core.Actor()]),
            INVOLVES_SLOT,
        )

    def test_a_condition_need_not_name_the_slots_it_involves(self):
        assert (
            violated_on(
                pouring.LipOverReceptacle(condition_on_slot=pouring.PouringLip()),
                INVOLVES_SLOT,
            )
            == []
        )


# %% the kinds only a perspective has


class TestTheFramesOwnKinds:
    @pytest.mark.parametrize("kind", FRAME_CONDITIONS)
    def test_a_frame_kind_specialises_the_core_condition(self, kind):
        assert issubclass(kind, action_core.SlotCondition)

    def test_a_frame_hypothesis_is_offered_its_own_kinds_and_the_cores(self):
        types = GenerationSchema.for_root(action_frame.FrameStereotype).types
        for kind in [
            action_core.SlotCondition,
            action_core.StateCondition,
            *FRAME_CONDITIONS,
        ]:
            assert kind.__name__ in types

    def test_a_failures_hypothesis_is_offered_the_cores_but_not_the_frames(self):
        types = GenerationSchema.for_root(failures.FailureStereotype).types
        assert action_core.SlotCondition.__name__ in types
        assert action_core.StateCondition.__name__ in types
        for kind in FRAME_CONDITIONS:
            assert kind.__name__ not in types


# %% stating a condition beside the given core


class TestStatingACoreCondition:
    def test_a_perspective_states_a_core_condition_no_core_hypothesis_states(
        self, tmp_path
    ):
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
        )
        core_view = service.view_for(core_query())
        frame_view = service.view_for(frame_query_on(core_view))
        assert len(frame_view.record.exchanges) == 1
        assert frame_view.instances_of(action_core.StateCondition)
