"""
Tests that the manipulation roles of the predictive manipulation microworld are kinds of
the action core's frame slots: each role is put under the slot whose CRAM-AICOR role it
already specialises, a part of a participant is a part slot of that participant, and the
role filling a slot is read from the generated classes.
"""

from __future__ import annotations

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef

from llm_reasoner.knowledge.ontologies import KNOWLEDGE_ONTOLOGY_MODULES
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.described_actions import DescribedActions
from llm_reasoner.knowledge.ontologies.exceptions import SeveralRolesFillSlot
from llm_reasoner.knowledge.ontologies.frame_fields import SlotDescription
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    gaze,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import PropertyTerm

POURING_ACTION = "coraplex.robot_plans.actions.composite.tool_based.PouringAction"
"""
The designator whose fields the mapping kept as data maps to the frame.
"""

ROLES_UNDER_SLOTS = [
    (core.Actor, action_core.AgentSlot),
    (core.Manipulandum, action_core.PatientSlot),
    (prehension.GraspingEffector, action_core.InstrumentSlot),
    (prehension.GraspSite, action_core.PartSlot),
    (prehension.InitialSupport, action_core.ParticipantSlot),
    (prehension.FinalSupport, action_core.ParticipantSlot),
    (gaze.Obstacle, action_core.ParticipantSlot),
    (pouring.SourceContainer, action_core.SourceSlot),
    (pouring.TargetReceptacle, action_core.GoalSlot),
    (pouring.PouredSubstance, action_core.SubstanceSlot),
    (pouring.PouringLip, action_core.PartSlot),
]
"""
Each manipulation role with the slot of the core it is directly put under.
"""

PARTS = [
    (pouring.PouringLip, action_core.SourceSlot),
    (prehension.GraspSite, action_core.PatientSlot),
]
"""
Each role that is a part of a participant, with the slot of the participant.
"""

# %% fixtures


@pytest.fixture(scope="module")
def families() -> Graph:
    """
    Every module of both families in one graph.
    """
    union = Graph()
    for module in KNOWLEDGE_ONTOLOGY_MODULES:
        union.parse(module.path, format="turtle")
    return union


# %% roles as slots


class TestRolesAsSlots:
    def test_a_manipulation_role_is_a_frame_slot(self):
        assert issubclass(core.ManipulationRole, action_core.FrameSlot)

    @pytest.mark.parametrize("role, slot", ROLES_UNDER_SLOTS)
    def test_a_role_is_directly_under_its_slot(self, role, slot):
        assert slot in role.__bases__

    @pytest.mark.parametrize("role, whole", PARTS)
    def test_a_part_is_a_part_of_its_participant(self, families, role, whole):
        is_part_of_slot = URIRef(
            PropertyTerm.of_field(
                clazz=action_core.PartSlot, field_name="is_part_of_slot"
            ).iri
        )
        assert any(
            isinstance(node, BNode)
            and (node, RDF.type, OWL.Restriction) in families
            and (node, OWL.onProperty, is_part_of_slot) in families
            and families.value(node, OWL.someValuesFrom) == URIRef(whole.term.iri)
            for node in families.objects(URIRef(role.term.iri), RDFS.subClassOf)
        )


# %% the role filling a slot


class TestRoleFillingASlot:
    @pytest.mark.parametrize(
        "slot, role",
        [
            (action_core.AgentSlot, core.Actor),
            (action_core.PatientSlot, core.Manipulandum),
            (action_core.InstrumentSlot, prehension.GraspingEffector),
            (action_core.SourceSlot, pouring.SourceContainer),
            (action_core.GoalSlot, pouring.TargetReceptacle),
            (action_core.SubstanceSlot, pouring.PouredSubstance),
        ],
    )
    def test_the_role_filling_a_slot_is_the_one_directly_under_it(self, slot, role):
        assert SlotDescription(slot=slot).role() is role

    def test_a_slot_no_role_is_directly_under_has_none(self):
        assert SlotDescription(slot=action_core.FrameSlot).role() is None

    def test_a_slot_several_roles_are_directly_under_is_refused(self):
        with pytest.raises(SeveralRolesFillSlot):
            SlotDescription(slot=action_core.PartSlot).role()


class TestFieldRoles:
    def test_a_field_filling_a_slot_fills_the_role_under_that_slot(self):
        described = DescribedActions().describing(POURING_ACTION)
        slot_fields = {
            name: part
            for name, part in described.frame_fields.items()
            if isinstance(part, SlotDescription)
        }
        assert slot_fields
        assert described.field_roles() == {
            name: part.role() for name, part in slot_fields.items()
        }

    def test_the_pouring_containers_fill_the_pouring_roles(self):
        roles = DescribedActions().describing(POURING_ACTION).field_roles()
        assert roles["source_container"] is pouring.SourceContainer
        assert roles["target_container"] is pouring.TargetReceptacle
