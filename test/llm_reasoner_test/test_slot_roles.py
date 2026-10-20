"""
Tests of the file linking the two families: which manipulation role of the predictive
manipulation microworld fills which slot of the action core's frame.

The test that reasons needs CRAM-AICOR, DUL, ``owlready2`` and a Java runtime, is
skipped without them and is slow.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef

from llm_reasoner.knowledge.ontologies import (
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.described_actions import DescribedActions
from llm_reasoner.knowledge.ontologies.frame_fields import SlotDescription
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import PropertyTerm
from llm_reasoner.knowledge.ontologies.slot_roles import (
    SLOT_ROLES_FILE,
    SLOT_ROLES_NAMESPACE,
    SlotRoles,
)

from .test_predictive_manipulation_ontology import (
    cram_aicor_path,
    declared_terms,
    dul_path,
    ontology_iri,
    owlready2,
    referenced_terms,
)

POURING_ACTION = "coraplex.robot_plans.actions.composite.tool_based.PouringAction"
"""
The designator whose fields the mapping kept as data maps to the frame.
"""

IS_PART_OF_SLOT = URIRef(
    PropertyTerm.of_field(action_core.PartSlot, "is_part_of_slot").iri
)
"""
The core's property relating a part slot to the slot of the participant it is part of.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The file linking the families.
    """
    return Graph().parse(SLOT_ROLES_FILE, format="turtle")


@pytest.fixture(scope="module")
def families() -> Graph:
    """
    Every module of both families in one graph.
    """
    union = Graph()
    for module in KNOWLEDGE_ONTOLOGY_MODULES:
        union.parse(module.path, format="turtle")
    return union


def named_bases(graph: Graph, cls: URIRef) -> set[URIRef]:
    """
    :return: The named classes the graph puts the class directly under.
    """
    return {
        base for base in graph.objects(cls, RDFS.subClassOf) if isinstance(base, URIRef)
    }


def all_bases(graph: Graph, cls: URIRef) -> set[URIRef]:
    """
    :return: Every named class the graph puts the class under, directly or not.
    """
    found = {cls}
    pending = [cls]
    while pending:
        for base in named_bases(graph, pending.pop()) - found:
            found.add(base)
            pending.append(base)
    return found - {cls}


def linked_roles(graph: Graph) -> set[URIRef]:
    """
    :return: The classes the file puts under a slot of the core.
    """
    return {
        subject
        for subject, base in graph.subject_objects(RDFS.subClassOf)
        if isinstance(subject, URIRef)
        and str(base).startswith(ActionPerspectiveFile.ACTION_CORE.namespace)
    }


# %% the file


class TestFile:
    def test_it_imports_the_core_and_the_modules_of_the_roles_it_names(self, graph):
        assert set(graph.objects(ontology_iri(graph), OWL.imports)) == {
            URIRef(module.namespace.rstrip("#"))
            for module in (
                ActionPerspectiveFile.ACTION_CORE,
                PredictiveManipulationOntologyFile.CORE,
                PredictiveManipulationOntologyFile.PREHENSION,
                PredictiveManipulationOntologyFile.POURING,
            )
        }

    def test_its_namespace_is_its_own(self, graph):
        assert str(ontology_iri(graph)) == SLOT_ROLES_NAMESPACE.rstrip("#")

    def test_it_declares_no_term_of_its_own(self, graph):
        assert declared_terms(graph) == set()

    def test_it_uses_only_terms_the_families_declare(self, graph, families):
        imported = set(graph.objects(None, OWL.imports))
        assert referenced_terms(graph) - imported - {ontology_iri(graph)} <= (
            declared_terms(families)
        )

    def test_it_puts_only_manipulation_roles_under_slots(self, graph, families):
        manipulation_role = URIRef(core.ManipulationRole.term.iri)
        for role in linked_roles(graph):
            assert manipulation_role in all_bases(families, role), role

    def test_it_puts_them_only_under_frame_slots(self, graph, families):
        frame_slot = URIRef(action_core.FrameSlot.term.iri)
        for role in linked_roles(graph):
            for slot in named_bases(graph, role):
                assert frame_slot in all_bases(families, slot), slot

    def test_a_role_and_its_slot_specialise_the_same_linked_roles(
        self, graph, families
    ):
        own = (
            ActionPerspectiveFile.ACTION_CORE.namespace,
            *(module.namespace for module in PredictiveManipulationOntologyFile),
        )
        for role in linked_roles(graph):
            for slot in named_bases(graph, role):
                slot_links = {
                    base
                    for base in all_bases(families, slot)
                    if not str(base).startswith(own)
                }
                assert slot_links <= all_bases(families, role), (role, slot)

    def test_the_pouring_lip_is_a_part_of_the_source(self, graph):
        lip = URIRef(pouring.PouringLip.term.iri)
        assert any(
            isinstance(node, BNode)
            and (node, RDF.type, OWL.Restriction) in graph
            and (node, OWL.onProperty, IS_PART_OF_SLOT) in graph
            and graph.value(node, OWL.someValuesFrom)
            == URIRef(action_core.SourceSlot.term.iri)
            for node in graph.objects(lip, RDFS.subClassOf)
        )


# %% reading it


class TestSlotRoles:
    @pytest.mark.parametrize(
        "slot, role",
        [
            (action_core.AgentSlot, core.Actor),
            (action_core.PatientSlot, core.Manipulandum),
            (action_core.InstrumentSlot, prehension.GraspingEffector),
            (action_core.SourceSlot, pouring.SourceContainer),
            (action_core.GoalSlot, pouring.TargetReceptacle),
            (action_core.SubstanceSlot, pouring.PouredSubstance),
            (action_core.PartSlot, pouring.PouringLip),
        ],
    )
    def test_the_role_filling_a_slot_is_the_one_the_file_puts_under_it(
        self, slot, role
    ):
        assert SlotRoles.shipped().role_filling(slot) is role

    def test_a_slot_the_file_puts_no_role_under_has_none(self):
        assert SlotRoles.shipped().role_filling(action_core.FrameSlot) is None


class TestFieldRoles:
    def test_a_field_filling_a_slot_fills_the_role_under_that_slot(self):
        described = DescribedActions().describing(POURING_ACTION)
        slot_fields = {
            name: part
            for name, part in described.frame_fields.items()
            if isinstance(part, SlotDescription)
        }
        assert slot_fields
        assert described.field_roles(SlotRoles.shipped()) == {
            name: SlotRoles.shipped().role_filling(part.slot)
            for name, part in slot_fields.items()
        }

    def test_the_pouring_containers_fill_the_pouring_roles(self):
        described = DescribedActions().describing(POURING_ACTION)
        roles = described.field_roles(SlotRoles.shipped())
        assert roles["source_container"] is pouring.SourceContainer
        assert roles["target_container"] is pouring.TargetReceptacle


# %% reasoning


@pytest.mark.slow
class TestReasoning:
    def test_both_families_and_the_links_are_consistent_and_every_class_satisfiable(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        for module in KNOWLEDGE_ONTOLOGY_MODULES:
            merged.parse(module.path, format="turtle")
        merged.parse(SLOT_ROLES_FILE, format="turtle")
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        namespaces = tuple(module.namespace for module in KNOWLEDGE_ONTOLOGY_MODULES)
        assert [
            cls
            for cls in world.classes()
            if cls.iri.startswith(namespaces) and owlready2.Nothing in cls.equivalent_to
        ] == []
