"""
Tests of the action core, the hub every action perspective attaches to: the structure of
its module file, what it says of an action's frame and of pouring in particular, its
links to CRAM-AICOR and DUL, the vocabulary generated from it, and what an OWL reasoner
concludes from it.

As for the predictive manipulation modules, tests that need CRAM-AICOR or DUL read them
from the files named by :class:`LinkedOntologyVariable` and are skipped when those are
unset, and the test that reasons needs ``owlready2`` and a Java runtime and is slow.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Graph, Namespace, URIRef
from rdflib.collection import Collection
from typing_extensions import Optional, Set

from llm_reasoner.knowledge.ontologies import (
    ACTION_PERSPECTIVES_NAMESPACE_ROOT,
    CRAM_AICOR_NAMESPACE,
    CRAM_AICOR_ONTOLOGY_IRI,
    DUL_NAMESPACE,
    DUL_ONTOLOGY_IRI,
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontology_translation.writing import VocabularyWriter

from .test_predictive_manipulation_ontology import (
    cram_aicor_graph,
    cram_aicor_path,
    declared_terms,
    dul_graph,
    dul_path,
    ontology_iri,
    owlready2,
    referenced_terms,
)

CORE = Namespace(ActionPerspectiveFile.ACTION_CORE.namespace)
"""
The terms of the action core.
"""

AICOR = Namespace(CRAM_AICOR_NAMESPACE)
"""
The terms of CRAM-AICOR.
"""

DUL = Namespace(DUL_NAMESPACE)
"""
The terms of DOLCE+DnS Ultralite.
"""

PARTICIPANT_SLOTS = {
    CORE.AgentSlot: AICOR.AgentRole,
    CORE.SourceSlot: AICOR.SourceRole,
    CORE.SubstanceSlot: AICOR.SubstanceRole,
    CORE.GoalSlot: AICOR.GoalRole,
    CORE.InstrumentSlot: AICOR.InstrumentRole,
    CORE.PatientSlot: AICOR.PatientRole,
}
"""
Each kind of participant slot, with the CRAM-AICOR participant role it specialises.
"""

POURING_PARTICIPANTS = (
    CORE.AgentSlot,
    CORE.SourceSlot,
    CORE.SubstanceSlot,
    CORE.GoalSlot,
)
"""
The participants of a pour, as CRAM-AICOR's pouring intention names them.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The action core module.
    """
    return Graph().parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")


def superclasses(graph: Graph, cls: URIRef) -> Set[URIRef]:
    """
    :return: The named classes a class is stated to specialise, directly or not.
    """
    found: Set[URIRef] = set()
    pending = [cls]
    while pending:
        for parent in graph.objects(pending.pop(), RDFS.subClassOf):
            if isinstance(parent, URIRef) and parent not in found:
                found.add(parent)
                pending.append(parent)
    return found


def restriction_fillers(graph: Graph, cls: URIRef, prop: URIRef) -> list:
    """
    :return: What the restrictions of a class on a property require: the class a value
        must belong to, or the value it must be.
    """
    return [
        graph.value(restriction, OWL.someValuesFrom)
        or graph.value(restriction, OWL.hasValue)
        for restriction in graph.objects(cls, RDFS.subClassOf)
        if isinstance(restriction, BNode)
        and graph.value(restriction, OWL.onProperty) == prop
    ]


def intersection_parts(graph: Graph, expression) -> list:
    """
    :return: The classes and restrictions an intersection is made of; the expression
        itself when it is no intersection.
    """
    members = graph.value(expression, OWL.intersectionOf)
    return [expression] if members is None else list(Collection(graph, members))


def requires_value_of_kind(
    graph: Graph,
    cls: URIRef,
    prop: URIRef,
    kind: URIRef,
    condition: Optional[tuple] = None,
) -> bool:
    """
    :return: Whether a class requires a value of a property belonging to a kind, and,
        when a condition (a property and what it requires) is given, meeting it.
    """
    for filler in restriction_fillers(graph, cls, prop):
        parts = intersection_parts(graph, filler)
        if kind not in parts:
            continue
        if condition is None:
            return True
        condition_property, condition_filler = condition
        if any(
            isinstance(part, BNode)
            and graph.value(part, OWL.onProperty) == condition_property
            and condition_filler
            in (graph.value(part, OWL.someValuesFrom), graph.value(part, OWL.hasValue))
            for part in parts
        ):
            return True
    return False


# %% the module file


class TestModuleFile:
    def test_the_core_comes_before_every_other_module(self):
        assert KNOWLEDGE_ONTOLOGY_MODULES[0] is ActionPerspectiveFile.ACTION_CORE

    def test_it_declares_one_ontology_importing_cram_aicor_and_dul(self, graph):
        iri = ontology_iri(graph)
        assert str(iri) == ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#")
        assert set(graph.objects(iri, OWL.imports)) == {
            URIRef(CRAM_AICOR_ONTOLOGY_IRI),
            URIRef(DUL_ONTOLOGY_IRI),
        }

    def test_every_term_is_declared_in_its_namespace(self, graph):
        assert all(
            str(term).startswith(ActionPerspectiveFile.ACTION_CORE.namespace)
            for term in declared_terms(graph)
            if term != ontology_iri(graph)
        )

    def test_only_its_own_cram_aicor_and_dul_terms_are_used(self, graph):
        outside = {
            term
            for term in referenced_terms(graph)
            if not str(term).startswith(
                (
                    ACTION_PERSPECTIVES_NAMESPACE_ROOT,
                    CRAM_AICOR_NAMESPACE,
                    DUL_NAMESPACE,
                )
            )
            and term not in (URIRef(CRAM_AICOR_ONTOLOGY_IRI), URIRef(DUL_ONTOLOGY_IRI))
        }
        assert outside == set()

    def test_every_term_has_a_label_and_a_definition(self, graph):
        for term in declared_terms(graph):
            assert graph.value(term, RDFS.label) is not None, term
            assert graph.value(term, RDFS.comment) is not None, term

    def test_every_property_is_attached_to_a_class(self, graph):
        restricted = set(graph.objects(None, OWL.onProperty))
        for property_type in (OWL.ObjectProperty, OWL.DatatypeProperty):
            for prop in graph.subjects(RDF.type, property_type):
                assert (
                    graph.value(prop, RDFS.domain) is not None or prop in restricted
                ), prop

    def test_every_property_range_is_a_class_of_the_core(self, graph):
        for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
            for value_range in graph.objects(prop, RDFS.range):
                assert str(value_range).startswith(CORE), prop


# %% frames


class TestFrames:
    def test_a_frame_is_a_description(self, graph):
        assert DUL.Description in superclasses(graph, CORE.ActionFrame)

    def test_a_frame_of_a_manipulation_is_what_cram_aicor_calls_a_manipulation_frame(
        self, graph
    ):
        assert {CORE.ActionFrame, AICOR.ManipulationFrame} <= superclasses(
            graph, CORE.ManipulationActionFrame
        )

    def test_a_frame_of_any_action_is_not_necessarily_a_manipulation_frame(self, graph):
        assert AICOR.ManipulationFrame not in superclasses(graph, CORE.ActionFrame)

    def test_every_kind_of_participant_slot_specialises_its_cram_aicor_role(
        self, graph
    ):
        for slot, role in PARTICIPANT_SLOTS.items():
            assert {CORE.ParticipantSlot, role} <= superclasses(graph, slot), slot

    def test_every_slot_is_a_cram_aicor_role_slot(self, graph):
        for slot in (CORE.ParticipantSlot, CORE.PartSlot):
            assert AICOR.RoleSlot in superclasses(graph, slot), slot

    def test_an_objective_is_a_cram_aicor_telos(self, graph):
        assert AICOR.Telos in superclasses(graph, CORE.Objective)

    def test_a_consequence_is_a_cram_aicor_effect(self, graph):
        assert AICOR.Effect in superclasses(graph, CORE.Consequence)

    def test_a_parameter_dimension_is_a_dul_parameter_of_cram_aicor_parameter_space(
        self, graph
    ):
        assert {DUL.Parameter, AICOR.ActionParameterSpace} <= superclasses(
            graph, CORE.ParameterDimension
        )

    def test_the_kind_of_quantity_is_not_reserved_for_parameter_dimensions(self, graph):
        assert graph.value(CORE.hasQuantityKind, RDFS.domain) is None


# %% pouring


class TestPouringFrame:
    def test_it_is_a_frame_of_a_manipulation(self, graph):
        assert CORE.ManipulationActionFrame in superclasses(graph, CORE.PouringFrame)

    def test_it_frames_cram_aicors_pouring(self, graph):
        assert AICOR.Pouring in restriction_fillers(
            graph, CORE.PouringFrame, CORE.framesAction
        )

    @pytest.mark.parametrize("participant", POURING_PARTICIPANTS)
    def test_it_defines_a_slot_for_each_participant_of_a_pour(self, graph, participant):
        assert participant in restriction_fillers(
            graph, CORE.PouringFrame, CORE.definesSlot
        )

    @pytest.mark.parametrize("whole", (CORE.SourceSlot, CORE.GoalSlot))
    def test_it_defines_a_part_of_the_source_and_of_the_goal(self, graph, whole):
        assert requires_value_of_kind(
            graph,
            CORE.PouringFrame,
            CORE.definesSlot,
            CORE.PartSlot,
            (CORE.isPartOfSlot, whole),
        )

    def test_it_is_for_the_substance_having_accumulated_in_the_goal(self, graph):
        assert requires_value_of_kind(
            graph,
            CORE.PouringFrame,
            CORE.statesObjective,
            CORE.Objective,
            (CORE.aimsAtState, AICOR.AccumulatedState),
        )

    def test_it_brings_about_a_consequence_and_leaves_choices_open(self, graph):
        assert CORE.Consequence in restriction_fillers(
            graph, CORE.PouringFrame, CORE.bringsAbout
        )
        assert CORE.ParameterDimension in restriction_fillers(
            graph, CORE.PouringFrame, CORE.leavesOpen
        )


# %% the generated vocabulary


class TestVocabulary:
    def test_the_committed_module_matches_the_regenerated_source(self):
        module_file = ActionPerspectiveFile.ACTION_CORE
        assert module_file.vocabulary_path.read_text() == (
            VocabularyWriter.for_knowledge_modules().source(module_file)
        )

    def test_a_pouring_frame_is_a_frame_of_a_manipulation_in_python_too(self):
        assert issubclass(action_core.PouringFrame, action_core.ManipulationActionFrame)
        assert issubclass(action_core.ManipulationActionFrame, action_core.ActionFrame)

    def test_the_kinds_of_quantity_are_an_enumeration_of_the_ontologys_members(
        self, graph
    ):
        members = set(graph.subjects(RDF.type, CORE.QuantityKind))
        assert {member.name for member in action_core.QuantityKind} == {
            str(member).split("#")[-1].upper() for member in members
        }


# %% links to CRAM-AICOR and DUL


class TestExternalLinks:
    def test_every_cram_aicor_term_used_is_declared_by_cram_aicor(
        self, graph, cram_aicor_graph
    ):
        used = {
            term
            for term in referenced_terms(graph)
            if str(term).startswith(CRAM_AICOR_NAMESPACE)
        }
        assert used - declared_terms(cram_aicor_graph) == set()

    def test_every_dul_term_used_is_declared_by_dul(self, graph, dul_graph):
        used = {
            term
            for term in referenced_terms(graph)
            if str(term).startswith(DUL_NAMESPACE)
        }
        assert used - declared_terms(dul_graph) == set()


# %% reasoning


@pytest.mark.slow
class TestReasoning:
    def test_the_core_is_consistent_with_cram_aicor_and_dul_and_every_class_satisfiable(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        merged.parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        core_classes = [
            cls
            for cls in world.classes()
            if cls.iri.startswith(ActionPerspectiveFile.ACTION_CORE.namespace)
        ]
        assert core_classes
        assert [
            cls for cls in core_classes if owlready2.Nothing in cls.equivalent_to
        ] == []
