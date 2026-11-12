"""
Tests of the pouring frame, the action core's frame of pouring: the structure of its
module file, what it says of a pour, the situations a pour is performed in, the
vocabulary generated from it, and what an OWL reasoner concludes from it.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, Graph, Namespace, URIRef

from llm_reasoner.knowledge.vocabulary.ontology_modules import (
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionCoreAnnotation,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    pouring_frame,
)
from llm_reasoner.knowledge.vocabulary.translation.writing import VocabularyWriter

from .test_action_core_ontology import (
    AICOR,
    CORE,
    requires_value_of_kind,
    restriction_fillers,
    superclasses,
)
from .test_predictive_manipulation_ontology import (
    cram_aicor_path,
    declared_terms,
    dul_path,
    ontology_iri,
    owlready2,
)

POURING_FRAME = Namespace(ActionPerspectiveFile.POURING_FRAME.namespace)
"""
The terms of the pouring frame.
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
    The pouring frame module.
    """
    return Graph().parse(ActionPerspectiveFile.POURING_FRAME.path, format="turtle")


# %% the module file


class TestModuleFile:
    def test_it_comes_after_the_action_core_it_imports(self):
        assert KNOWLEDGE_ONTOLOGY_MODULES.index(
            ActionPerspectiveFile.ACTION_CORE
        ) < KNOWLEDGE_ONTOLOGY_MODULES.index(ActionPerspectiveFile.POURING_FRAME)

    def test_it_declares_one_ontology_importing_the_action_core(self, graph):
        iri = ontology_iri(graph)
        assert str(iri) == ActionPerspectiveFile.POURING_FRAME.namespace.rstrip("#")
        assert set(graph.objects(iri, OWL.imports)) == {
            URIRef(ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#"))
        }

    def test_every_term_is_declared_in_its_namespace(self, graph):
        assert all(
            str(term).startswith(ActionPerspectiveFile.POURING_FRAME.namespace)
            for term in declared_terms(graph)
            if term != ontology_iri(graph)
        )


# %% pouring


class TestPouringFrame:
    def test_it_is_a_frame_of_a_manipulation(self, graph):
        assert CORE.ManipulationActionFrame in superclasses(
            graph, POURING_FRAME.PouringFrame
        )

    def test_it_frames_cram_aicors_pouring(self, graph):
        assert AICOR.Pouring in restriction_fillers(
            graph, POURING_FRAME.PouringFrame, CORE.framesAction
        )

    @pytest.mark.parametrize("participant", POURING_PARTICIPANTS)
    def test_it_defines_a_slot_for_each_participant_of_a_pour(self, graph, participant):
        assert participant in restriction_fillers(
            graph, POURING_FRAME.PouringFrame, CORE.definesSlot
        )

    @pytest.mark.parametrize("whole", (CORE.SourceSlot, CORE.GoalSlot))
    def test_it_defines_a_part_of_the_source_and_of_the_goal(self, graph, whole):
        assert requires_value_of_kind(
            graph,
            POURING_FRAME.PouringFrame,
            CORE.definesSlot,
            CORE.PartSlot,
            (CORE.isPartOfSlot, whole),
        )

    def test_it_is_for_the_substance_having_accumulated_in_the_goal(self, graph):
        assert requires_value_of_kind(
            graph,
            POURING_FRAME.PouringFrame,
            CORE.statesObjective,
            CORE.Objective,
            (CORE.aimsAtState, AICOR.AccumulatedState),
        )

    def test_it_brings_about_a_consequence_and_leaves_choices_open(self, graph):
        assert CORE.Consequence in restriction_fillers(
            graph, POURING_FRAME.PouringFrame, CORE.bringsAbout
        )
        assert CORE.ParameterDimension in restriction_fillers(
            graph, POURING_FRAME.PouringFrame, CORE.leavesOpen
        )


# %% the situations a pour is performed in


class TestSituations:
    @pytest.mark.parametrize(
        "partition",
        (POURING_FRAME.MaterialRegime, POURING_FRAME.TransferTerminationReason),
    )
    def test_its_situations_are_open_value_partitions(self, graph, partition):
        assert (
            partition,
            URIRef(ActionCoreAnnotation.IS_OPEN_VALUE_PARTITION),
            None,
        ) in graph

    def test_every_situation_defines_a_variant_of_the_frame(self, graph):
        members = {
            member
            for partition in (
                POURING_FRAME.MaterialRegime,
                POURING_FRAME.TransferTerminationReason,
            )
            for member in graph.subjects(RDF.type, partition)
        }
        defining = {
            value
            for definition in graph.objects(None, OWL.equivalentClass)
            for value in graph.transitive_objects(definition, None)
            if value in members
        }
        assert defining == members


# %% the generated vocabulary


class TestVocabulary:
    def test_the_committed_module_matches_the_regenerated_source(self):
        module_file = ActionPerspectiveFile.POURING_FRAME
        assert module_file.vocabulary_path.read_text() == (
            VocabularyWriter.for_knowledge_modules().source(module_file)
        )

    def test_a_pouring_frame_is_a_frame_of_a_manipulation_in_python_too(self):
        assert issubclass(
            pouring_frame.PouringFrame, action_core.ManipulationActionFrame
        )
        assert issubclass(action_core.ManipulationActionFrame, action_core.ActionFrame)


# %% reasoning


@pytest.mark.slow
class TestReasoning:
    def test_it_is_consistent_with_the_core_cram_aicor_and_dul_and_satisfiable(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        merged.parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
        merged.parse(ActionPerspectiveFile.POURING_FRAME.path, format="turtle")
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        frame_classes = [
            cls
            for cls in world.classes()
            if cls.iri.startswith(ActionPerspectiveFile.POURING_FRAME.namespace)
        ]
        assert frame_classes
        assert [
            cls for cls in frame_classes if owlready2.Nothing in cls.equivalent_to
        ] == []
