"""
Tests of the predictive manipulation microworld ontology: the structure of its module
files, its links to the CRAM-AICOR and DUL ontologies, and what an OWL reasoner
concludes from it.

Tests that need CRAM-AICOR or DUL read them from the files named by
:class:`LinkedOntologyVariable` and are skipped when those are unset. Tests that reason
additionally need ``owlready2`` and a Java runtime for HermiT; each run classifies both
external ontologies with the microworld and takes minutes, so they are marked slow.
"""

from __future__ import annotations

import os
import shutil
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, RDFS, XSD, Graph, Namespace, URIRef
from typing_extensions import Any, Callable

from llm_reasoner.knowledge.ontologies import (
    ActionPerspectiveFile,
    CRAM_AICOR_NAMESPACE,
    CRAM_AICOR_ONTOLOGY_IRI,
    DUL_NAMESPACE,
    DUL_ONTOLOGY_IRI,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    LinkedOntologyVariable,
    PredictiveManipulationOntologyFile,
)

# %% locations


class ReasoningCase(StrEnum):
    """
    Assertions about individuals that a reasoner checks against the microworld.
    """

    POURING_PROTOTYPE = "pouring_prototype.ttl"
    """
    A complete generic pouring phase structure, as a language model would generate it.
    """

    FLOW_ONSET_CLASSIFICATION = "flow_onset_classification.ttl"
    """
    An observed event classified by a flow-onset event type.
    """

    SOURCE_CONTAINER_BINDING = "source_container_binding.ttl"
    """
    A world object bound during execution to the source container slot.
    """

    STRUCTURE_DEFINING_GRIP_FORCE_CORRIDOR = (
        "structure_defining_grip_force_corridor.ttl"
    )
    """
    A prehension structure that declares the grip-force corridor of its grasp.
    """

    MISMATCH_CONFIRMING_SUBGOAL = "mismatch_confirming_subgoal.ttl"
    """
    A comparison whose observation disagrees with its prediction, yet confirms a
    subgoal.
    """

    TRANSITION_PERMITTED_BY_MISMATCH = "transition_permitted_by_mismatch.ttl"
    """
    A phase transition that a mismatching comparison is said to permit.
    """

    PARAMETER_WITH_RANGE_AND_CATEGORY = "parameter_with_range_and_category.ttl"
    """
    A quantitative parameter that is also given a categorical value.
    """

    PREDICTION_TYPED_AS_OBSERVATION = "prediction_typed_as_observation.ttl"
    """
    One individual declared as both a sensory prediction and a sensory observation.
    """

    MASS_RANGE_AS_LIFTOFF_THRESHOLD = "mass_range_as_liftoff_threshold.ttl"
    """
    A lift-off threshold, which is a weight force, whose range is given in kilograms.
    """

    FRICTION_PREDICTED_FROM_VISION = "friction_predicted_from_vision.ttl"
    """
    A friction estimate said to be predicted from a visual cue, which vision cannot
    supply.
    """

    PHASE_TYPED_AS_PREDICTION = "phase_typed_as_prediction.ttl"
    """
    One individual declared both as a phase, a DUL task, and as a prediction, a DUL
    description.
    """

    FORCES_UNLOADED = "forces_unloaded.ttl"
    """
    The event at which the support takes over the weight from the grasp.
    """

    VISUAL_PREDICTION_OF_REACH_TARGET = "visual_prediction_of_reach_target.ttl"
    """
    A visual sensory goal at the hand reaching its target pose.
    """

    GRIP_FOLLOWING_LOAD_PREDICTION = "grip_following_load_prediction.ttl"
    """
    A grip-load coupling whose grip force follows a predicted change of load.
    """

    LIFTOFF_DRIVEN_BY_GRIP_FORCE = "liftoff_driven_by_grip_force.ttl"
    """
    A lift-off said to be driven by grip force rather than by load force.
    """

    SLIP_RESISTED_BY_WEIGHT = "slip_resisted_by_weight.ttl"
    """
    A slip said to be resisted by weight rather than by friction at the grasp.
    """

    CRUSH_DRIVEN_BY_LOAD_FORCE = "crush_driven_by_load_force.ttl"
    """
    Crushing said to be driven by load force rather than by grip force.
    """

    FLOW_ONSET_RESISTED_BY_FRICTION = "flow_onset_resisted_by_friction.ttl"
    """
    A flow onset said to overcome friction rather than containment by the source.
    """

    TACTILE_PREDICTION_OF_REACH_TARGET = "tactile_prediction_of_reach_target.ttl"
    """
    A touch prediction said to be caused by reaching a pose, which touch cannot sense.
    """

    @property
    def path(self) -> Path:
        """
        The location of the case's Turtle file.
        """
        return (
            Path(__file__).parent
            / "predictive_manipulation_ontology_cases"
            / self.value
        )


CORE = Namespace(PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + "core#")
"""
The namespace of the core module, which declares the provenance vocabulary.
"""

CASE = Namespace(PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + "case#")
"""
The namespace of the individuals in the reasoning cases.
"""

CRAM_AICOR = Namespace(CRAM_AICOR_NAMESPACE)
"""
The namespace of the CRAM-AICOR ontology.
"""

DUL = Namespace(DUL_NAMESPACE)
"""
The namespace of the DOLCE+DnS Ultralite ontology.
"""

ACTION_CORE_ONTOLOGY_IRI = ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#")
"""
The action core, whose condition polarity the core module uses.
"""

EXTERNAL_ONTOLOGIES = (
    CRAM_AICOR_ONTOLOGY_IRI,
    DUL_ONTOLOGY_IRI,
    ACTION_CORE_ONTOLOGY_IRI,
)
"""
The ontologies the core module imports, whose terms the microworld links to or uses.
"""

STANDARD_NAMESPACES = (str(RDF), str(RDFS), str(OWL), str(XSD))
"""
The namespaces of the languages the ontology is written in.
"""

DECLARATION_TYPES = (
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.NamedIndividual,
)
"""
The types whose instances are terms an ontology declares.
"""


# %% fixtures


@pytest.fixture(scope="module")
def module_graphs() -> dict[PredictiveManipulationOntologyFile, Graph]:
    """
    Every predictive manipulation module parsed on its own.
    """
    return {
        module: Graph().parse(module.path, format="turtle")
        for module in PredictiveManipulationOntologyFile
    }


def configured_path(variable: LinkedOntologyVariable) -> Path:
    """
    Return the ontology file an environment variable names, or skip when it names none.
    """
    location = os.environ.get(variable)
    if location is None or not Path(location).is_file():
        pytest.skip(f"set {variable} to the ontology file")
    return Path(location)


@pytest.fixture(scope="module")
def cram_aicor_path() -> Path:
    """
    The CRAM-AICOR ontology file.
    """
    return configured_path(LinkedOntologyVariable.CRAM_AICOR_ONTOLOGY)


@pytest.fixture(scope="module")
def dul_path() -> Path:
    """
    The DOLCE+DnS Ultralite ontology file.
    """
    return configured_path(LinkedOntologyVariable.DUL_ONTOLOGY)


@pytest.fixture(scope="module")
def cram_aicor_graph(cram_aicor_path: Path) -> Graph:
    """
    The CRAM-AICOR ontology.
    """
    return Graph().parse(cram_aicor_path, format="turtle")


@pytest.fixture(scope="module")
def dul_graph(dul_path: Path) -> Graph:
    """
    The DOLCE+DnS Ultralite ontology.
    """
    return Graph().parse(dul_path, format="xml")


@pytest.fixture(scope="module")
def owlready2() -> ModuleType:
    """
    The ``owlready2`` module, or a skip when it or a Java runtime is unavailable.
    """
    module = pytest.importorskip("owlready2")
    if shutil.which("java") is None:
        pytest.skip("HermiT needs a Java runtime")
    return module


@pytest.fixture
def reasoner(
    owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path: Path
) -> Callable[..., Any]:
    """
    Reason over both external ontologies, every predictive manipulation module and the
    cases given.
    """

    def reason_with(*cases: ReasoningCase) -> Any:
        return reason(owlready2, cram_aicor_path, dul_path, tmp_path, *cases)

    return reason_with


def declared_terms(graph: Graph) -> set[URIRef]:
    """
    Return every term the graph declares.
    """
    return {
        subject
        for declaration_type in DECLARATION_TYPES
        for subject in graph.subjects(RDF.type, declaration_type)
        if isinstance(subject, URIRef)
    }


def merged(graphs: dict[PredictiveManipulationOntologyFile, Graph]) -> Graph:
    """
    Return one graph holding every statement of the given module graphs.
    """
    union = Graph()
    for graph in graphs.values():
        union += graph
    return union


def microworld_properties(graph: Graph) -> set[URIRef]:
    """
    Return every object and datatype property the microworld declares.
    """
    return {
        prop
        for property_type in (OWL.ObjectProperty, OWL.DatatypeProperty)
        for prop in graph.subjects(RDF.type, property_type)
        if str(prop).startswith(PREDICTIVE_MANIPULATION_NAMESPACE_ROOT)
    }


def ontology_iri(graph: Graph) -> URIRef:
    """
    Return the IRI of the single ontology a module file declares.
    """
    (iri,) = graph.subjects(RDF.type, OWL.Ontology)
    return iri


def referenced_terms(graph: Graph) -> set[URIRef]:
    """
    Return every IRI the graph mentions outside the languages it is written in.
    """
    return {
        node
        for triple in graph
        for node in triple
        if isinstance(node, URIRef) and not str(node).startswith(STANDARD_NAMESPACES)
    }


def reason(
    owlready2: ModuleType,
    cram_aicor_path: Path,
    dul_path: Path,
    scratch: Path,
    *cases: ReasoningCase,
) -> Any:
    """
    Run HermiT over CRAM-AICOR, DUL, the action core, every predictive manipulation
    module and the given cases.

    The files are merged into one graph without their import statements, so the reasoner
    never fetches an ontology from the web.

    :return: The reasoned ``owlready2`` world.
    :raises owlready2.OwlReadyInconsistentOntologyError: when the merged ontology is
        inconsistent.
    """
    merged = Graph().parse(cram_aicor_path, format="turtle")
    merged.parse(dul_path, format="xml")
    merged.parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
    for path in [module.path for module in PredictiveManipulationOntologyFile] + [
        case.path for case in cases
    ]:
        merged.parse(path, format="turtle")
    merged.remove((None, OWL.imports, None))
    merged_path = scratch / "merged.nt"
    merged.serialize(merged_path, format="nt", encoding="utf-8")
    world = owlready2.World()
    world.get_ontology(merged_path.as_uri()).load()
    owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
    return world


def inferred_types(owlready2: ModuleType, world: Any, individual: URIRef) -> set[str]:
    """
    Return the IRIs of every named class an individual is inferred to belong to.
    """
    return {
        cls.iri
        for cls in world[str(individual)].INDIRECT_is_a
        if isinstance(cls, owlready2.ThingClass)
    }


# %% module files


class TestModuleFiles:
    """
    The module files are well-formed, self-describing and keep to their own scope.
    """

    def test_every_module_declares_one_ontology(self, module_graphs):
        for graph in module_graphs.values():
            assert len(list(graph.subjects(RDF.type, OWL.Ontology))) == 1

    def test_every_term_is_declared_in_its_module_namespace(self, module_graphs):
        for graph in module_graphs.values():
            namespace = str(ontology_iri(graph)) + "#"
            foreign = {
                term
                for term in declared_terms(graph)
                if not str(term).startswith(namespace)
            }
            assert foreign == set()

    def test_only_microworld_cram_aicor_dul_and_action_core_terms_are_used(
        self, module_graphs
    ):
        for graph in module_graphs.values():
            outside = {
                term
                for term in referenced_terms(graph)
                if not str(term).startswith(
                    (PREDICTIVE_MANIPULATION_NAMESPACE_ROOT, *EXTERNAL_ONTOLOGIES)
                )
            }
            assert outside == set()

    def test_external_terms_are_never_described(self, module_graphs):
        for graph in module_graphs.values():
            described = {
                subject
                for subject in graph.subjects()
                if str(subject).startswith(EXTERNAL_ONTOLOGIES)
            }
            assert described == set()

    def test_core_imports_cram_aicor_dul_and_the_action_core(self, module_graphs):
        imported = set(
            module_graphs[PredictiveManipulationOntologyFile.CORE].objects(
                None, OWL.imports
            )
        )
        assert imported == {URIRef(iri) for iri in EXTERNAL_ONTOLOGIES}

    def test_modules_import_only_modules_listed_before_them(self, module_graphs):
        modules = list(PredictiveManipulationOntologyFile)
        external = {URIRef(iri) for iri in EXTERNAL_ONTOLOGIES}
        for position, module in enumerate(modules):
            earlier = {
                ontology_iri(module_graphs[before]) for before in modules[:position]
            }
            imported = set(module_graphs[module].objects(None, OWL.imports))
            assert imported - external <= earlier

    def test_every_term_has_a_label_and_a_definition(self, module_graphs):
        for graph in module_graphs.values():
            undocumented = {
                term
                for term in declared_terms(graph)
                if graph.value(term, RDFS.label) is None
                or graph.value(term, RDFS.comment) is None
            }
            assert undocumented == set()

    def test_every_class_states_one_evidence_status(self, module_graphs):
        statuses = set(
            module_graphs[PredictiveManipulationOntologyFile.CORE].subjects(
                RDF.type, CORE.EvidenceStatus
            )
        )
        for graph in module_graphs.values():
            for cls in graph.subjects(RDF.type, OWL.Class):
                if not isinstance(cls, URIRef):
                    continue
                stated = set(graph.objects(cls, CORE.evidenceStatus))
                assert len(stated) == 1, cls
                assert stated <= statuses, cls

    def test_every_property_is_attached_to_a_class(self, module_graphs):
        microworld = merged(module_graphs)
        restricted = {
            microworld.value(restriction, OWL.onProperty)
            for cls, restriction in microworld.subject_objects(RDFS.subClassOf)
            if str(cls).startswith(PREDICTIVE_MANIPULATION_NAMESPACE_ROOT)
        }
        unattached = {
            prop
            for prop in microworld_properties(microworld)
            if microworld.value(prop, RDFS.domain) is None and prop not in restricted
        }
        assert unattached == set()

    def test_every_property_range_is_a_microworld_or_action_core_class_or_a_datatype(
        self, module_graphs
    ):
        microworld = merged(module_graphs)
        outside = {
            (prop, range_)
            for prop in microworld_properties(microworld)
            for range_ in microworld.objects(prop, RDFS.range)
            if not str(range_).startswith(
                (
                    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
                    ACTION_CORE_ONTOLOGY_IRI,
                    str(XSD),
                )
            )
        }
        assert outside == set()

    def test_every_cited_claim_is_a_declared_source_claim(self, module_graphs):
        claims = set(
            module_graphs[PredictiveManipulationOntologyFile.CORE].subjects(
                RDF.type, CORE.SourceClaim
            )
        )
        for graph in module_graphs.values():
            cited = set(graph.objects(None, CORE.representsClaim))
            assert cited <= claims


# %% links to CRAM-AICOR and DUL


class TestExternalLinks:
    """
    Every link into CRAM-AICOR or DUL names a term that ontology declares.
    """

    def test_every_cram_aicor_term_used_is_declared_by_cram_aicor(
        self, module_graphs, cram_aicor_graph
    ):
        available = declared_terms(cram_aicor_graph)
        for graph in module_graphs.values():
            used = {
                term
                for term in referenced_terms(graph)
                if str(term).startswith(CRAM_AICOR_NAMESPACE)
            }
            assert used - available == set()

    def test_every_dul_term_used_is_declared_by_dul(self, module_graphs, dul_graph):
        available = declared_terms(dul_graph)
        for graph in module_graphs.values():
            used = {
                term
                for term in referenced_terms(graph)
                if str(term).startswith(DUL_NAMESPACE)
            }
            assert used - available == set()


# %% reasoning


@pytest.mark.slow
class TestReasoning:
    """
    HermiT accepts the microworld together with CRAM-AICOR and DUL, draws the
    conclusions the links are made for, and rejects the contradictions the sources rule
    out.
    """

    def test_microworld_is_consistent_with_every_class_satisfiable(self, reasoner):
        world = reasoner()
        assert list(world.inconsistent_classes()) == []

    @pytest.mark.parametrize(
        "linked_class", [CRAM_AICOR.ForceDynamicSchema, DUL.Workflow]
    )
    def test_pouring_prototype_is_inferred_to_be(
        self, reasoner, owlready2, linked_class
    ):
        world = reasoner(ReasoningCase.POURING_PROTOTYPE)
        assert str(linked_class) in inferred_types(
            owlready2, world, CASE.pouring_structure
        )

    def test_event_classified_by_flow_onset_is_an_efflux_onset_event(
        self, reasoner, owlready2
    ):
        world = reasoner(ReasoningCase.FLOW_ONSET_CLASSIFICATION)
        assert str(CRAM_AICOR.EffluxOnsetEvent) in inferred_types(
            owlready2, world, CASE.observed_efflux
        )

    def test_object_bound_to_source_container_slot_is_a_container(
        self, reasoner, owlready2
    ):
        world = reasoner(ReasoningCase.SOURCE_CONTAINER_BINDING)
        assert str(CRAM_AICOR.Container) in inferred_types(
            owlready2, world, CASE.kitchen_jug
        )

    def test_grip_force_corridor_defined_by_a_structure_is_a_region(
        self, reasoner, owlready2
    ):
        world = reasoner(ReasoningCase.STRUCTURE_DEFINING_GRIP_FORCE_CORRIDOR)
        assert str(DUL.Region) in inferred_types(owlready2, world, CASE.corridor)

    def test_forces_unloaded_is_a_force_dynamic_event_type(self, reasoner, owlready2):
        world = reasoner(ReasoningCase.FORCES_UNLOADED)
        assert str(CRAM_AICOR.ForceDynamicEventType) in inferred_types(
            owlready2, world, CASE.forces_unloaded
        )

    def test_reach_target_predicted_by_vision_is_not_inferred_to_be_mechanical(
        self, reasoner, owlready2
    ):
        world = reasoner(ReasoningCase.VISUAL_PREDICTION_OF_REACH_TARGET)
        assert str(CORE.MechanicalEventType) not in inferred_types(
            owlready2, world, CASE.reach_done
        )

    def test_load_followed_by_grip_force_is_a_continuous_prediction(
        self, reasoner, owlready2
    ):
        world = reasoner(ReasoningCase.GRIP_FOLLOWING_LOAD_PREDICTION)
        assert str(CORE.ContinuousSignalPrediction) in inferred_types(
            owlready2, world, CASE.falling_load
        )

    @pytest.mark.parametrize(
        "case",
        [
            ReasoningCase.MISMATCH_CONFIRMING_SUBGOAL,
            ReasoningCase.TRANSITION_PERMITTED_BY_MISMATCH,
            ReasoningCase.PARAMETER_WITH_RANGE_AND_CATEGORY,
            ReasoningCase.PREDICTION_TYPED_AS_OBSERVATION,
            ReasoningCase.MASS_RANGE_AS_LIFTOFF_THRESHOLD,
            ReasoningCase.FRICTION_PREDICTED_FROM_VISION,
            ReasoningCase.PHASE_TYPED_AS_PREDICTION,
            ReasoningCase.LIFTOFF_DRIVEN_BY_GRIP_FORCE,
            ReasoningCase.SLIP_RESISTED_BY_WEIGHT,
            ReasoningCase.CRUSH_DRIVEN_BY_LOAD_FORCE,
            ReasoningCase.FLOW_ONSET_RESISTED_BY_FRICTION,
            ReasoningCase.TACTILE_PREDICTION_OF_REACH_TARGET,
        ],
    )
    def test_contradiction_is_inconsistent(self, reasoner, owlready2, case):
        with pytest.raises(owlready2.OwlReadyInconsistentOntologyError):
            reasoner(case)
