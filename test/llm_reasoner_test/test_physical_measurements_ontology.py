"""
Tests of the physical measurement modules: the structure of the flow onset module, the
vocabulary generated from it, that no generated knowledge reaches it, the closed-world
rules a measurement meets, and what an OWL reasoner concludes from it.

Tests that need CRAM-AICOR or DUL read the pinned copies
(:class:`~llm_reasoner.knowledge.ontologies.linked.LinkedOntologyFile`), and the test that
reasons needs ``owlready2`` and a Java runtime and is slow.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from krrood.entity_query_language.factories import the
from rdflib import OWL, RDFS, Graph, Namespace, URIRef

from llm_reasoner.knowledge.acceptance.derived import (
    RangesHold,
    RestrictionsHold,
    RestrictionViolation,
)
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Violation
from llm_reasoner.knowledge.acceptance.shapes import ShapesHold, ShapeViolation
from llm_reasoner.knowledge.generation.exceptions import MeasurementIsNotGenerated
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.generation.schema import TypeScope
from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
    PhysicalMeasurementFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import pouring_frame
from llm_reasoner.knowledge.ontologies.physical_measurements import flow_onset
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core, pouring
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity
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

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The flow onset module.
    """
    return Graph().parse(PhysicalMeasurementFile.FLOW_ONSET.path, format="turtle")


@pytest.fixture
def measurement() -> flow_onset.FlowOnsetTiltMeasurement:
    """
    A flow onset derived from the shape of a cup half full of a liquid.
    """
    return flow_onset.FlowOnsetTiltMeasurement(
        measured_for_container=flow_onset.ContainerModel(
            mesh_digest="0" * 64, mesh_file_name="cup.stl"
        ),
        at_fill_fraction=0.5,
        has_material_regime=pouring_frame.MaterialRegime.LIQUID,
        gravity_azimuth=90.0,
        has_measured_range=core.MagnitudeRange(
            lower_bound=48.0, upper_bound=48.0, has_unit=core.MeasurementUnit.DEGREE
        ),
        follows_setup=flow_onset.GeometricDerivationSetup(voxel_pitch=0.001),
    )


def violations_of(measurement: flow_onset.FlowOnsetTiltMeasurement) -> list[Violation]:
    """
    :return: What the measurement, and every fragment it reaches, fails of the
        restrictions, ranges and shapes their classes state.
    """
    return AcceptanceSuite(
        rules=[RestrictionsHold(), RangesHold(), ShapesHold()]
    ).violations(ShapesHold.reachable([measurement]))


SHAPES = Namespace(PhysicalMeasurementFile.FLOW_ONSET.shapes_namespace)
"""
The shapes of the flow onset module.
"""


# %% the module file


class TestModuleFile:
    def test_the_measurements_come_after_every_module_they_use(self):
        position = KNOWLEDGE_ONTOLOGY_MODULES.index
        assert position(PhysicalMeasurementFile.FLOW_ONSET) > max(
            position(PredictiveManipulationOntologyFile.CORE),
            position(ActionPerspectiveFile.ACTION_CORE),
        )

    def test_it_declares_one_ontology_importing_the_modules_it_uses(self, graph):
        iri = ontology_iri(graph)
        assert str(iri) == PhysicalMeasurementFile.FLOW_ONSET.namespace.rstrip("#")
        assert set(graph.objects(iri, OWL.imports)) == {
            URIRef(PredictiveManipulationOntologyFile.CORE.namespace.rstrip("#")),
            URIRef(ActionPerspectiveFile.POURING_FRAME.namespace.rstrip("#")),
        }

    def test_every_term_is_declared_in_its_namespace(self, graph):
        assert all(
            str(term).startswith(PhysicalMeasurementFile.FLOW_ONSET.namespace)
            for term in declared_terms(graph)
            if term != ontology_iri(graph)
        )

    def test_every_term_has_a_label_and_a_definition(self, graph):
        for term in declared_terms(graph):
            assert graph.value(term, RDFS.label) is not None, term
            assert graph.value(term, RDFS.comment) is not None, term

    def test_a_measurement_names_the_quantity_it_gives_a_value_for(self, graph):
        quantity_of = URIRef(
            PhysicalMeasurementFile.FLOW_ONSET.namespace + "measuresQuantityOf"
        )
        measured = URIRef(flow_onset.FlowOnsetTiltMeasurement.term.iri)
        assert graph.value(measured, quantity_of) == URIRef(
            pouring.FlowOnsetThreshold.term.iri
        )


# %% the generated vocabulary


class TestVocabulary:
    def test_the_committed_module_matches_the_regenerated_source(self):
        module_file = PhysicalMeasurementFile.FLOW_ONSET
        assert module_file.vocabulary_path.read_text() == (
            VocabularyWriter.for_knowledge_modules().source(module_file)
        )

    def test_each_kind_of_setup_is_a_measurement_setup(self):
        assert issubclass(
            flow_onset.GeometricDerivationSetup, flow_onset.MeasurementSetup
        )
        assert issubclass(
            flow_onset.PhysicsSimulationSetup, flow_onset.MeasurementSetup
        )


# %% generation never reaches it


class TestNeverGenerated:
    def test_its_modules_hold_no_generated_knowledge(self):
        assert not any(
            module_file.holds_generated_knowledge
            for module_file in PhysicalMeasurementFile
        )

    def test_no_module_of_generated_knowledge_may_use_it(self):
        for module_file in KNOWLEDGE_ONTOLOGY_MODULES:
            if module_file.holds_generated_knowledge:
                assert not set(PhysicalMeasurementFile) & module_file.usable_modules

    @pytest.mark.parametrize(
        "root", [pouring.PouringPhaseStructure, pouring_frame.PouringFrame]
    )
    def test_a_hypothesis_is_never_offered_a_measurement_class(self, root):
        offered = TypeScope.of_root(root).generated_types(DescriptionEntity)
        assert root in offered
        assert [
            offered_type
            for offered_type in offered
            if offered_type.__module__ == flow_onset.__name__
        ] == []

    def test_a_request_for_a_measurement_is_refused(self):
        with pytest.raises(MeasurementIsNotGenerated):
            HypothesisRequest.from_match(
                the(flow_onset.FlowOnsetTiltMeasurement)(at_fill_fraction=0.5),
                instruction="",
            )


# %% the closed-world rules


class TestRules:
    def test_a_complete_measurement_breaks_no_rule(self, measurement):
        assert violations_of(measurement) == []

    def test_a_measurement_without_its_value_breaks_a_restriction(self, measurement):
        measurement.has_measured_range = None
        (violation,) = violations_of(measurement)
        assert isinstance(violation, RestrictionViolation)
        assert violation.fragment is measurement

    def test_a_tilt_in_a_length_unit_breaks_a_restriction(self, measurement):
        measurement.has_measured_range = replace(
            measurement.has_measured_range, has_unit=core.MeasurementUnit.MILLIMETRE
        )
        assert any(
            isinstance(violation, RestrictionViolation)
            and violation.fragment is measurement
            for violation in violations_of(measurement)
        )

    @pytest.mark.parametrize("fill", [-0.1, 1.2])
    def test_a_fill_outside_the_container_breaks_its_shape(self, measurement, fill):
        measurement.at_fill_fraction = fill
        (violation,) = violations_of(measurement)
        assert isinstance(violation, ShapeViolation)
        assert violation.shape == SHAPES.FillFractionWithinTheContainer

    @pytest.mark.parametrize("azimuth", [-1.0, 360.0])
    def test_an_azimuth_beyond_one_turn_breaks_its_shape(self, measurement, azimuth):
        measurement.gravity_azimuth = azimuth
        (violation,) = violations_of(measurement)
        assert isinstance(violation, ShapeViolation)
        assert violation.shape == SHAPES.AzimuthWithinOneTurn

    def test_a_tilt_beyond_half_a_turn_breaks_its_shape(self, measurement):
        measurement.has_measured_range = replace(
            measurement.has_measured_range, upper_bound=190.0
        )
        (violation,) = violations_of(measurement)
        assert isinstance(violation, ShapeViolation)
        assert violation.shape == SHAPES.OnsetWithinHalfATurn


# %% links to CRAM-AICOR and DUL


class TestExternalLinks:
    def test_only_llm_reasoner_cram_aicor_and_dul_terms_are_used(self, graph):
        outside = {
            term
            for term in referenced_terms(graph)
            if not str(term).startswith(
                (
                    PhysicalMeasurementFile.FLOW_ONSET.namespace_root,
                    PredictiveManipulationOntologyFile.CORE.namespace_root,
                    ActionPerspectiveFile.ACTION_CORE.namespace_root,
                    CRAM_AICOR_NAMESPACE,
                    DUL_NAMESPACE,
                )
            )
        }
        assert outside == set()

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
    def test_it_is_consistent_with_every_module_cram_aicor_and_dul(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        for module_file in KNOWLEDGE_ONTOLOGY_MODULES:
            merged.parse(module_file.path, format="turtle")
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        measurement_classes = [
            cls
            for cls in world.classes()
            if cls.iri.startswith(PhysicalMeasurementFile.FLOW_ONSET.namespace)
        ]
        assert measurement_classes
        assert [
            cls for cls in measurement_classes if owlready2.Nothing in cls.equivalent_to
        ] == []
