"""
Tests of the Python vocabulary generated from the predictive manipulation ontology: that
the committed modules are current, that they mirror the ontology's classes, value
partitions and properties, and that krrood's property descriptors infer over them.
"""

from __future__ import annotations

import inspect
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, RDFS, Graph, URIRef
from krrood.class_diagrams.class_diagram import WrappedClass
from typing_extensions import Any, List, Optional

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    gaze,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity, PropertyTerm
from llm_reasoner.knowledge.ontology_translation.reading import (
    ConflictingPythonNames,
    FieldTypeOutOfReach,
    MissingPropertyRange,
    OntologyReader,
)
from llm_reasoner.knowledge.ontology_translation.writing import VocabularyWriter

VOCABULARY_MODULES: dict[PredictiveManipulationOntologyFile, ModuleType] = {
    PredictiveManipulationOntologyFile.CORE: core,
    PredictiveManipulationOntologyFile.OBJECT_DYNAMICS: object_dynamics,
    PredictiveManipulationOntologyFile.PREHENSION: prehension,
    PredictiveManipulationOntologyFile.POURING: pouring,
    PredictiveManipulationOntologyFile.GAZE: gaze,
}
"""
The generated Python module of every ontology module.
"""

CASES_DIRECTORY = Path(__file__).parent / "ontology_translation_cases"
"""
The directory holding the ontology fragments the tests add to the microworld.
"""


class TranslationCase(StrEnum):
    """
    Ontology fragments added to the microworld to exercise the reader.
    """

    PROPERTY_NAMED_LIKE_A_CLASS = "property_named_like_a_class.ttl"
    """
    A property whose descriptor class would share its name with a class.
    """

    ENUMERATION_FROM_A_LATER_MODULE = "enumeration_from_a_later_module.ttl"
    """
    A property of a core class whose values come from a later module's enumeration.
    """

    PROPERTY_WITHOUT_RANGE_RESTRICTED_TO_MICROWORLD_CLASS = (
        "property_without_range_restricted_to_microworld_class.ttl"
    )
    """
    A property without a range whose restriction names a microworld class as filler.
    """

    @property
    def path(self) -> Path:
        """
        The location of the case's Turtle file.
        """
        return CASES_DIRECTORY / self.value


# %% fixtures


@pytest.fixture(scope="module")
def ontology() -> Graph:
    """
    Every predictive manipulation module in one graph.
    """
    graph = Graph()
    for module_file in PredictiveManipulationOntologyFile:
        graph.parse(module_file.path, format="turtle")
    return graph


@pytest.fixture(scope="module")
def writer() -> VocabularyWriter:
    """
    The writer that generates the vocabulary modules.
    """
    return VocabularyWriter.for_predictive_manipulation()


# %% helpers


def is_microworld_term(term: object) -> bool:
    """
    Return whether a node is a term the microworld itself declares.
    """
    return isinstance(term, URIRef) and str(term).startswith(
        PREDICTIVE_MANIPULATION_NAMESPACE_ROOT
    )


def value_partitions(graph: Graph) -> set[URIRef]:
    """
    Return every microworld class defined by enumerating its members.
    """
    return {
        cls
        for cls, equivalent in graph.subject_objects(OWL.equivalentClass)
        if is_microworld_term(cls) and graph.value(equivalent, OWL.oneOf) is not None
    }


def description_classes() -> dict[str, type[DescriptionEntity]]:
    """
    Return every generated description class by the IRI of its ontology class.
    """
    return {
        cls.term.iri: cls
        for module in VOCABULARY_MODULES.values()
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, DescriptionEntity)
        and cls is not DescriptionEntity
        and cls.__module__ == module.__name__
    }


def field_type(cls: type, name: str) -> Any:
    """
    Return the resolved type of a generated class's field, as krrood reads it.
    """
    (wrapped,) = [
        wrapped for wrapped in WrappedClass(cls).fields if wrapped.field.name == name
    ]
    return wrapped.resolved_type


def member_properties(graph: Graph) -> set[URIRef]:
    """
    Return every microworld property stated of the members of a value partition.
    """
    partitions = value_partitions(graph)
    restricted_on_partitions = {
        graph.value(restriction, OWL.onProperty)
        for partition in partitions
        for restriction in graph.objects(partition, RDFS.subClassOf)
    }
    return {
        prop
        for prop in restricted_on_partitions
        if graph.value(prop, RDFS.domain) is None
    }


def reader_with(case: TranslationCase) -> OntologyReader:
    """
    Return a reader over every module, with the case's statements added to the core.
    """
    module_graphs = {
        module_file: Graph().parse(module_file.path, format="turtle")
        for module_file in PredictiveManipulationOntologyFile
    }
    module_graphs[PredictiveManipulationOntologyFile.CORE].parse(
        case.path, format="turtle"
    )
    return OntologyReader(module_graphs)


def enumerations() -> list[type[StrEnum]]:
    """
    Return every generated value-partition enumeration.
    """
    return [
        cls
        for module in VOCABULARY_MODULES.values()
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, StrEnum) and cls.__module__ == module.__name__
    ]


# %% generated source


class TestGeneratedSource:
    """
    The committed vocabulary modules are exactly what the ontology generates.
    """

    @pytest.mark.parametrize("module_file", list(PredictiveManipulationOntologyFile))
    def test_committed_module_matches_regenerated_source(self, writer, module_file):
        assert module_file.vocabulary_path.read_text() == writer.source(module_file)


# %% classes and value partitions


class TestClasses:
    """
    Every ontology class and value partition has its Python counterpart.
    """

    def test_every_ontology_class_has_a_description_class(self, ontology):
        partitions = value_partitions(ontology)
        ontology_classes = {
            cls
            for cls in ontology.subjects(RDF.type, OWL.Class)
            if is_microworld_term(cls) and cls not in partitions
        }
        assert {URIRef(iri) for iri in description_classes()} == ontology_classes

    def test_subclass_relations_between_microworld_classes_are_kept(self, ontology):
        classes = description_classes()
        missing = {
            (subclass, superclass)
            for subclass, superclass in ontology.subject_objects(RDFS.subClassOf)
            if str(subclass) in classes
            and str(superclass) in classes
            and not issubclass(classes[str(subclass)], classes[str(superclass)])
        }
        assert missing == set()

    def test_links_to_cram_aicor_and_dul_are_recorded(self):
        assert set(core.ActionPhase.term.linked_terms) >= {
            CRAM_AICOR_NAMESPACE + "Phase",
            DUL_NAMESPACE + "Task",
        }

    def test_every_value_partition_member_is_an_enumeration_member(self, ontology):
        members = {
            member
            for partition in value_partitions(ontology)
            for equivalent in ontology.objects(partition, OWL.equivalentClass)
            for member in ontology.items(ontology.value(equivalent, OWL.oneOf))
        }
        values = {URIRef(member) for enum in enumerations() for member in enum}
        assert values == members


# %% properties


class TestProperties:
    """
    Every ontology property is a field, holding one value when it is functional.
    """

    def test_field_typed_by_an_earlier_superclass_records_its_range(self):
        term = PropertyTerm.of_field(core.ActionPhaseStructure, "defines_gaze_fixation")
        assert term.range == gaze.GazeFixation.term.iri

    def test_every_microworld_property_is_a_field(self, ontology):
        properties = {
            prop
            for property_type in (OWL.ObjectProperty, OWL.DatatypeProperty)
            for prop in ontology.subjects(RDF.type, property_type)
            if is_microworld_term(prop)
        }
        fields = {
            URIRef(term.iri)
            for cls in description_classes().values()
            for name in cls.__dataclass_fields__
            for term in [PropertyTerm.of_field(cls, name)]
            if term is not None
        }
        assert properties - member_properties(ontology) <= fields

    def test_functional_property_holds_one_value(self):
        assert field_type(core.ActionPhase, "aims_for") == Optional[core.Subgoal]

    def test_non_functional_property_holds_many_values(self):
        assert (
            field_type(core.ActionPhase, "has_active_parameter") == List[core.Parameter]
        )

    def test_property_of_partition_members_is_an_enumeration_property(self):
        assert (
            core.MeasurementUnit.KILOGRAM.measures_dimension
            == core.PhysicalDimension.MASS
        )

    @pytest.mark.parametrize(
        "unit, dimension",
        [
            ("MILLIMETRE", "LENGTH"),
            ("CENTIMETRE", "LENGTH"),
            ("GRAM", "MASS"),
            ("LITRE", "VOLUME"),
        ],
    )
    def test_unit_at_the_scale_of_manipulation_measures_its_dimension(
        self, unit, dimension
    ):
        assert (
            core.MeasurementUnit[unit].measures_dimension
            == core.PhysicalDimension[dimension]
        )

    def test_no_field_is_declared_only_on_a_defined_class(self):
        defined = {
            cls
            for cls in description_classes().values()
            if cls.term.definition is not None
        }
        declared_on_defined = {
            (cls.__name__, name)
            for cls in defined
            for name in vars(cls).get("__annotations__", {})
            if PropertyTerm.of_field(cls, name) is not None
        }
        assert declared_on_defined == set()

    def test_property_named_like_a_class_is_refused(self):
        reader = reader_with(TranslationCase.PROPERTY_NAMED_LIKE_A_CLASS)
        with pytest.raises(ConflictingPythonNames):
            reader.module_declaration(PredictiveManipulationOntologyFile.CORE)

    def test_property_whose_restriction_reveals_a_missing_range_is_refused(self):
        reader = reader_with(
            TranslationCase.PROPERTY_WITHOUT_RANGE_RESTRICTED_TO_MICROWORLD_CLASS
        )
        with pytest.raises(MissingPropertyRange):
            reader.module_declaration(PredictiveManipulationOntologyFile.CORE)

    def test_field_valued_by_a_later_modules_enumeration_is_refused(self):
        reader = reader_with(TranslationCase.ENUMERATION_FROM_A_LATER_MODULE)
        with pytest.raises(FieldTypeOutOfReach):
            reader.module_declaration(PredictiveManipulationOntologyFile.CORE)


# %% krrood descriptors


class TestPropertyDescriptors:
    """
    The generated descriptors let krrood infer what the ontology's sub-properties
    entail.
    """

    def test_sub_property_descriptor_specialises_its_super_property(self):
        assert issubclass(core.DefinesPhase, core.DefinesComponent)

    def test_symmetric_property_is_inferred_in_both_directions(self):
        first = object_dynamics.DynamicsPerturbation()
        second = object_dynamics.DynamicsPerturbation()
        first.matched_with = [second]
        assert first in second.matched_with

    def test_phase_defined_by_a_structure_is_one_of_its_components(self):
        phase = core.ActionPhase()
        structure = core.ActionPhaseStructure(defines_phase=[phase])
        assert phase in structure.defines_component
