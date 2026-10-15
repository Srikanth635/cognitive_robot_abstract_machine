"""
Tests of writing the ontology of a code base: which classes it describes, what it says
of each class and field, and how it writes them as OWL.
"""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic
from typing_extensions import Dict

from llm_reasoner.world.class_ontology.__main__ import main
from llm_reasoner.world.class_ontology.field_types import FieldTypeReader, PlainType
from llm_reasoner.world.class_ontology.reading import (
    ClassDescription,
    ClassOntology,
    ClassOntologyReader,
    FieldDescription,
)
from llm_reasoner.world.class_ontology.selection import (
    ClassSelection,
    SelectionKey,
    UnknownClass,
)
from llm_reasoner.world.class_ontology.writing import (
    AnnotationName,
    ClassOntologyWriter,
)
from llm_reasoner.world.classes import DocumentedClass, identifier_of

from .class_ontology_mimics import data, designators, geometry, kinds

MIMICS = f"{__package__}.class_ontology_mimics"
"""
The package of the classes the ontology is written of.
"""

NAMESPACE = "http://www.example.org/ont/mimics#"
"""
The namespace the ontology of the mimics is written in.
"""

CASES = Path(__file__).parent / "class_ontology_cases"
"""
Where the files the tests read are kept.
"""

# %% helpers


def mimic_selection() -> ClassSelection:
    """
    :return: The designators and the kinds of the mimics, with the data structures
        described in full and the bookkeeping of designators left out.
    """
    return ClassSelection(
        namespace=NAMESPACE,
        packages=(MIMICS,),
        roots=(
            identifier_of(designators.ActionDescription),
            identifier_of(kinds.SemanticAnnotation),
        ),
        expanded_packages=(data.__name__,),
        ignored_bases=(identifier_of(designators.Designator),),
    )


@pytest.fixture(scope="module")
def ontology() -> ClassOntology:
    """
    :return: The ontology of the mimics.
    """
    return ClassOntologyReader(selection=mimic_selection()).read()


def described(ontology: ClassOntology, described_class: type) -> ClassDescription:
    return next(
        description
        for description in ontology.classes
        if description.identifier == identifier_of(described_class)
    )


def fields_of(
    ontology: ClassOntology, described_class: type
) -> Dict[str, FieldDescription]:
    return {
        description.name: description
        for description in described(ontology, described_class).fields
    }


def iri(described_class: type) -> URIRef:
    return URIRef(NAMESPACE + identifier_of(described_class))


def field_iri(described_class: type, name: str) -> URIRef:
    return URIRef(NAMESPACE + identifier_of(described_class) + "." + name)


# %% what a class says of itself in full


class TestFullDocumentation:
    def test_a_class_is_documented_by_its_whole_docstring(self):
        assert DocumentedClass.of(designators.PourAction).documentation == (
            inspect.cleandoc(designators.PourAction.__doc__)
        )

    def test_a_dataclass_without_a_docstring_has_no_documentation(self):
        assert DocumentedClass.of(geometry.Pose).documentation == ""

    def test_a_field_means_every_paragraph_of_its_docstring(self):
        documentation = DocumentedClass.of(designators.PourAction).meanings
        assert documentation["tilt"] == (
            "How far the vessel is tilted, in radians.\n\nMeasured from upright."
        )

    def test_a_field_a_parent_declares_keeps_the_parents_documentation(self):
        documentation = DocumentedClass.of(designators.PourAction).meanings
        assert documentation["speed_limit"].startswith(
            "The fastest the effector may move"
        )


# %% what is selected


class TestSelection:
    def test_a_selection_is_read_from_its_file(self):
        written = json.loads((CASES / "selection.json").read_text())
        selection = ClassSelection.from_json(written)
        assert selection.namespace == written[SelectionKey.NAMESPACE]
        assert selection.packages == tuple(written[SelectionKey.PACKAGES])
        assert selection.roots == tuple(written[SelectionKey.ROOTS])
        assert selection.expanded_packages == tuple(
            written[SelectionKey.EXPANDED_PACKAGES]
        )
        assert selection.ignored_bases == tuple(written[SelectionKey.IGNORED_BASES])

    def test_a_root_the_package_does_not_have_is_refused_with_its_name(self):
        missing = f"{designators.__name__}.FlyAction"
        selection = ClassSelection(
            namespace=NAMESPACE,
            packages=(MIMICS,),
            roots=(missing,),
            expanded_packages=(),
            ignored_bases=(),
        )
        with pytest.raises(UnknownClass) as refused:
            ClassOntologyReader(selection=selection).read()
        assert refused.value.identifier == missing

    def test_a_module_that_cannot_be_imported_is_skipped_and_reported(self, ontology):
        (skipped,) = ontology.skipped_modules
        assert skipped.module == f"{MIMICS}.broken"
        assert skipped.error_type == ImportError.__name__


# %% which classes are described


class TestDescribedClasses:
    def test_the_roots_and_their_subclasses_in_the_packages_are_described(
        self, ontology
    ):
        identifiers = {description.identifier for description in ontology.classes}
        assert {
            identifier_of(selected)
            for selected in (
                designators.ActionDescription,
                designators.PourAction,
                designators.NavigateAction,
                designators.InspectAction,
                kinds.SemanticAnnotation,
                kinds.Container,
                kinds.Vessel,
            )
        } <= identifiers

    def test_a_private_class_is_not_described(self, ontology):
        identifiers = {description.identifier for description in ontology.classes}
        assert identifier_of(designators._DraftAction) not in identifiers

    def test_a_base_whose_fields_are_ignored_is_not_described(self, ontology):
        identifiers = {description.identifier for description in ontology.classes}
        assert identifier_of(designators.Designator) not in identifiers

    def test_a_class_a_field_refers_to_is_described_without_its_fields(self, ontology):
        pose = described(ontology, geometry.Pose)
        assert pose.fields == ()

    def test_a_class_of_an_expanded_package_is_described_with_its_fields(
        self, ontology
    ):
        assert set(fields_of(ontology, data.Grasp)) == {"approach", "offset"}

    def test_an_enumeration_is_described_with_its_members(self, ontology):
        (side,) = [
            enumeration
            for enumeration in ontology.enumerations
            if enumeration.identifier == identifier_of(data.Side)
        ]
        assert side.members == tuple(member.name for member in data.Side)

    def test_an_enumeration_a_described_class_refers_to_is_described_too(
        self, ontology
    ):
        identifiers = {enumeration.identifier for enumeration in ontology.enumerations}
        assert identifier_of(data.Approach) in identifiers


# %% what is said of a class


class TestClassDescription:
    def test_a_class_is_named_and_documented(self, ontology):
        pour = described(ontology, designators.PourAction)
        assert pour.name == designators.PourAction.__name__
        assert (
            pour.documentation
            == DocumentedClass.of(designators.PourAction).documentation
        )

    def test_a_class_specialises_the_nearest_described_classes_above_it(self, ontology):
        assert described(ontology, designators.PourAction).parents == (
            identifier_of(designators.ActionDescription),
        )
        assert described(ontology, kinds.Vessel).parents == (
            identifier_of(kinds.Container),
        )

    def test_a_class_above_which_nothing_is_described_specialises_nothing(
        self, ontology
    ):
        assert described(ontology, designators.ActionDescription).parents == ()

    def test_a_field_of_a_mixin_is_described_on_the_class_using_it(self, ontology):
        assert "speed_limit" in fields_of(ontology, designators.PourAction)
        assert "speed_limit" not in fields_of(ontology, designators.NavigateAction)

    def test_a_field_a_described_parent_declares_is_not_repeated(self, ontology):
        assert "rim_height" in fields_of(ontology, kinds.Container)
        assert "rim_height" not in fields_of(ontology, kinds.Vessel)

    def test_a_field_whose_type_a_subclass_narrows_is_restated_on_the_subclass(
        self, ontology
    ):
        place = fields_of(ontology, designators.PantryStoreAction)["place"]
        assert place.field_type.target == identifier_of(kinds.Vessel)
        assert fields_of(ontology, designators.StoreAction)[
            "place"
        ].field_type.target == (identifier_of(kinds.Container))

    def test_the_fields_of_an_ignored_base_are_left_out(self, ontology):
        for description in ontology.classes:
            assert "plan_node" not in {field.name for field in description.fields}

    def test_a_private_field_is_left_out(self, ontology):
        assert "_attempts" not in fields_of(ontology, designators.PourAction)


# %% what is said of a field


class TestFieldDescription:
    def test_a_field_is_documented_in_full(self, ontology):
        tilt = fields_of(ontology, designators.PourAction)["tilt"]
        assert tilt.documentation == (
            DocumentedClass.of(designators.PourAction).meanings["tilt"]
        )

    def test_a_field_holding_an_individual_refers_to_its_class(self, ontology):
        source = fields_of(ontology, designators.PourAction)["source"]
        assert source.field_type.target == identifier_of(kinds.Vessel)
        assert source.field_type.plain is None

    def test_a_field_holding_a_member_refers_to_its_enumeration(self, ontology):
        side = fields_of(ontology, designators.PourAction)["side"]
        assert side.field_type.target == identifier_of(data.Side)

    def test_a_field_holding_a_number_is_a_plain_value_with_its_default(self, ontology):
        tilt = fields_of(ontology, designators.PourAction)["tilt"]
        assert tilt.field_type.plain is PlainType.FLOAT
        assert tilt.field_type.target is None
        assert tilt.default == repr(designators.PourAction.tilt)

    def test_an_optional_field_says_so(self, ontology):
        speed_limit = fields_of(ontology, designators.PourAction)["speed_limit"]
        assert speed_limit.field_type.optional
        assert speed_limit.field_type.plain is PlainType.FLOAT
        assert not fields_of(ontology, designators.PourAction)[
            "tilt"
        ].field_type.optional

    def test_a_field_holding_several_refers_to_the_class_of_each(self, ontology):
        waypoints = fields_of(ontology, designators.PourAction)["waypoints"]
        assert waypoints.field_type.collection
        assert waypoints.field_type.target == identifier_of(geometry.Pose)
        assert waypoints.default is None

    def test_a_type_that_cannot_be_resolved_is_kept_as_written(self, ontology):
        clue = fields_of(ontology, designators.InspectAction)["clue"]
        assert clue.field_type.target is None
        assert clue.field_type.plain is None
        assert "Mystery" in clue.field_type.written

    def test_a_field_of_a_bounded_type_variable_refers_to_its_bound(self, ontology):
        place = fields_of(ontology, designators.StoreAction)["place"]
        assert place.field_type.target == identifier_of(kinds.Container)

    def test_a_type_variable_a_subclass_fixes_is_read_as_what_it_fixes(self, ontology):
        field_type = FieldTypeReader(described=designators.PantryStoreAction).read(
            next(
                declared
                for declared in dataclass_fields(designators.PantryStoreAction)
                if declared.name == "place"
            )
        )
        assert field_type.target == identifier_of(kinds.Vessel)

    def test_a_type_variable_a_subclass_passes_on_is_read_as_its_bound(self):
        field_type = FieldTypeReader(described=designators.CarefulStoreAction).read(
            next(
                declared
                for declared in dataclass_fields(designators.CarefulStoreAction)
                if declared.name == "place"
            )
        )
        assert field_type.target == identifier_of(kinds.Container)

    def test_a_variadic_type_variable_fixed_to_one_type_is_read_as_that_type(
        self, ontology
    ):
        sensors = fields_of(ontology, kinds.WatchingThing)["sensors"]
        assert sensors.field_type.target == identifier_of(kinds.Camera)
        assert sensors.field_type.collection

    def test_a_variadic_type_variable_passed_on_is_left_unresolved(self):
        field_type = FieldTypeReader(described=kinds.MovingSensingThing).read(
            next(
                declared
                for declared in dataclass_fields(kinds.MovingSensingThing)
                if declared.name == "sensors"
            )
        )
        assert field_type.target is None
        assert field_type.collection

    def test_the_type_as_written_survives_code_that_overwrites_a_fields_type(
        self, monkeypatch
    ):
        speed_limit = next(
            declared
            for declared in dataclass_fields(designators.PourAction)
            if declared.name == "speed_limit"
        )
        monkeypatch.setattr(speed_limit, "type", typing.Optional[float])
        field_type = FieldTypeReader(described=designators.PourAction).read(speed_limit)
        assert field_type.written == (
            designators.HasSpeedLimit.__annotations__["speed_limit"]
        )

    def test_the_other_fields_of_that_class_are_still_resolved(self, ontology):
        target_location = fields_of(ontology, designators.InspectAction)[
            "target_location"
        ]
        assert target_location.field_type.target == identifier_of(geometry.Pose)


# %% writing it as OWL


@pytest.fixture(scope="module")
def graph(ontology) -> Graph:
    """
    :return: The ontology of the mimics as OWL.
    """
    return ClassOntologyWriter(ontology=ontology, namespace=NAMESPACE).graph()


ANNOTATIONS = Namespace(NAMESPACE)
"""
The annotation properties the ontology writes, in its own namespace.
"""


class TestWriting:
    def test_a_class_is_written_with_its_name_documentation_and_parents(self, graph):
        pour = iri(designators.PourAction)
        assert (pour, RDF.type, OWL.Class) in graph
        assert graph.value(pour, RDFS.label) == Literal(designators.PourAction.__name__)
        assert graph.value(pour, RDFS.comment) == Literal(
            DocumentedClass.of(designators.PourAction).documentation
        )
        assert (pour, RDFS.subClassOf, iri(designators.ActionDescription)) in graph
        assert graph.value(pour, ANNOTATIONS[AnnotationName.PYTHON_NAME]) == Literal(
            identifier_of(designators.PourAction)
        )

    def test_a_field_holding_an_individual_is_an_object_property(self, graph):
        source = field_iri(designators.PourAction, "source")
        assert (source, RDF.type, OWL.ObjectProperty) in graph
        assert (source, RDF.type, OWL.FunctionalProperty) in graph
        assert graph.value(source, RDFS.domain) == iri(designators.PourAction)
        assert graph.value(source, RDFS.range) == iri(kinds.Vessel)
        assert graph.value(source, RDFS.label) == Literal("source")

    def test_a_field_holding_a_number_is_a_datatype_property_with_its_default(
        self, graph
    ):
        tilt = field_iri(designators.PourAction, "tilt")
        assert (tilt, RDF.type, OWL.DatatypeProperty) in graph
        assert graph.value(tilt, RDFS.range) == XSD.double
        assert graph.value(tilt, ANNOTATIONS[AnnotationName.DEFAULT]) == Literal(
            repr(designators.PourAction.tilt)
        )

    def test_a_field_holding_several_is_not_functional(self, graph):
        waypoints = field_iri(designators.PourAction, "waypoints")
        assert (waypoints, RDF.type, OWL.FunctionalProperty) not in graph
        assert graph.value(waypoints, RDFS.range) == iri(geometry.Pose)

    def test_an_optional_field_is_marked_so(self, graph):
        speed_limit = field_iri(designators.PourAction, "speed_limit")
        assert graph.value(speed_limit, ANNOTATIONS[AnnotationName.OPTIONAL]) == (
            Literal(True)
        )

    def test_a_field_whose_type_is_unknown_keeps_its_type_as_written(self, graph):
        clue = field_iri(designators.InspectAction, "clue")
        assert graph.value(clue, RDFS.range) is None
        assert "Mystery" in str(
            graph.value(clue, ANNOTATIONS[AnnotationName.PYTHON_TYPE])
        )

    def test_an_enumeration_lists_its_members_as_individuals(self, graph):
        right = URIRef(
            NAMESPACE + identifier_of(data.Side) + "." + data.Side.RIGHT.name
        )
        assert (right, RDF.type, iri(data.Side)) in graph
        assert (right, RDF.type, OWL.NamedIndividual) in graph
        assert graph.value(right, RDFS.label) == Literal(data.Side.RIGHT.name)

    def test_the_ontology_names_the_modules_it_skipped(self, graph, ontology):
        header = URIRef(NAMESPACE.rstrip("#"))
        assert (header, RDF.type, OWL.Ontology) in graph
        skipped = {
            str(value)
            for value in graph.objects(
                header, ANNOTATIONS[AnnotationName.SKIPPED_MODULE]
            )
        }
        assert skipped == {
            f"{module.module}: {module.error_type}"
            for module in ontology.skipped_modules
        }

    def test_the_turtle_reads_back_as_the_same_graph(self, ontology, graph):
        written = ClassOntologyWriter(ontology=ontology, namespace=NAMESPACE).turtle()
        assert isomorphic(Graph().parse(data=written, format="turtle"), graph)

    def test_writing_twice_gives_the_same_text(self, ontology):
        writer = ClassOntologyWriter(ontology=ontology, namespace=NAMESPACE)
        assert writer.turtle() == writer.turtle()


# %% the command


class TestCommand:
    def test_the_command_writes_the_ontology_a_selection_file_asks_for(
        self, tmp_path, graph
    ):
        selection = mimic_selection()
        selection_file = tmp_path / "selection.json"
        selection_file.write_text(
            json.dumps(
                {
                    SelectionKey.NAMESPACE: selection.namespace,
                    SelectionKey.PACKAGES: list(selection.packages),
                    SelectionKey.ROOTS: list(selection.roots),
                    SelectionKey.EXPANDED_PACKAGES: list(selection.expanded_packages),
                    SelectionKey.IGNORED_BASES: list(selection.ignored_bases),
                }
            )
        )
        output = tmp_path / "ontology.ttl"
        main([str(selection_file), str(output)])
        assert isomorphic(Graph().parse(output, format="turtle"), graph)
