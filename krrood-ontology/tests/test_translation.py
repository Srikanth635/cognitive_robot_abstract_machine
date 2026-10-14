"""Check structural mappings and failures before source generation."""

from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, XSD, URIRef

from krrood_ontology import OntologyDocument, OntologyInput, OntologyInspector
from krrood_ontology.model import OntologyTranslationError, PropertyKind
from krrood_ontology.translation import OntologyTranslator


@pytest.fixture
def document() -> OntologyDocument:
    return OntologyDocument.load(
        OntologyInput(Path(__file__).parents[1] / "examples" / "university.ttl")
    )


def translate(document: OntologyDocument, structural_only: bool = False):
    return OntologyTranslator().translate(
        document, OntologyInspector().inspect(document), structural_only
    )


def test_inheritance_and_property_types(document: OntologyDocument) -> None:
    model = translate(document)
    classes = {item.name: item for item in model.classes}
    assert classes["Student"].parent_name == "Person"
    assert [item.name for item in model.classes].index("Person") < [
        item.name for item in model.classes
    ].index("Student")
    assert classes["Student"].properties[0].name == "takes_course"
    assert classes["Student"].properties[0].range_type == "Course"
    assert classes["Student"].properties[0].kind is PropertyKind.OBJECT
    assert classes["Person"].properties[0].range_type == "str"
    assert model.diagnostics[0].code == "individuals_deferred"


def test_semantic_features_require_explicit_opt_in() -> None:
    path = Path(__file__).parent / "fixtures" / "restrictions.ttl"
    document = OntologyDocument.load(OntologyInput(path))
    with pytest.raises(OntologyTranslationError) as raised:
        translate(document)
    assert str(OWL.someValuesFrom) in {item.iri for item in raised.value.diagnostics}
    model = translate(document, structural_only=True)
    assert str(OWL.imports) in {item.iri for item in model.diagnostics}
    assert "Parent" in {item.name for item in model.classes}


@pytest.mark.parametrize(
    "case,code",
    [
        ("cycle", "inheritance_cycle"),
        ("multiple_domains", "property_signature"),
        ("missing_range", "property_signature"),
        ("unknown_parent", "unsupported_parent"),
        ("class_collision", "class_collision"),
        ("inherited_field_collision", "field_collision"),
        ("multiple_parents", "multiple_parents"),
        ("datatype_range", "datatype_range"),
        ("invalid_identifier", "invalid_identifier"),
        ("property_kind_conflict", "property_kind_conflict"),
        ("undeclared_property", "undeclared_property"),
        ("undeclared_class", "undeclared_class"),
    ],
)
def test_invalid_mappings_fail(
    document: OntologyDocument, case: str, code: str
) -> None:
    graph = document.graph
    base = "https://example.org/university#"
    person, student, course, relation = [
        URIRef(base + name) for name in ("Person", "Student", "Course", "takesCourse")
    ]
    if case == "cycle":
        graph.add((person, RDFS.subClassOf, student))
    elif case == "multiple_domains":
        graph.add((relation, RDFS.domain, person))
    elif case == "missing_range":
        graph.remove((relation, RDFS.range, None))
    elif case == "unknown_parent":
        graph.set((student, RDFS.subClassOf, URIRef(base + "Unknown")))
    elif case == "class_collision":
        graph.add((URIRef("https://another.example/#Person"), RDF.type, OWL.Class))
    elif case == "inherited_field_collision":
        second = URIRef("https://another.example/#name")
        graph.add((second, RDF.type, OWL.ObjectProperty))
        graph.add((second, RDFS.domain, student))
        graph.add((second, RDFS.range, course))
    elif case == "multiple_parents":
        graph.add((student, RDFS.subClassOf, course))
    elif case == "datatype_range":
        graph.set((URIRef(base + "name"), RDFS.range, XSD.positiveInteger))
    elif case == "invalid_identifier":
        graph.add((URIRef(base + "%40bad"), RDF.type, OWL.Class))
    elif case == "property_kind_conflict":
        graph.add((relation, RDF.type, OWL.DatatypeProperty))
    elif case == "undeclared_property":
        graph.remove((relation, RDF.type, OWL.ObjectProperty))
    elif case == "undeclared_class":
        graph.remove((student, RDF.type, OWL.Class))
    with pytest.raises(OntologyTranslationError) as raised:
        translate(document)
    assert raised.value.diagnostics[0].code == code


def test_krrood_case_conversion_preserves_iri_mapping(
    document: OntologyDocument,
) -> None:
    iri = "https://example.org/university#bad-name"
    document.graph.add((URIRef(iri), RDF.type, OWL.Class))
    model = translate(document)
    concept = next(item for item in model.classes if item.iri == iri)
    assert concept.name == "BadName"
