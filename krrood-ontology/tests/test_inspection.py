"""Exercise real RDF parsing and distinguish inventory from inference."""

from pathlib import Path

import pytest
from rdflib import OWL, RDF, URIRef

from krrood_ontology.config import OntologyInput, RDFFormat
from krrood_ontology.inspection import OntologyInspector
from krrood_ontology.ontology import OntologyDocument


@pytest.fixture
def university() -> OntologyDocument:
    path = Path(__file__).parents[1] / "examples" / "university.ttl"
    return OntologyDocument.load(OntologyInput(path))


def test_explicit_vocabulary_and_individuals(university: OntologyDocument) -> None:
    report = OntologyInspector().inspect(university)
    base = "https://example.org/university#"
    assert report.classes == tuple(
        base + name for name in ("Course", "Person", "Student")
    )
    assert report.object_properties == (base + "takesCourse",)
    assert report.datatype_properties == (base + "name",)
    assert report.individuals == (base + "Alice", base + "Course42")
    assert report.rdf_format == RDFFormat.TURTLE.value


def test_imports_and_restrictions_remain_assertions(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "restrictions.ttl"
    document = OntologyDocument.load(OntologyInput(source))
    report = OntologyInspector().inspect(document)
    assert report.imports == ("https://example.invalid/external.owl",)
    assert str(OWL.Restriction) in report.owl_features
    assert str(OWL.someValuesFrom) in report.owl_features
    assert str(OWL.equivalentClass) in report.owl_features
    assert len(report.classes) == 2
    assert (
        URIRef("https://example.org/test#Alice"),
        RDF.type,
        URIRef("https://example.org/test#Parent"),
    ) not in document.graph


def test_xml_owl_extension_uses_rdflib(
    university: OntologyDocument, tmp_path: Path
) -> None:
    destination = tmp_path / "university.owl"
    university.graph.serialize(destination=destination, format="xml")
    loaded = OntologyDocument.load(OntologyInput(destination))
    assert loaded.format is RDFFormat.XML
    assert (
        OntologyInspector().inspect(loaded).classes
        == OntologyInspector().inspect(university).classes
    )


def test_extension_override(university: OntologyDocument, tmp_path: Path) -> None:
    destination = tmp_path / "ontology.data"
    destination.write_bytes(university.source.path.read_bytes())
    with pytest.raises(ValueError):
        OntologyDocument.load(OntologyInput(destination))
    loaded = OntologyDocument.load(OntologyInput(destination, RDFFormat.TURTLE))
    assert len(loaded.graph) == len(university.graph)
