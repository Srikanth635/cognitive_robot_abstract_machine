"""Verify semantic planning and the generated KRROOD runtime behavior."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, URIRef

from krrood_ontology import (
    CompilationOptions,
    OntologyCompiler,
    OntologyDocument,
    OntologyInput,
    OntologyInspector,
    OntologyTranslationError,
)
from krrood_ontology.translation import OntologyTranslator


@pytest.fixture
def document() -> OntologyDocument:
    return OntologyDocument.load(
        OntologyInput(Path(__file__).parents[1] / "examples" / "relations.ttl")
    )


def translate(document: OntologyDocument, structural_only: bool = False):
    return OntologyTranslator().translate(
        document, OntologyInspector().inspect(document), structural_only
    )


def test_inverse_declaration_expands_in_both_directions(
    document: OntologyDocument,
) -> None:
    model = translate(document)
    descriptors = {item.property.name: item for item in model.descriptors}
    assert (
        descriptors["works_for"].inverse_iri == descriptors["has_employee"].property.iri
    )
    assert (
        descriptors["has_employee"].inverse_iri == descriptors["works_for"].property.iri
    )
    assert descriptors["knows"].inverse_name == descriptors["knows"].name
    assert descriptors["knows"].symmetric is True
    assert descriptors["contains_part"].transitive is True
    assert descriptors["heads_of"].parent_name == descriptors["employee_of"].name
    assert descriptors["works_for"].property.container == "set"
    assert descriptors["works_for"].property.constructor is False


def test_structural_only_disables_relation_axioms(document: OntologyDocument) -> None:
    model = translate(document, structural_only=True)
    assert model.descriptors == ()
    assert all(
        item.container == "list" and item.constructor for item in model.properties
    )
    assert str(OWL.inverseOf) in {item.iri for item in model.diagnostics}
    assert str(RDFS.subPropertyOf) in {item.iri for item in model.diagnostics}


@pytest.mark.parametrize(
    "case,code",
    [
        ("wrong_inverse", "inverse_signature"),
        ("unknown_inverse", "inverse_target"),
        ("multiple_inverses", "multiple_inverses"),
        ("subproperty_cycle", "subproperty_cycle"),
        ("multiple_superproperties", "multiple_superproperties"),
        ("transitive_ancestor", "transitive_superproperty"),
        ("datatype_characteristic", "semantic_property_kind"),
        ("transitive_signature", "semantic_signature"),
        ("subproperty_signature", "subproperty_signature"),
    ],
)
def test_unsupported_semantic_mappings_fail(
    document: OntologyDocument, case: str, code: str
) -> None:
    graph = document.graph
    base = "https://example.org/relations#"
    works, inverse, child, knows, part = [
        URIRef(base + name)
        for name in ("worksFor", "hasEmployee", "employeeOf", "knows", "containsPart")
    ]
    if case == "wrong_inverse":
        graph.set((inverse, RDFS.range, URIRef(base + "Company")))
    elif case == "unknown_inverse":
        graph.set((works, OWL.inverseOf, URIRef(base + "Unknown")))
    elif case == "multiple_inverses":
        graph.add((works, OWL.inverseOf, works))
        graph.set((works, RDFS.range, URIRef(base + "Person")))
        graph.set((inverse, RDFS.domain, URIRef(base + "Person")))
        graph.set((child, RDFS.range, URIRef(base + "Person")))
        graph.set((URIRef(base + "headsOf"), RDFS.range, URIRef(base + "Person")))
    elif case == "subproperty_cycle":
        graph.add((works, RDFS.subPropertyOf, child))
    elif case == "multiple_superproperties":
        graph.add((URIRef(base + "headsOf"), RDFS.subPropertyOf, works))
    elif case == "transitive_ancestor":
        graph.add((knows, RDF.type, OWL.TransitiveProperty))
    elif case == "datatype_characteristic":
        graph.add((URIRef(base + "name"), RDF.type, OWL.TransitiveProperty))
    elif case == "transitive_signature":
        graph.set((part, RDFS.range, URIRef(base + "Company")))
    elif case == "subproperty_signature":
        graph.set((child, RDFS.range, URIRef(base + "Organization")))
    with pytest.raises(OntologyTranslationError) as raised:
        translate(document)
    assert raised.value.diagnostics[0].code == code


def test_generated_relation_semantics(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    result = OntologyCompiler().compile(
        OntologyInput(root / "examples" / "relations.ttl"),
        CompilationOptions(tmp_path / "output", "relations_model"),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(result.output_directory) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "verify_relations.py")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["phase"] == 10
    assert manifest["relation_implications"] is True
    assert manifest["inference_applied"] is False
    assert manifest["retraction_supported"] is False
