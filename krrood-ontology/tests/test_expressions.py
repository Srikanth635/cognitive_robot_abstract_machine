"""Verify expression translation boundaries and generated native EQL behavior."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdflib import OWL, RDF, XSD, BNode, Literal, URIRef

from krrood_ontology import (
    CompilationOptions,
    OntologyCompiler,
    OntologyDocument,
    OntologyInput,
    OntologyInspector,
    OntologyTranslationError,
)
from krrood_ontology.translation import OntologyTranslator

ROOT = Path(__file__).parents[1]
BASE = "https://example.org/expressions#"


@pytest.fixture
def document():
    return OntologyDocument.load(OntologyInput(ROOT / "examples/expressions.ttl"))


def translate(document, structural_only=False):
    return OntologyTranslator().translate(
        document, OntologyInspector().inspect(document), structural_only
    )


def test_expression_metadata_and_structural_omission(document):
    model = translate(document)
    assert len(model.expressions) == 9
    structural = translate(document, True)
    assert structural.expressions == ()
    assert str(OWL.equivalentClass) in {item.iri for item in structural.diagnostics}


def test_reverse_equivalence_assertion_is_normalized(document):
    graph = document.graph
    named = URIRef(BASE + "WorkingPerson")
    expression = graph.value(named, OWL.equivalentClass)
    expected = translate(document).expressions
    graph.add((expression, OWL.equivalentClass, named))
    assert translate(document).expressions == expected


def test_reverse_equivalence_only_is_supported(document):
    graph = document.graph
    named = URIRef(BASE + "WorkingPerson")
    expression = graph.value(named, OWL.equivalentClass)
    expected = translate(document).expressions
    graph.remove((named, OWL.equivalentClass, expression))
    graph.add((expression, OWL.equivalentClass, named))
    assert translate(document).expressions == expected


@pytest.mark.parametrize(
    "case,code",
    [
        ("multiple_operator", "expression_shape"),
        ("list_cycle", "expression_list"),
        ("list_multiple_first", "expression_list"),
        ("anonymous_cycle", "expression_cycle"),
        ("unknown_class", "expression_class"),
        ("unknown_property", "expression_property"),
        ("datatype_some", "expression_datatype_existential"),
        ("wrong_literal", "expression_literal"),
        ("anonymous_value", "expression_value"),
        ("empty_enum", "expression_enumeration"),
        ("named_definition_cycle", "expression_definition_cycle"),
        ("unattached", "expression_unattached"),
        ("extra_predicate", "expression_extra_predicate"),
        ("named_equivalence", "expression_root"),
        ("multiple_definitions", "expression_multiple_definitions"),
        ("all_values", "semantic_feature_deferred"),
    ],
)
def test_unsupported_expression_mappings_fail(document, case, code):
    graph = document.graph
    working = graph.value(URIRef(BASE + "WorkingPerson"), OWL.equivalentClass)
    head = graph.value(working, OWL.intersectionOf)
    restriction = list(graph.items(head))[1]
    if case == "multiple_operator":
        graph.add((working, OWL.unionOf, head))
    elif case == "list_cycle":
        graph.set((head, RDF.rest, head))
    elif case == "list_multiple_first":
        graph.add((head, RDF.first, URIRef(BASE + "Company")))
    elif case == "anonymous_cycle":
        graph.set((restriction, OWL.someValuesFrom, working))
    elif case == "unknown_class":
        graph.set((restriction, OWL.someValuesFrom, URIRef(BASE + "Unknown")))
    elif case == "unknown_property":
        graph.set((restriction, OWL.onProperty, URIRef(BASE + "unknown")))
    elif case == "datatype_some":
        graph.set((restriction, OWL.onProperty, URIRef(BASE + "name")))
    elif case == "wrong_literal":
        root = graph.value(URIRef(BASE + "NamedAlice"), OWL.equivalentClass)
        item = list(graph.items(graph.value(root, OWL.intersectionOf)))[1]
        graph.set((item, OWL.hasValue, Literal(42, datatype=XSD.integer)))
    elif case == "anonymous_value":
        root = graph.value(URIRef(BASE + "AcmeWorker"), OWL.equivalentClass)
        item = list(graph.items(graph.value(root, OWL.intersectionOf)))[1]
        graph.set((item, OWL.hasValue, BNode()))
    elif case == "empty_enum":
        root = graph.value(URIRef(BASE + "SelectedPerson"), OWL.equivalentClass)
        graph.set((root, OWL.oneOf, RDF.nil))
    elif case == "named_definition_cycle":
        graph.set((head, RDF.first, URIRef(BASE + "WorkingPerson")))
    elif case == "unattached":
        graph.add((BNode(), OWL.oneOf, RDF.nil))
    elif case == "extra_predicate":
        graph.add((working, URIRef(BASE + "extra"), Literal("value")))
    elif case == "named_equivalence":
        graph.add(
            (URIRef(BASE + "Person"), OWL.equivalentClass, URIRef(BASE + "Company"))
        )
    elif case == "multiple_definitions":
        root, cell = BNode(), BNode()
        graph.add((URIRef(BASE + "WorkingPerson"), OWL.equivalentClass, root))
        graph.add((root, OWL.oneOf, cell))
        graph.add((cell, RDF.first, URIRef(BASE + "Alice")))
        graph.add((cell, RDF.rest, RDF.nil))
    elif case == "all_values":
        graph.add((restriction, OWL.allValuesFrom, URIRef(BASE + "Company")))
    with pytest.raises(OntologyTranslationError) as raised:
        translate(document)
    assert raised.value.diagnostics[0].code == code


def test_generated_expression_queries(tmp_path):
    result = OntologyCompiler().compile(
        OntologyInput(ROOT / "examples/expressions.ttl"),
        CompilationOptions(tmp_path / "output", "expressions_model"),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(result.output_directory)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples/verify_expressions.py")],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(result.manifest_file.read_text())
    assert manifest["class_expression_queries"] is True
    assert manifest["automatic_reclassification"] is False
    assert manifest["phase"] == 10
