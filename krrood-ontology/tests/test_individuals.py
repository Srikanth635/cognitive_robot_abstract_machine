"""Verify A-Box preflight and real generated-model loading."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, XSD, BNode, Literal, URIRef

from krrood_ontology import (
    CompilationOptions,
    OntologyCompiler,
    OntologyDocument,
    OntologyInput,
)
from krrood_ontology.individuals import IndividualLoader, IndividualPlanner
from krrood_ontology.runtime import IndividualLoadingError


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = Path(__file__).parents[1]
    directory = tmp_path_factory.mktemp("phase4") / "generated"
    OntologyCompiler().compile(
        OntologyInput(root / "examples/relations.ttl"),
        CompilationOptions(directory, "phase4_test_model"),
    )
    sys.path.insert(0, str(directory))
    module = importlib.import_module("phase4_test_model")
    yield module.MODEL
    sys.path.remove(str(directory))


@pytest.fixture
def document():
    return OntologyDocument.load(
        OntologyInput(Path(__file__).parents[1] / "examples/individuals.ttl")
    )


@pytest.mark.parametrize(
    "case,code",
    [
        ("missing_type", "missing_type"),
        ("unknown_type", "unknown_type"),
        ("unrelated_types", "unrelated_types"),
        ("missing_target", "unresolved_reference"),
        ("wrong_range", "range_mismatch"),
        ("wrong_domain", "domain_mismatch"),
        ("wrong_datatype", "datatype_mismatch"),
        ("language_field", "datatype_mismatch"),
        ("resource_literal", "literal_required"),
        ("unknown_property", "unknown_property"),
        ("same_as", "identity_or_import_unsupported"),
        ("anonymous", "anonymous_individual"),
        ("punning", "punning_unsupported"),
        ("imports", "identity_or_import_unsupported"),
    ],
)
def test_invalid_assertions_fail_before_construction(model, document, case, code):
    graph = document.graph
    base = "https://example.org/relations#"
    alice, acme, name, works = [
        URIRef(base + item) for item in ("Alice", "Acme", "name", "worksFor")
    ]
    if case == "missing_type":
        graph.remove((alice, RDF.type, None))
    elif case == "unknown_type":
        graph.add((alice, RDF.type, URIRef(base + "Unknown")))
    elif case == "unrelated_types":
        graph.add((alice, RDF.type, URIRef(base + "Organization")))
    elif case == "missing_target":
        graph.set((alice, works, URIRef(base + "Absent")))
    elif case == "wrong_range":
        graph.set((alice, works, URIRef(base + "Office")))
    elif case == "wrong_domain":
        graph.add((acme, name, Literal("Acme")))
    elif case == "wrong_datatype":
        graph.set((alice, name, Literal(12)))
    elif case == "language_field":
        graph.set((alice, name, Literal("Alice", lang="en")))
    elif case == "resource_literal":
        graph.set((alice, name, acme))
    elif case == "unknown_property":
        graph.add((alice, URIRef(base + "unknown"), Literal("value")))
    elif case == "same_as":
        graph.add((alice, OWL.sameAs, acme))
    elif case == "anonymous":
        graph.add((BNode(), RDF.type, URIRef(base + "Person")))
    elif case == "punning":
        graph.add((URIRef(base + "Person"), RDF.type, OWL.NamedIndividual))
    elif case == "imports":
        graph.add((URIRef(base + "ontology"), OWL.imports, URIRef(base + "external")))
    with pytest.raises(IndividualLoadingError) as raised:
        IndividualPlanner(model).plan(document)
    assert raised.value.code == code


def test_annotations_retain_rdf_terms(model, document):
    _, _, annotations = IndividualPlanner(model).plan(document)
    label = annotations["https://example.org/relations#Alice"][str(RDFS.label)][0]
    assert label.language == "en"


def test_failed_load_never_constructs_objects(model, document, tmp_path, monkeypatch):
    base = "https://example.org/relations#"
    document.graph.add(
        (URIRef(base + "Alice"), URIRef(base + "unmapped"), Literal("bad"))
    )
    source = tmp_path / "invalid.ttl"
    document.graph.serialize(destination=source, format="turtle")
    calls = []

    def reject_construction(self):
        calls.append(self)
        raise AssertionError("Preflight must reject before construction")

    for concrete in model.classes.values():
        monkeypatch.setattr(concrete, "__init__", reject_construction)
    with pytest.raises(IndividualLoadingError) as raised:
        IndividualLoader(model).load(source)
    assert raised.value.code == "unknown_property"
    assert calls == []


def test_literal_conversion_and_invalid_lexical_value(model):
    from dataclasses import replace
    from datetime import date, datetime, time
    from decimal import Decimal

    planner = IndividualPlanner(model)
    mapping = model.properties["https://example.org/relations#name"]
    cases = [
        (XSD.string, "Alice", "Alice"),
        (XSD.boolean, "false", False),
        (XSD.integer, "42", 42),
        (XSD.decimal, "1.25", Decimal("1.25")),
        (XSD.double, "1.5", 1.5),
        (XSD.float, "1.5", 1.5),
        (XSD.date, "2026-09-30", date(2026, 9, 30)),
        (XSD.dateTime, "2026-09-30T12:00:00", datetime(2026, 9, 30, 12)),
        (XSD.time, "12:00:00", time(12)),
    ]
    for datatype, lexical, expected in cases:
        value = planner.literal(
            Literal(lexical, datatype=datatype),
            replace(mapping, range_iri=str(datatype)),
            "test",
        )
        assert value == expected
        assert type(value) is type(expected)
    with pytest.raises(IndividualLoadingError) as raised:
        planner.literal(
            Literal("bad", datatype=XSD.integer, normalize=False),
            replace(mapping, range_iri=str(XSD.integer)),
            "test",
        )
    assert raised.value.code == "invalid_literal"


def test_generated_individual_loading(tmp_path):
    root = Path(__file__).parents[1]
    paths = []
    for example, package in [
        ("university", "university_model"),
        ("relations", "relations_model"),
    ]:
        result = OntologyCompiler().compile(
            OntologyInput(root / f"examples/{example}.ttl"),
            CompilationOptions(tmp_path / example, package),
        )
        paths.append(str(result.output_directory))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        paths + [environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, str(root / "examples/verify_individuals.py")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(result.manifest_file.read_text())
    assert manifest["individual_loader_available"] is True
    assert manifest["individuals_loaded"] is False
