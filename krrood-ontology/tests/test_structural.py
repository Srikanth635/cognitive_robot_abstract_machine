"""Verify generic Phase 7 projection, native associations and faithful omissions."""

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
)
from krrood_ontology.model import OntologyTranslationError
from krrood_ontology.structural import StructuralTranslator

BASE = "https://example.org/structural#"
FIXTURE = Path(__file__).parent / "fixtures/local_usage.ttl"


def plan(document):
    return StructuralTranslator().translate(
        document, OntologyInspector().inspect(document)
    )


def test_local_projection_preserves_unknown_signatures_and_parents():
    model = plan(OntologyDocument.load(OntologyInput(FIXTURE)))
    classes = {x.iri: x for x in model.classes}
    combined = classes[BASE + "Combined"]
    assert set(combined.asserted_parent_iris) == {
        BASE + "Left",
        BASE + "Right",
        BASE + "Entity",
    }
    assert set(combined.parent_names) == {"Left", "Right"}
    assert classes[BASE + "None"].name == "None_"
    assert classes[BASE + "Type"].name == "Type_"
    props = {x.iri: x for x in model.properties}
    assert (
        props[BASE + "connects"].domain_iri == props[BASE + "connects"].range_iri == ""
    )
    assert props[BASE + "tag"].range_type == "_Any"
    assert len([b for b in model.property_bindings if b.iri == BASE + "connects"]) == 2
    assert not any(b.iri == BASE + "unused" for b in model.property_bindings)
    assert len(model.schema_axioms) == 3
    assert all(x["predicate_iri"] == str(RDFS.subClassOf) for x in model.schema_axioms)


def test_native_multiple_bindings_and_individual_loader(tmp_path):
    result = OntologyCompiler().compile(
        OntologyInput(FIXTURE),
        CompilationOptions(
            tmp_path / "out",
            "local_model",
            structural_only=True,
            binding_policy="local-usage",
        ),
    )
    script = """
from local_model import *
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
from dataclasses import fields
loaded=load_individuals(SOURCE)
a=loaded[BASE+'a']; b=loaded[BASE+'b']
assert isinstance(a, Left) and isinstance(a, Right) and isinstance(a, Entity)
assert isinstance(a, Symbol)
assert a.connects[0] is b
assert a.tag==['ready']
assert type(Left.connects) is type(Right.connects)
assert isinstance(Left.connects,PropertyDescriptor)
assert len(type(Left.connects).domain_range_map[type(Left.connects)])==2
assert not hasattr(Entity(), 'unused')
assert set(MODEL.properties)=={BASE+'connects',BASE+'tag',BASE+'unused'}
assert SCHEMA_AXIOMS and not CLASS_EXPRESSIONS
assert MODEL.mapping_for(BASE+'connects',BASE+'Entity') is None
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(result.output_directory)
    r = subprocess.run(
        [sys.executable, "-c", f"BASE={BASE!r}\nSOURCE={str(FIXTURE)!r}\n" + script],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    manifest = json.loads(result.manifest_file.read_text())
    assert manifest["binding_policy"] == "local-usage"
    assert manifest["inference_applied"] is False


def test_generic_namespace_and_missing_binding_are_not_domain_inference(tmp_path):
    renamed = tmp_path / "other.ttl"
    renamed.write_text(
        FIXTURE.read_text().replace(BASE, "urn:another:").replace(":None ", ":Choice ")
    )
    m = plan(OntologyDocument.load(OntologyInput(renamed)))
    assert {c.iri for c in m.classes} == {
        "urn:another:" + n
        for n in ("Entity", "Left", "Right", "Combined", "Choice", "Type")
    }
    assert all("structural#" not in p.iri for p in m.properties)


def test_cycles_and_collisions_fail_before_output(tmp_path):
    doc = OntologyDocument.load(OntologyInput(FIXTURE))
    doc.graph.add((URIRef(BASE + "Entity"), RDFS.subClassOf, URIRef(BASE + "Combined")))
    with pytest.raises(OntologyTranslationError, match="inheritance_cycle"):
        plan(doc)
    doc = OntologyDocument.load(OntologyInput(FIXTURE))
    doc.graph.add((URIRef("https://other.example/#Left"), RDF.type, OWL.Class))
    with pytest.raises(OntologyTranslationError, match="class_collision"):
        plan(doc)


def test_binding_policy_cannot_silently_enable_semantics(tmp_path):
    with pytest.raises(ValueError, match="structural_only"):
        CompilationOptions(tmp_path, "model", binding_policy="local-usage")
