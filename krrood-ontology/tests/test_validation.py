"""Run the generated validation API in isolation from KRROOD's cached global graph."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    path = tmp_path_factory.mktemp("phase8")
    return OntologyCompiler().compile(
        OntologyInput(Path(__file__).parent / "fixtures/validation.ttl"),
        CompilationOptions(
            path / "out",
            "validation_model",
            structural_only=True,
            binding_policy="local-usage",
        ),
    )


def run(generated, code):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(generated.output_directory)
    p = subprocess.run(
        [sys.executable, "-c", "from validation_model import *\n" + code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_incomplete_missing_witnesses_are_unknown(generated):
    run(
        generated,
        """
a=Child()
r=model_validator.validate({'urn:a':a})
checks=[c for c in r.checks if c.class_iri=='https://example.org/validation#Assembly']
assert len(checks)==4
assert all(c.status=='unknown' for c in checks),checks
assert r.valid is None
r=model_validator.validate({'urn:a':a}, complete=True, identity_policy='python')
assert len(r.violations)==3,r.to_dict()
assert r.valid is False
""",
    )


def test_fulfilled_inherited_constraints_and_literal_functionality(generated):
    run(
        generated,
        """
a=Child();b=Part();a.parts.append(b);a.tag.append('ready');a.primary.append(b)
r=model_validator.validate({'urn:a':a,'urn:b':b},complete=True,identity_policy='python')
assert r.valid is True,r.to_dict()
a.tag.append('different')
r=model_validator.validate({'urn:a':a,'urn:b':b},complete=True,identity_policy='python')
assert any(c.kind=='functional' for c in r.violations)
""",
    )


def test_qualified_cardinality_and_owl_identity_are_separate(generated):
    run(
        generated,
        """
a=Maximum();b=Part();c=Part();d=Other();a.parts.extend([b,c,d])
scope={'urn:a':a,'urn:b':b,'urn:c':c,'urn:d':d}
r=model_validator.validate(scope,complete=True)
assert any(c.kind=='max' and c.status=='unknown' for c in r.checks),r.to_dict()
r=model_validator.validate(scope,complete=True,identity_policy='python')
assert any(c.kind=='max' and c.status=='violation' and c.observed_count==2 for c in r.checks)
a.parts.clear();a.parts.extend([b,d])
r=model_validator.validate(scope,complete=True,identity_policy='python')
assert r.valid is True,r.to_dict()
""",
    )


def test_all_disjoint_and_out_of_scope_references(generated):
    run(
        generated,
        """
a=Assembly();b=Other();a.parts.append(b);a.tag.append('ready')
r=model_validator.validate({'urn:a':a},complete=True,identity_policy='python')
assert any(c.status=='unknown' and 'scope' in c.message for c in r.checks)
r=model_validator.validate({'urn:a':a,'urn:b':b},complete=True,identity_policy='python')
assert any(c.kind=='all' and c.status=='violation' for c in r.checks)
r=model_validator.validate({'urn:conflict':Conflict()},complete=True,identity_policy='python')
assert any(c.kind=='disjoint' for c in r.violations)
""",
    )


def test_bool_is_not_numeric_zero_and_union_fillers_work(generated):
    run(
        generated,
        """
a=Flag();a.flag.append(0)
r=model_validator.validate({'urn:a':a},complete=True,identity_policy='python')
assert r.valid is False,r.to_dict()
a.flag[:]=[False]
assert model_validator.validate({'urn:a':a},complete=True,identity_policy='python').valid is True
u=UnionOwner();p=Other();u.parts.append(p)
assert model_validator.validate({'urn:u':u,'urn:p':p},complete=True,identity_policy='python').valid is True
""",
    )


def test_minimum_distinctness_and_object_functionality(generated):
    run(
        generated,
        """
from rdflib import Graph,URIRef,OWL
s=Minimum();p=Part();q=Part();s.parts.extend([p,q])
scope={'urn:s':s,'urn:p':p,'urn:q':q}
r=model_validator.validate(scope,complete=True)
assert any(c.kind=='min' and c.status=='unknown' for c in r.checks)
g=Graph();g.add((URIRef('urn:p'),OWL.differentFrom,URIRef('urn:q')))
r=model_validator.validate(scope,complete=True,identity_graph=g)
assert all(c.status=='satisfied' for c in r.checks if c.kind=='min'),r.to_dict()
a=Assembly();a.primary.extend([p,q]);a.parts.append(p);a.tag.append('ready')
scope={'urn:a':a,'urn:p':p,'urn:q':q}
r=model_validator.validate(scope,complete=True)
assert any(c.kind=='functional' and c.property_iri.endswith('#primary') and c.status=='unknown' for c in r.checks)
r=model_validator.validate(scope,complete=True,identity_graph=g)
assert any(c.kind=='functional' and c.property_iri.endswith('#primary') for c in r.violations)
""",
    )


def test_named_values_and_sameas_are_not_silently_normalized(generated):
    run(
        generated,
        """
from rdflib import Graph,URIRef,OWL
b=Bound();p=Part();b.parts.append(p)
r=model_validator.validate({'urn:b':b},complete=True,identity_policy='python')
assert any(c.status=='unknown' for c in r.checks)
r=model_validator.validate({'urn:b':b,'https://example.org/validation#required':p},complete=True)
assert r.valid is True,r.to_dict()
g=Graph();g.add((URIRef('urn:b'),OWL.sameAs,URIRef('urn:p')))
r=model_validator.validate({'urn:b':b,'https://example.org/validation#required':p},identity_graph=g)
assert r.unsupported and r.valid is None
""",
    )


def test_zero_cardinality_false_existential_and_unsupported_reporting(tmp_path):
    source = tmp_path / "edge.ttl"
    source.write_text("""
@prefix : <https://example.org/edge#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
:Empty a owl:Class ; rdfs:subClassOf [a owl:Restriction; owl:onProperty :p; owl:qualifiedCardinality "0"^^xsd:nonNegativeInteger; owl:onClass :Empty] .
:Flag a owl:Class ; rdfs:subClassOf [a owl:Restriction; owl:onProperty :flag; owl:someValuesFrom xsd:boolean] .
:Deferred a owl:Class ; rdfs:subClassOf [owl:complementOf :Empty] .
:p a owl:ObjectProperty .
:flag a owl:DatatypeProperty .
""")
    generated = OntologyCompiler().compile(
        OntologyInput(source),
        CompilationOptions(
            tmp_path / "out",
            "validation_model",
            structural_only=True,
            binding_policy="local-usage",
        ),
    )
    run(
        generated,
        """
a=Empty()
assert model_validator.validate({'urn:a':a},complete=True).valid is True
f=Flag();f.flag.append(False)
assert model_validator.validate({'urn:f':f},complete=True).valid is True
d=Deferred();r=model_validator.validate({'urn:d':d},complete=True)
assert r.unsupported and r.valid is None
assert len(model_validator.coverage()['unsupported_constraints'])==1
""",
    )
