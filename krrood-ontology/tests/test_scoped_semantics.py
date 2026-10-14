"""Verify native implications and positive definitions independently of cached graphs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    return OntologyCompiler().compile(
        OntologyInput(Path(__file__).parent / "fixtures/scoped_semantics.ttl"),
        CompilationOptions(
            tmp_path_factory.mktemp("phase9") / "out",
            "scoped_model",
            structural_only=True,
            binding_policy="local-usage",
            scoped_semantics=True,
        ),
    )


def run(generated, code):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(generated.output_directory)
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            'from scoped_model import *\nBASE="https://example.org/scoped#"\n' + code,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_union_alias_enum_and_nested_queries(generated):
    run(
        generated,
        """
a=Alpha();b=Beta();e=Carrier();e.link.append(a)
scope={BASE+'one':a,BASE+'two':b,'urn:e':e}
assert model_queries.match(BASE+'Either',scope)==[a,b]
assert model_queries.match(BASE+'Alias',scope)==[a]
assert model_queries.match(BASE+'Linked',scope)==[e]
assert model_queries.match(BASE+'Choice',{BASE+'one':a})==[a]
assert type(e) is Carrier and not isinstance(e,Linked)
x=MarkedAlias();x.flag.append(False)
y=MarkedAlias();y.flag.append(0)
assert model_queries.match(BASE+'Marked',{'urn:x':x,'urn:y':y})==[x,y]
# Explicit alias membership is positive evidence even when necessary values conflict.
r=model_validator.validate({'urn:x':x,'urn:y':y},complete=True,identity_policy='python')
assert any(c.subject_iri=='urn:y' and c.kind=='has_value' for c in r.violations)
""",
    )


def test_native_cross_type_closure_and_reflexivity(generated):
    run(
        generated,
        """
a=Alpha();b=Beta();c=Entity();scope={'urn:a':a,'urn:b':b,'urn:c':c}
seed=model_relations.initialize_reflexivity(scope)
assert seed.added==3
assert all(v in v.connected for v in scope.values())
a.connected.add(b);b.connected.add(c)
assert c in a.connected and a in c.connected
facts=model_relations.snapshot(scope)
assert any(f.subject_iri=='urn:a' and f.object_iri=='urn:c' and f.inferred for f in facts)
assert any(f.subject_iri=='urn:a' and f.object_iri=='urn:b' and not f.inferred for f in facts)
assert model_relations.initialize_reflexivity(scope).added==0
a.connected.add(c)
assert any(f.subject_iri=='urn:a' and f.object_iri=='urn:c' and f.inferred for f in model_relations.snapshot(scope))
""",
    )


def test_asymmetry_conflicts_and_scope(generated):
    run(
        generated,
        """
a=Alpha();b=Beta();a.before.add(b)
r=model_validator.validate({'urn:a':a},complete=True,identity_policy='python')
assert any(c.kind=='asymmetric' and c.status=='unknown' for c in r.checks)
b.before.add(a)
r=model_validator.validate({'urn:a':a,'urn:b':b},complete=True,identity_policy='python')
assert any(c.kind=='asymmetric' for c in r.violations)
""",
    )


def test_positive_witness_and_alias_parent(generated):
    run(
        generated,
        """
x=FlagCarrier();x.flag.append(False)
y=FlagCarrier();y.flag.append(0)
assert model_queries.match(BASE+'Marked',{'urn:x':x,'urn:y':y})==[x]
z=Alias()
assert model_queries.match(BASE+'Entity',{'urn:z':z})==[z]
""",
    )


def test_recursive_definitions_rejected_before_publication(tmp_path):
    from krrood_ontology.model import OntologyTranslationError

    source = tmp_path / "cycle.ttl"
    source.write_text("""@prefix : <urn:cycle:> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
:A a owl:Class; owl:equivalentClass [a owl:Restriction; owl:onProperty :p; owl:someValuesFrom :A] .
:p a owl:ObjectProperty .""")
    output = tmp_path / "output"
    with pytest.raises(OntologyTranslationError, match="definition_cycle"):
        OntologyCompiler().compile(
            OntologyInput(source),
            CompilationOptions(
                output,
                "cycle_model",
                structural_only=True,
                binding_policy="local-usage",
                scoped_semantics=True,
            ),
        )
    assert not output.exists()


@pytest.fixture(scope="module")
def signatures(tmp_path_factory):
    root = tmp_path_factory.mktemp("signatures9")
    source = root / "signatures.ttl"
    source.write_text("""@prefix : <urn:signature:> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:A a owl:Class . :B a owl:Class . :C a owl:Class .
:p a owl:ObjectProperty; rdfs:domain :A; rdfs:range :B; owl:inverseOf :q .
:q a owl:ObjectProperty; rdfs:domain :B; rdfs:range :A .
:child a owl:ObjectProperty; rdfs:subPropertyOf :parent .
:parent a owl:ObjectProperty .
:self a owl:ObjectProperty, owl:ReflexiveProperty; rdfs:domain :A; rdfs:range :A .""")
    return OntologyCompiler().compile(
        OntologyInput(source),
        CompilationOptions(
            root / "out",
            "scoped_model",
            structural_only=True,
            binding_policy="local-usage",
            scoped_semantics=True,
        ),
    )


def test_native_inverse_superproperty_and_asserted_signatures(signatures):
    run(
        signatures,
        """
a=A();b=B();c=C();a.p.add(b)
assert a in b.q
a.child.add(c)
assert c in a.parent
assert MODEL.properties['urn:signature:p'].domain_iri=='urn:signature:A'
assert MODEL.properties['urn:signature:p'].range_iri=='urn:signature:B'
seed=model_relations.initialize_reflexivity({'urn:a':a,'urn:b':b,'urn:c':c})
assert seed.added==1 and len(seed.skipped)==2
c.p.add(a)
r=model_validator.validate({'urn:a':a,'urn:b':b,'urn:c':c},complete=True,identity_policy='python')
assert any(v.kind=='domain' and v.subject_iri=='urn:c' for v in r.violations)
assert any(v.kind=='range' and v.subject_iri=='urn:c' for v in r.violations)
""",
    )


def test_loader_still_checks_asserted_domain(signatures, tmp_path):
    run(
        signatures,
        """
from krrood_ontology.runtime import IndividualLoadingError
from pathlib import Path
from tempfile import TemporaryDirectory
with TemporaryDirectory() as d:
    source=Path(d)/'abox.ttl'
    source.write_text('@prefix : <urn:signature:> . <urn:c> a :C; :p <urn:b> . <urn:b> a :B .')
    try:
        load_individuals(source)
    except IndividualLoadingError as error:
        assert error.code=='domain_mismatch'
    else:
        raise AssertionError('Declared domain was replaced by operational storage signature')
""",
    )


def test_nested_restrictions_reverse_definitions_inherited_alias_fields(generated):
    run(
        generated,
        """
a=Alpha();inner=Carrier();outer=Carrier()
inner.link.append(a);outer.link.append(inner)
scope={'urn:a':a,'urn:inner':inner,'urn:outer':outer}
assert model_queries.match(BASE+'Nested',scope)==[outer]
x=FlagCarrier();x.flag.append(False)
assert model_queries.match(BASE+'ReverseMarked',{'urn:x':x})==[x]
y=InheritedFlagAlias();y.flag.append(False)
assert model_queries.match(BASE+'Marked',{'urn:y':y})==[y]
""",
    )


def test_missing_import_rejected_in_scoped_profile(tmp_path):
    from krrood_ontology.imports import ImportResolutionError

    source = tmp_path / "import.ttl"
    source.write_text("""@prefix owl: <http://www.w3.org/2002/07/owl#> .
<urn:o> a owl:Ontology; owl:imports <urn:missing> . <urn:C> a owl:Class .""")
    output = tmp_path / "output"
    with pytest.raises(ImportResolutionError, match="No supplied local file"):
        OntologyCompiler().compile(
            OntologyInput(source),
            CompilationOptions(
                output,
                "import_model",
                structural_only=True,
                binding_policy="local-usage",
                scoped_semantics=True,
            ),
        )
    assert not output.exists()
