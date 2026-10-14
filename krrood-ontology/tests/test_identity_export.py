"""Generic identity evidence, loss checks and source/current round-trip contracts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase10")
    source = Path(__file__).parent / "fixtures/identity_roundtrip.ttl"
    return OntologyCompiler().compile(
        OntologyInput(source),
        CompilationOptions(
            root / "out",
            "roundtrip_model",
            structural_only=True,
            binding_policy="local-usage",
            scoped_semantics=True,
        ),
    )


def run(generated, code):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(generated.output_directory)
    source = str(Path(__file__).parent / "fixtures/identity_roundtrip.ttl")
    script = (
        'from roundtrip_model import *\nfrom pathlib import Path\nfrom rdflib import Graph,URIRef,Literal,OWL,RDF,XSD\nfrom rdflib.compare import isomorphic\nBASE="https://example.org/roundtrip#"\nSOURCE=Path('
        + repr(source)
        + ")\n"
        + code
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_distinctness_loading_without_unique_name_assumption(generated):
    run(
        generated,
        """
loaded=load_individuals(SOURCE)
assert len(loaded.by_iri)==3 and len(loaded.distinct_groups)==2
assert len({id(v) for v in loaded.by_iri.values()})==3
assert isinstance(loaded[BASE+'one'],Special)
assert model_queries.match(BASE+'Pick',loaded)==list(loaded.by_iri.values())
assert len(loaded[BASE+'one'].loose)==2
assert {type(v) for v in loaded[BASE+'one'].loose}=={bool,int}
assert loaded.annotations[BASE+'one'][BASE+'note'][0].language=='de'
assert loaded.identity_policy=='distinct'
""",
    )


def test_source_graph_and_retained_bytes_round_trip(generated):
    run(
        generated,
        """
from tempfile import TemporaryDirectory
loaded=load_individuals(SOURCE)
loaded[BASE+'one'].amount.clear()
assert isomorphic(model_exporter.graph(loaded),loaded.asserted_graph)
with TemporaryDirectory() as d:
    path=Path(d)/'snapshot.ttl'
    model_exporter.write(loaded,path)
    assert loaded.source_bytes==SOURCE.read_bytes()
    assert isomorphic(Graph().parse(path),loaded.asserted_graph)
    restored=load_individuals(path)
    assert restored[BASE+'one'].amount
    try: model_exporter.write(loaded,path)
    except FileExistsError: pass
    else: raise AssertionError('Existing export overwritten')
""",
    )


def test_current_projection_preserves_source_literals_and_omits_inference(generated):
    run(
        generated,
        """
loaded=load_individuals(SOURCE)
result=model_exporter.export(loaded,mode='current')
assert result.added_triples==0 and result.removed_triples==0
assert result.omitted_inferred_facts>0
assert isomorphic(result.graph,loaded.asserted_graph)
assert (URIRef(BASE+'one'),URIRef(BASE+'amount'),Literal('1.2300',datatype=XSD.decimal)) in result.graph
included=model_exporter.export(loaded,mode='current',include_inferred=True)
assert included.added_triples>0
assert (URIRef(BASE+'one'),URIRef(BASE+'links'),URIRef(BASE+'three')) in included.graph
""",
    )


def test_current_edits_export_reload_preserve_schema_identity_annotations(generated):
    run(
        generated,
        """
from decimal import Decimal
from tempfile import TemporaryDirectory
loaded=load_individuals(SOURCE)
loaded[BASE+'one'].amount[:]=[Decimal('2.50')]
with TemporaryDirectory() as d:
    path=Path(d)/'current.ttl'
    result=model_exporter.write(loaded,path,mode='current')
    assert result.added_triples==1 and result.removed_triples==1
    restored=load_individuals(path)
    assert restored[BASE+'one'].amount==[Decimal('2.50')]
    assert restored.distinct_groups==loaded.distinct_groups
    assert isomorphic(model_exporter.graph(restored,mode='current'),result.graph)
    assert (URIRef(BASE+'Special'),RDF.type,OWL.Class) in result.graph
    assert restored.annotations==loaded.annotations
""",
    )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("same_as", "identity_or_import_unsupported"),
        ("self_different", "identity_conflict"),
        ("duplicate", "identity_conflict"),
        ("missing", "unresolved_identity"),
        ("cycle", "expression_list"),
    ],
)
def test_bad_identity_fails_before_construction(generated, case, expected):
    run(
        generated,
        """
from krrood_ontology.individuals import IndividualPlanner
from krrood_ontology.runtime import IndividualLoadingError
from krrood_ontology.ontology import OntologyDocument
from krrood_ontology.config import OntologyInput
from rdflib import BNode
from tempfile import TemporaryDirectory
from unittest.mock import patch
case="""
        + repr(case)
        + """
graph=Graph().parse(SOURCE)
head=next(graph.objects(next(graph.subjects(RDF.type,OWL.AllDifferent)),OWL.distinctMembers))
if case=='same_as': graph.add((URIRef(BASE+'one'),OWL.sameAs,URIRef(BASE+'two')))
if case=='self_different': graph.add((URIRef(BASE+'one'),OWL.differentFrom,URIRef(BASE+'one')))
if case=='duplicate': graph.set((head,RDF.first,URIRef(BASE+'two')))
if case=='missing': graph.set((head,RDF.first,URIRef(BASE+'absent')))
if case=='cycle': graph.set((head,RDF.rest,head))
with TemporaryDirectory() as d:
    path=Path(d)/'bad.ttl';graph.serialize(destination=path,format='turtle')
    with patch.object(Special,'__init__',side_effect=AssertionError('Constructed before preflight')):
        try: load_individuals(path)
        except IndividualLoadingError as error:
            assert error.code=="""
        + repr(expected)
        + """,error
        else: raise AssertionError('Invalid identity accepted')
""",
    )


def test_current_invalid_reference_does_not_create_file(generated):
    run(
        generated,
        """
from tempfile import TemporaryDirectory
from krrood_ontology.export import IndividualExportError
loaded=load_individuals(SOURCE)
loaded[BASE+'one'].links.add(Entity())
with TemporaryDirectory() as d:
    path=Path(d)/'bad.ttl'
    try: model_exporter.write(loaded,path,mode='current')
    except IndividualExportError: pass
    else: raise AssertionError('Unregistered reference exported')
    assert not path.exists()
""",
    )


def test_relocation_preserves_relative_iri_meaning(generated):
    run(
        generated,
        """
from tempfile import TemporaryDirectory
with TemporaryDirectory() as d:
    root=Path(d); source=root/'input'/'data.ttl';source.parent.mkdir()
    source.write_text('@prefix : <https://example.org/roundtrip#> . <person> a :Entity; :links <other> . <other> a :Entity .')
    loaded=load_individuals(source)
    destination=root/'output'/'copy.ttl';destination.parent.mkdir()
    model_exporter.write(loaded,destination)
    restored=load_individuals(destination)
    assert set(restored.by_iri)==set(loaded.by_iri)
    assert isomorphic(restored.asserted_graph,loaded.asserted_graph)
""",
    )


def test_current_rejects_silent_registry_growth(generated):
    run(
        generated,
        """
from krrood_ontology.export import IndividualExportError
loaded=load_individuals(SOURCE)
loaded.by_iri['urn:new']=Entity()
try: model_exporter.export(loaded,mode='current')
except IndividualExportError: pass
else: raise AssertionError('Registry growth silently discarded')
""",
    )
