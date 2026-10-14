"""Load prepared RDF without reparsing or duplicating the native materializer."""

import importlib
import sys
from pathlib import Path

import pytest
from krrood.symbol_graph.symbol_graph import SymbolGraph
from rdflib import RDF, XSD, Graph, Literal, URIRef

from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput
from krrood_ontology.export import IndividualExporter
from krrood_ontology.individuals import IndividualLoader
from krrood_ontology.runtime import IndividualLoadingError


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = tmp_path_factory.mktemp("graph-loading") / "out"
    OntologyCompiler().compile(
        OntologyInput(Path(__file__).parent / "fixtures/construction.ttl"),
        CompilationOptions(
            root,
            "graph_loading_model",
            structural_only=True,
            binding_policy="local-usage",
            scoped_semantics=True,
        ),
    )
    sys.path.insert(0, str(root))
    module = importlib.import_module("graph_loading_model")
    yield module.MODEL
    sys.path.remove(str(root))


@pytest.fixture
def graph():
    result = Graph()
    result.add((URIRef("urn:a"), RDF.type, URIRef("urn:construction:Special")))
    result.add((URIRef("urn:b"), RDF.type, URIRef("urn:construction:Entity")))
    result.add((URIRef("urn:a"), URIRef("urn:construction:connected"), URIRef("urn:b")))
    return result


def test_graph_shared_native_references(model, graph):
    SymbolGraph.clear()  # This test owns its native graph.
    loaded = IndividualLoader(model).load_graph(graph)
    assert type(loaded["urn:a"]) is model.classes["urn:construction:Special"]
    assert next(iter(loaded["urn:a"].connected)) is loaded["urn:b"]


def test_graph_is_defensive_and_preserves_provenance(model, graph):
    SymbolGraph.clear()
    loaded = IndividualLoader(model).load_graph(graph)
    before = frozenset(graph)
    graph.remove((None, None, None))
    assert frozenset(loaded.asserted_graph) == before
    assert loaded.source_bytes and loaded.source_format == "nt"


def test_graph_preflight_failure_creates_no_symbols(model, graph):
    graph.add((URIRef("urn:c"), RDF.type, URIRef("urn:unknown")))
    native = SymbolGraph()
    before = tuple(native.wrapped_instances)
    with pytest.raises(IndividualLoadingError):
        IndividualLoader(model).load_graph(graph)
    assert tuple(native.wrapped_instances) == before


def test_literal_lexical_form_is_not_reparsed(model, graph):
    SymbolGraph.clear()
    literal = Literal("001.2300", datatype=XSD.decimal, normalize=False)
    graph.add((URIRef("urn:a"), URIRef("urn:construction:amount"), literal))
    loaded = IndividualLoader(model).load_graph(graph)
    assert (
        next(
            loaded.asserted_graph.objects(
                URIRef("urn:a"), URIRef("urn:construction:amount")
            )
        )
        == literal
    )
    assert frozenset(
        IndividualExporter(model).graph(loaded, mode="current")
    ) == frozenset(graph)


def test_explicit_export_skips_fields_outside_their_domain():
    root = Path(__file__).parents[1] / "examples/generated"
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("university_model")
        SymbolGraph.clear()
        loaded = IndividualLoader(module.MODEL).load(root / "source/closure.ttl")
        assert frozenset(
            IndividualExporter(module.MODEL).graph(loaded, mode="current")
        ) == frozenset(loaded.asserted_graph)
    finally:
        sys.path.remove(str(root))
