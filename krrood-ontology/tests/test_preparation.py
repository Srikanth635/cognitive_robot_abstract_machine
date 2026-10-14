"""Generic candidate RDF preflight against an actual generated model."""

import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rdflib import OWL, RDF, XSD, BNode, Graph, URIRef

from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput
from krrood_ontology.contracts import (
    CheckStatus,
    ContractError,
    CreateEntity,
    EntityReference,
    FeedbackCategory,
    LiteralAssertion,
    LiteralValue,
    ObjectAssertion,
    Proposal,
)
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.sessions import AcceptanceBlockedError, OntologySystem

NS = "https://example.org/university#"


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    parent = tmp_path_factory.mktemp("phase12")
    root = parent / "generated"
    source = parent / "schema.ttl"
    source.write_text(
        (Path(__file__).parents[1] / "examples/university.ttl").read_text()
        + "\n<https://example.org/university#age> a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:integer .\n"
    )
    OntologyCompiler().compile(
        OntologyInput(source),
        CompilationOptions(root, "phase12_test_model"),
    )
    sys.path.insert(0, str(root))
    module = importlib.import_module("phase12_test_model")
    yield root, module.MODEL
    sys.path.remove(str(root))


@pytest.fixture
def preparer(compiled):
    root, model = compiled
    return CandidatePreparer.from_manifest(
        model,
        root / "manifest.json",
        schema_id="urn:test:schema",
        base_revision="accepted-0",
    )


def create(local="student", kind="Student", iri=None):
    return CreateEntity("create-" + local, local, (NS + kind,), iri)


def ref(local="student"):
    return EntityReference(local_id=local)


def prepare(preparer, operations):
    session = OntologySystem(preparer.context, checker=preparer).begin_candidate(
        Proposal(tuple(operations))
    )
    return session, preparer.prepare(session.snapshot)


def test_candidate_preserves_baseline_and_forward_references(preparer):
    session, result = prepare(
        preparer,
        (
            ObjectAssertion("link", ref(), NS + "takesCourse", ref("course")),
            create(),
            create("course", "Course"),
            LiteralAssertion("name", ref(), NS + "name", LiteralValue("Ada")),
        ),
    )
    assert result.report.results[0].status is CheckStatus.PASS
    candidate = result.candidate
    s, c = candidate.iri_for("student"), candidate.iri_for("course")
    assert (URIRef(s), URIRef(NS + "takesCourse"), URIRef(c)) in candidate.graph()
    assert len(candidate.additions) == 4
    assert (URIRef(NS + "Alice"), RDF.type, URIRef(NS + "Student")) in candidate.graph()
    candidate.require_current(session.snapshot)
    session.check()
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()
    assert all(r.status is CheckStatus.UNSUPPORTED for r in result.report.results[1:])


def test_existing_reference_and_duplicate_assertion_are_additive(preparer):
    _, result = prepare(
        preparer,
        (
            create(),
            ObjectAssertion(
                "link", ref(), NS + "takesCourse", EntityReference(iri=NS + "Course42")
            ),
            LiteralAssertion(
                "same-name",
                EntityReference(iri=NS + "Alice"),
                NS + "name",
                LiteralValue("Alice"),
            ),
        ),
    )
    assert result.report.results[0].status is CheckStatus.PASS
    assert len(result.candidate.additions) == 2


@pytest.mark.parametrize(
    "operations,code",
    [
        ((create(kind="Unknown"),), "unknown_type"),
        ((create(iri=NS + "Alice"),), "iri_collision"),
        ((create(iri=NS + "Student"),), "iri_collision"),
        ((create(iri="urn:same"), create("other", iri="urn:same")), "iri_collision"),
        (
            (LiteralAssertion("name", ref(), NS + "name", LiteralValue("Ada")),),
            "unresolved_reference",
        ),
        (
            (
                create(),
                ObjectAssertion(
                    "link", ref(), NS + "takesCourse", EntityReference(iri="urn:absent")
                ),
            ),
            "unresolved_reference",
        ),
        (
            (
                create(),
                ObjectAssertion(
                    "link",
                    ref(),
                    NS + "takesCourse",
                    EntityReference(iri=NS + "Course"),
                ),
            ),
            "unresolved_reference",
        ),
        (
            (
                create(),
                LiteralAssertion("name", ref(), NS + "unknown", LiteralValue("Ada")),
            ),
            "unknown_property",
        ),
        (
            (
                create(),
                LiteralAssertion(
                    "name", ref(), NS + "takesCourse", LiteralValue("Ada")
                ),
            ),
            "property_kind",
        ),
        (
            (create(), ObjectAssertion("name", ref(), NS + "name", ref())),
            "property_kind",
        ),
        (
            (create(), ObjectAssertion("link", ref(), NS + "takesCourse", ref())),
            "range_mismatch",
        ),
        (
            (
                create("course", "Course"),
                LiteralAssertion(
                    "name", ref("course"), NS + "name", LiteralValue("Ada")
                ),
            ),
            "domain_mismatch",
        ),
        (
            (
                create(),
                LiteralAssertion(
                    "name", ref(), NS + "name", LiteralValue("2", str(XSD.integer))
                ),
            ),
            "datatype_mismatch",
        ),
        (
            (
                create(),
                LiteralAssertion(
                    "name", ref(), NS + "name", LiteralValue("Ada", language="en")
                ),
            ),
            "datatype_mismatch",
        ),
    ],
)
def test_invalid_request_feedback_without_artifact(preparer, operations, code):
    _, result = prepare(preparer, operations)
    stage = result.report.results[0]
    assert stage.status is CheckStatus.FAIL
    assert stage.feedback[0].code == code
    assert stage.feedback[0].location.startswith("/operations/")
    assert result.candidate is None


def test_incompatible_multi_types_explicitly_unsupported(preparer):
    _, result = prepare(
        preparer, (CreateEntity("create", "s", (NS + "Student", NS + "Course")),)
    )
    assert result.report.results[0].status is CheckStatus.UNSUPPORTED
    assert result.report.results[0].feedback[0].code == "unrelated_types"
    assert result.report.results[0].feedback[0].category is FeedbackCategory.UNSUPPORTED
    assert result.candidate is None


def test_chain_compatible_types_use_existing_planner(preparer):
    _, result = prepare(
        preparer, (CreateEntity("create", "s", (NS + "Student", NS + "Person")),)
    )
    assert result.report.results[0].status is CheckStatus.PASS
    assert len(result.candidate.additions) == 2


def test_revision_invalidation_and_stable_session_allocation(preparer):
    session, first = prepare(preparer, (create(),))
    updated = session.replace_proposal(
        Proposal((create(), create("course", "Course"))), expected_revision=1
    )
    second = preparer.prepare(updated)
    assert first.candidate.iri_for("student") == second.candidate.iri_for("student")
    with pytest.raises(ContractError, match="stale_candidate"):
        first.candidate.require_current(updated)
    assert first.candidate.token != second.candidate.token
    _, independent = prepare(preparer, (create(),))
    assert first.candidate.iri_for("student") != independent.candidate.iri_for(
        "student"
    )


def test_graph_access_is_defensive_and_baseline_is_captured(compiled):
    root, model = compiled
    baseline = Graph()
    baseline.add((URIRef("urn:existing"), RDF.type, URIRef(NS + "Student")))
    adapter = CandidatePreparer.from_manifest(
        model,
        root / "manifest.json",
        schema_id="urn:schema",
        base_revision="r1",
        accepted_graph=baseline,
    )
    original = frozenset(baseline)
    _, result = prepare(adapter, (create(),))
    assert frozenset(baseline) == original
    baseline.remove((None, None, None))
    assert (
        URIRef("urn:existing"),
        RDF.type,
        URIRef(NS + "Student"),
    ) in result.candidate.graph()
    exposed = result.candidate.graph()
    exposed.remove((None, None, None))
    assert len(result.candidate.graph()) > 0
    delta = result.candidate.delta_graph()
    delta.remove((None, None, None))
    assert len(result.candidate.delta_graph()) == 1
    _, again = prepare(adapter, ())
    assert (
        URIRef("urn:existing"),
        RDF.type,
        URIRef(NS + "Student"),
    ) in again.candidate.graph()


def test_schema_fingerprint_and_context_guards(compiled, preparer):
    root, model = compiled
    with pytest.raises(ContractError, match="schema_changed"):
        CandidatePreparer(model, preparer.context, b"changed")
    foreign = replace(preparer.context, base_revision="accepted-1")
    snapshot = OntologySystem(foreign).begin_candidate(Proposal((create(),))).snapshot
    result = preparer.prepare(snapshot)
    assert result.candidate is None
    assert result.report.results[0].feedback[0].code == "context_mismatch"
    manifest = json.loads((root / "manifest.json").read_text())
    content = (root / manifest["runtime_ontology"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == preparer.context.schema_sha256


def test_forged_snapshot_digest_rejected(preparer):
    session, _ = prepare(preparer, (create(),))
    forged = replace(session.snapshot, proposal=Proposal(()))
    result = preparer.prepare(forged)
    assert result.candidate is None
    assert result.report.results[0].feedback[0].code == "candidate_digest"


def test_no_native_side_effects_for_success_or_failure(preparer):
    from krrood.symbol_graph.symbol_graph import SymbolGraph

    graph = SymbolGraph()
    before = len(graph.wrapped_instances), len(list(graph.relations()))
    prepare(
        preparer,
        (
            create(),
            create("course", "Course"),
            ObjectAssertion("link", ref(), NS + "takesCourse", ref("course")),
        ),
    )
    prepare(preparer, (create(kind="Missing"),))
    assert (len(graph.wrapped_instances), len(list(graph.relations()))) == before


def test_literal_lexical_form_preserved_and_conversion_preflighted(preparer):
    _, result = prepare(
        preparer,
        (
            create(),
            LiteralAssertion(
                "age", ref(), NS + "age", LiteralValue("001", str(XSD.integer))
            ),
        ),
    )
    assert result.report.results[0].status is CheckStatus.PASS
    value = next(
        result.candidate.graph().objects(
            URIRef(result.candidate.iri_for("student")), URIRef(NS + "age")
        )
    )
    assert str(value) == "001" and value.toPython() == 1
    _, bad = prepare(
        preparer,
        (
            create(),
            LiteralAssertion(
                "age",
                ref(),
                NS + "age",
                LiteralValue("not-an-integer", str(XSD.integer)),
            ),
        ),
    )
    assert bad.candidate is None
    assert bad.report.results[0].feedback[0].code == "invalid_literal"


def test_binding_scope_uses_existing_planner(compiled, preparer):
    root, model = compiled
    scoped = replace(model, binding_policy="local-usage", bindings=())
    adapter = CandidatePreparer.from_manifest(
        scoped,
        root / "manifest.json",
        schema_id="urn:test:schema",
        base_revision="accepted-0",
    )
    _, result = prepare(
        adapter,
        (create(), LiteralAssertion("name", ref(), NS + "name", LiteralValue("Ada"))),
    )
    assert result.candidate is None
    assert result.report.results[0].status is CheckStatus.UNSUPPORTED
    assert result.report.results[0].feedback[0].code == "unknown_property"


def test_bad_baseline_is_reported_not_silently_skipped(compiled):
    root, model = compiled
    baseline = Graph()
    baseline.add((URIRef("urn:bad"), RDF.type, URIRef("urn:UnknownClass")))
    adapter = CandidatePreparer.from_manifest(
        model,
        root / "manifest.json",
        schema_id="urn:schema",
        base_revision="r1",
        accepted_graph=baseline,
    )
    _, result = prepare(adapter, ())
    assert result.candidate is None
    assert result.report.results[0].feedback[0].code == "unknown_type"
    assert result.report.results[0].feedback[0].location == ""


def test_accepted_graph_is_additional_abox_not_a_second_schema(compiled):
    root, model = compiled
    schema = Graph()
    schema.add((URIRef(NS + "Student"), RDF.type, OWL.Class))
    with pytest.raises(ContractError, match="accepted_schema"):
        CandidatePreparer.from_manifest(
            model,
            root / "manifest.json",
            schema_id="urn:schema",
            base_revision="r1",
            accepted_graph=schema,
        )


def test_accepted_blank_nodes_are_copied_into_a_separate_identity_scope(compiled):
    root, model = compiled
    accepted = Graph()
    accepted.add((BNode("annotation-list"), RDF.type, RDF.List))
    adapter = CandidatePreparer.from_manifest(
        model,
        root / "manifest.json",
        schema_id="urn:schema",
        base_revision="r1",
        accepted_graph=accepted,
    )
    _, result = prepare(adapter, ())
    assert result.report.results[0].status is CheckStatus.PASS
    assert BNode("annotation-list") not in result.candidate.graph().all_nodes()
    assert any(isinstance(node, BNode) for node in result.candidate.graph().all_nodes())


def test_model_terms_and_property_kinds_must_belong_to_schema(compiled):
    root, model = compiled
    wrong = replace(
        model,
        classes={**model.classes, "urn:Undeclared": model.classes[NS + "Student"]},
    )
    with pytest.raises(ContractError, match="model_schema_mismatch"):
        CandidatePreparer.from_manifest(
            wrong, root / "manifest.json", schema_id="urn:schema", base_revision="r1"
        )
    prop = model.properties[NS + "name"]
    wrong = replace(
        model, properties={**model.properties, prop.iri: replace(prop, kind="object")}
    )
    with pytest.raises(ContractError, match="model_schema_mismatch"):
        CandidatePreparer.from_manifest(
            wrong, root / "manifest.json", schema_id="urn:schema", base_revision="r1"
        )


def test_ambiguous_feedback_does_not_blame_a_guessed_request(preparer):
    _, result = prepare(
        preparer,
        (
            create(),
            LiteralAssertion(
                "good-age", ref(), NS + "age", LiteralValue("1", str(XSD.integer))
            ),
            LiteralAssertion(
                "bad-age", ref(), NS + "age", LiteralValue("bad", str(XSD.integer))
            ),
        ),
    )
    feedback = result.report.results[0].feedback[0]
    assert feedback.code == "invalid_literal"
    assert feedback.location == "" and "Request good-age" not in feedback.message


def test_prepared_records_reject_mutable_containers_and_foreign_reports(preparer):
    from krrood_ontology.preparation import PreparationResult

    session, result = prepare(preparer, (create(),))
    with pytest.raises(ContractError, match="prepared_candidate"):
        replace(result.candidate, triples=set(result.candidate.triples))
    with pytest.raises(ContractError, match="prepared_candidate"):
        replace(result.candidate, local_iris=list(result.candidate.local_iris))
    _, foreign = prepare(preparer, (create(),))
    with pytest.raises(ContractError, match="preparation_result"):
        PreparationResult(foreign.report, result.candidate)
