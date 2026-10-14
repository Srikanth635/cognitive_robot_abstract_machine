"""Actual ontology reasoning and revision-bound service feedback."""

import hashlib
import importlib
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from krrood.symbol_graph.symbol_graph import SymbolGraph
from rdflib import OWL, RDF, XSD, Graph, URIRef

from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput
from krrood_ontology.construction import CandidateConstructor
from krrood_ontology.contracts import (
    CandidateContext,
    CandidateSnapshot,
    CheckStage,
    CheckStatus,
    ContractError,
    CreateEntity,
    EntityReference,
    LiteralAssertion,
    LiteralValue,
    ObjectAssertion,
    Proposal,
)
from krrood_ontology.preparation import CandidatePreparer, PreparedCandidate
from krrood_ontology.reasoning import (
    CandidateReasoner,
    IsolatedReasoner,
    ReasonerKind,
    UnavailableReasoner,
)
from krrood_ontology.sessions import AcceptanceBlockedError, OntologySystem

NS = "urn:reasoning:"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase14") / "generated"
    OntologyCompiler().compile(
        OntologyInput(Path(__file__).parent / "fixtures/reasoner.ttl"),
        CompilationOptions(
            root,
            "reasoning_model",
            structural_only=True,
            binding_policy="local-usage",
            scoped_semantics=True,
        ),
    )
    sys.path.insert(0, str(root))
    module = importlib.import_module("reasoning_model")
    yield root, module
    sys.path.remove(str(root))


@pytest.fixture(scope="module")
def constructor(generated):
    root, module = generated
    preparer = CandidatePreparer.from_manifest(
        module.MODEL,
        root / "manifest.json",
        schema_id="urn:test:reasoner",
        base_revision="r0",
    )
    return CandidateConstructor(preparer, root, "reasoning_model")


@pytest.fixture(scope="module")
def hermit():
    if not importlib.util.find_spec("owlready2") or not importlib.util.find_spec(
        "jpype"
    ):
        pytest.skip("Install the optional reasoner extra for real HermiT tests")
    import jpype

    try:
        jpype.getDefaultJVMPath()
    except jpype.JVMNotFoundException:
        pytest.skip("A Java runtime is required for real HermiT tests")
    return IsolatedReasoner(ReasonerKind.HERMIT)


def entity(local="a", kind="Normal"):
    return CreateEntity("create-" + local, local, (NS + kind,))


def link(subject="a", target="b", prop="link"):
    return ObjectAssertion(
        "link-" + subject + "-" + target,
        EntityReference(local_id=subject),
        NS + prop,
        EntityReference(local_id=target),
    )


def review(constructor, backend, operations):
    checker = CandidateReasoner(constructor, backend)
    session = OntologySystem(
        constructor.preparer.context, checker=checker
    ).begin_candidate(Proposal(tuple(operations)))
    return session, checker.review(session.snapshot)


def reasoner_stage(result):
    return next(r for r in result.report.results if r.stage is CheckStage.REASONER)


def test_default_backend_is_unavailable_and_acceptance_blocked(constructor):
    checker = CandidateReasoner(constructor)
    session = OntologySystem(
        constructor.preparer.context, checker=checker
    ).begin_candidate(Proposal((entity(),)))
    report = session.check()
    assert (
        next(r for r in report.results if r.stage is CheckStage.CONSTRUCTION).status
        is CheckStatus.PASS
    )
    assert (
        next(r for r in report.results if r.stage is CheckStage.REASONER).status
        is CheckStatus.UNSUPPORTED
    )
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()


def test_real_hermit_consistent_candidate_with_empty_unsatisfiable_classes(
    constructor, hermit
):
    session, result = review(constructor, hermit, (entity(),))
    assert reasoner_stage(result).status is CheckStatus.PASS, result.report.to_dict()
    assert result.evidence.consistent is True
    assert result.evidence.axioms_checked > 0
    assert result.evidence.worker_pid != os.getpid()
    assert result.evidence.capabilities.checks_owl2_dl_profile is True
    assert result.evidence.capabilities.can_certify_consistency is True
    assert result.evidence.profile_findings == ()
    result.evidence.require_current(session.snapshot)


def test_real_hermit_detects_disjoint_subclass_instance(constructor, hermit):
    _, result = review(constructor, hermit, (entity(kind="Impossible"),))
    assert result.construction is not None
    assert reasoner_stage(result).status is CheckStatus.FAIL
    assert result.evidence.consistent is False
    assert reasoner_stage(result).feedback[0].code == "ontology_inconsistent"


def test_missing_existential_witness_is_consistent_under_open_world(
    constructor, hermit
):
    _, result = review(constructor, hermit, (entity(kind="Required"),))
    assert reasoner_stage(result).status is CheckStatus.PASS
    assert result.evidence.consistent is True


def test_existential_and_all_nothing_conflict_detected(constructor, hermit):
    _, result = review(constructor, hermit, (entity(kind="ForbiddenWitness"),))
    assert reasoner_stage(result).status is CheckStatus.FAIL
    assert result.evidence.consistent is False


@pytest.mark.parametrize("prop,kind", [("functional", "Normal"), ("link", "Limited")])
def test_multiple_targets_without_distinctness_can_be_consistent(
    constructor, hermit, prop, kind
):
    _, result = review(
        constructor,
        hermit,
        (
            entity(kind=kind),
            entity("b"),
            entity("c"),
            link(prop=prop),
            link(target="c", prop=prop),
        ),
    )
    assert reasoner_stage(result).status is CheckStatus.PASS
    assert result.evidence.consistent is True


@pytest.mark.parametrize("prop,kind", [("functional", "Normal"), ("link", "Limited")])
def test_distinct_targets_conflict_with_functionality_or_max_one(
    constructor, hermit, prop, kind
):
    root = constructor.directory
    # Additional accepted facts are captured by the existing preparer, not new request operations.
    accepted = Graph()
    for local in ("b", "c"):
        accepted.add((URIRef("urn:accepted:" + local), RDF.type, URIRef(NS + "Normal")))
    accepted.add(
        (URIRef("urn:accepted:b"), OWL.differentFrom, URIRef("urn:accepted:c"))
    )
    preparer = CandidatePreparer.from_manifest(
        constructor.preparer.model,
        root / "manifest.json",
        schema_id=constructor.preparer.context.schema_id,
        base_revision="r0",
        accepted_graph=accepted,
    )
    isolated = CandidateConstructor(preparer, root, "reasoning_model")
    operations = (entity(kind=kind),) + tuple(
        ObjectAssertion(
            "link-" + local,
            EntityReference(local_id="a"),
            NS + prop,
            EntityReference(iri="urn:accepted:" + local),
        )
        for local in ("b", "c")
    )
    _, result = review(isolated, hermit, operations)
    assert reasoner_stage(result).status is CheckStatus.FAIL
    assert result.evidence.consistent is False


def test_real_hermit_asymmetric_conflict(constructor, hermit):
    _, result = review(
        constructor,
        hermit,
        (entity(), entity("b"), link(prop="before"), link("b", "a", "before")),
    )
    assert reasoner_stage(result).status is CheckStatus.FAIL


def test_literal_spelling_and_candidate_rdf_remain_unchanged(constructor, hermit):
    _, result = review(
        constructor,
        hermit,
        (
            entity(),
            LiteralAssertion(
                "decimal",
                EntityReference(local_id="a"),
                NS + "amount",
                LiteralValue("001.2300", str(XSD.decimal)),
            ),
        ),
    )
    assert reasoner_stage(result).status is CheckStatus.PASS
    prepared = result.evidence.prepared
    value = next(
        prepared.graph().objects(URIRef(prepared.iri_for("a")), URIRef(NS + "amount"))
    )
    assert str(value) == "001.2300"
    exposed = prepared.graph()
    exposed.remove((None, None, None))
    assert frozenset(result.evidence.graph()) == result.construction.prepared.triples


def test_reasoning_does_not_mutate_parent_native_graph_or_source(constructor, hermit):
    native = SymbolGraph()
    nodes, relations = tuple(native.wrapped_instances), tuple(native.relations())
    source = (constructor.directory / "source/closure.ttl").read_bytes()
    review(constructor, hermit, (entity(),))
    review(constructor, hermit, (entity(kind="Impossible"),))
    assert tuple(native.wrapped_instances) == nodes
    assert tuple(native.relations()) == relations
    assert (constructor.directory / "source/closure.ttl").read_bytes() == source


def test_revision_edit_invalidates_reasoner_evidence(constructor, hermit):
    session, result = review(constructor, hermit, (entity(),))
    session.replace_proposal(Proposal((entity("b"),)), expected_revision=1)
    with pytest.raises(ContractError, match="stale_candidate"):
        result.evidence.require_current(session.snapshot)


def test_failed_preflight_does_not_start_reasoner(constructor, monkeypatch):
    def forbidden(self, prepared):
        raise AssertionError("Reasoning started on invalid preflight")

    monkeypatch.setattr(UnavailableReasoner, "evaluate", forbidden)
    _, result = review(constructor, UnavailableReasoner(), (entity(kind="Unknown"),))
    assert reasoner_stage(result).status is CheckStatus.UNKNOWN
    assert result.evidence is None


def prepared_graph(text):
    graph = Graph().parse(data=text, format="turtle")
    context = CandidateContext(
        "urn:reasoner:schema", hashlib.sha256(text.encode()).hexdigest(), "r0"
    )
    snapshot = CandidateSnapshot.build("reasoner-test", 1, context, Proposal(()))
    return PreparedCandidate(snapshot.token, context, (), frozenset(graph), frozenset())


def test_owlrl_uses_existing_engine_but_no_clashes_is_unknown(constructor):
    _, result = review(constructor, IsolatedReasoner(ReasonerKind.OWL_RL), (entity(),))
    assert reasoner_stage(result).status is CheckStatus.UNKNOWN
    assert result.evidence.consistent is None
    assert result.evidence.capabilities.can_certify_consistency is False


def test_owlrl_returns_real_engine_clash_diagnostics(constructor):
    _, result = review(
        constructor, IsolatedReasoner(ReasonerKind.OWL_RL), (entity(kind="Impossible"),)
    )
    assert reasoner_stage(result).status is CheckStatus.FAIL
    assert result.evidence.consistent is False
    assert any(f.code == "owlrl_clash" for f in result.evidence.feedback)


def test_unresolved_imports_are_rejected_before_any_engine(monkeypatch):
    prepared = prepared_graph(
        "<urn:o> <http://www.w3.org/2002/07/owl#imports> <https://example.invalid/missing> ."
    )
    backend = IsolatedReasoner(ReasonerKind.OWL_RL)
    monkeypatch.setattr(
        IsolatedReasoner,
        "_run_worker",
        lambda *args: pytest.fail("Engine executed with an unresolved import"),
    )
    evidence = backend.evaluate(prepared)
    assert evidence.status is CheckStatus.UNSUPPORTED
    assert evidence.feedback[0].code == "reasoner_unresolved_import"


def test_timeout_returns_error_without_consistency_claim(constructor, monkeypatch):
    import subprocess

    def expired(*args):
        raise subprocess.TimeoutExpired("reasoner", 1)

    monkeypatch.setattr(IsolatedReasoner, "_run_worker", expired)
    _, result = review(constructor, IsolatedReasoner(ReasonerKind.OWL_RL), (entity(),))
    assert reasoner_stage(result).status is CheckStatus.ERROR
    assert result.evidence.consistent is None
    assert result.evidence.feedback[0].code == "reasoner_timeout"


def test_foreign_backend_evidence_is_rejected(constructor):
    class ForeignReasoner:
        capabilities = UnavailableReasoner().capabilities

        def evaluate(self, prepared):
            return UnavailableReasoner().evaluate(
                replace(prepared, token=replace(prepared.token, revision=2))
            )

    _, result = review(constructor, ForeignReasoner(), (entity(),))
    assert reasoner_stage(result).status is CheckStatus.ERROR
    assert reasoner_stage(result).feedback[0].code == "reasoner_protocol"
    assert result.evidence is None


def test_real_hermit_outside_dl_profile_is_unsupported(hermit):
    prepared = prepared_graph(
        (Path(__file__).parent / "fixtures/reasoner_non_dl.ttl").read_text()
    )
    evidence = hermit.evaluate(prepared)
    assert evidence.status is CheckStatus.UNSUPPORTED
    assert evidence.consistent is None
    assert any(
        f.code == "UseOfNonSimplePropertyInCardinalityRestriction"
        for f in evidence.profile_findings
    )


def test_backend_execution_failure_is_not_inconsistency(constructor, monkeypatch):
    def broken(*args):
        raise RuntimeError("private engine detail")

    monkeypatch.setattr(IsolatedReasoner, "_run_worker", broken)
    _, result = review(constructor, IsolatedReasoner(ReasonerKind.OWL_RL), (entity(),))
    assert reasoner_stage(result).status is CheckStatus.ERROR
    assert result.evidence.consistent is None
    assert "private engine detail" not in result.evidence.feedback[0].message


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_reasoner_deadline_rejected(seconds):
    with pytest.raises(ValueError):
        IsolatedReasoner(ReasonerKind.OWL_RL, timeout_seconds=seconds)


def test_each_reasoner_evaluation_uses_fresh_worker(constructor, hermit):
    session, first = review(constructor, hermit, (entity(),))
    second = hermit.evaluate(first.evidence.prepared)
    assert first.evidence.worker_pid != second.worker_pid
    assert first.evidence.prepared == second.prepared
    assert second.status is CheckStatus.PASS


def test_real_timeout_reaps_embedded_jvm_process(
    constructor, hermit, tmp_path, monkeypatch
):
    import shutil

    import krrood_ontology.reasoning as module

    runtime = tmp_path / "runtime/krrood_ontology"
    runtime.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures/reasoner_timeout_worker.py"
    shutil.copyfile(fixture, runtime / "reasoning_worker.py")
    monkeypatch.setattr(module, "__file__", str(runtime / "reasoning.py"))
    snapshot = (
        OntologySystem(constructor.preparer.context)
        .begin_candidate(Proposal((entity(),)))
        .snapshot
    )
    prepared = constructor.preparer.prepare(snapshot).candidate
    native = SymbolGraph()
    before = tuple(native.wrapped_instances), tuple(native.relations())
    evidence = replace(hermit, timeout_seconds=5).evaluate(prepared)
    assert evidence.status is CheckStatus.ERROR
    assert evidence.consistent is None
    assert evidence.feedback[0].code == "reasoner_timeout"
    pid = int((runtime / "reasoning_worker.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert before == (tuple(native.wrapped_instances), tuple(native.relations()))


def test_optional_dependency_missing_reports_unsupported(monkeypatch):
    from krrood_ontology.reasoning import _ReasonerJob
    from krrood_ontology.reasoning_worker import hermit_review

    prepared = prepared_graph("<urn:Entity> a <http://www.w3.org/2002/07/owl#Class> .")
    monkeypatch.setitem(sys.modules, "jpype", None)
    evidence = hermit_review(
        _ReasonerJob(prepared, ReasonerKind.HERMIT, 512, None), Path.cwd()
    )
    assert evidence.status is CheckStatus.UNSUPPORTED
    assert evidence.consistent is None
    assert evidence.feedback[0].code == "reasoner_dependency_missing"


def test_missing_java_shared_library_reports_unsupported(constructor, hermit):
    prepared = constructor.preparer.prepare(
        OntologySystem(constructor.preparer.context)
        .begin_candidate(Proposal((entity(),)))
        .snapshot
    ).candidate
    evidence = replace(hermit, jvm_path="/nonexistent/phase14/libjvm.so").evaluate(
        prepared
    )
    assert evidence.status is CheckStatus.UNSUPPORTED
    assert evidence.consistent is None
    assert evidence.feedback[0].code == "reasoner_java_missing"


def test_diagnostic_profile_cannot_claim_consistency_pass(constructor):
    from krrood_ontology.reasoning import OWL_RL_CAPABILITIES, ReasonerEvidence

    prepared = constructor.preparer.prepare(
        OntologySystem(constructor.preparer.context)
        .begin_candidate(Proposal((entity(),)))
        .snapshot
    ).candidate
    with pytest.raises(ContractError, match="reasoner_evidence"):
        ReasonerEvidence(prepared, OWL_RL_CAPABILITIES, CheckStatus.PASS, True)


def test_mutable_evidence_and_false_conclusions_are_rejected(constructor):
    from krrood_ontology.reasoning import HERMIT_CAPABILITIES, ReasonerEvidence

    prepared = constructor.preparer.prepare(
        OntologySystem(constructor.preparer.context)
        .begin_candidate(Proposal((entity(),)))
        .snapshot
    ).candidate
    with pytest.raises(ContractError):
        ReasonerEvidence(prepared, HERMIT_CAPABILITIES, CheckStatus.ERROR, False)
    with pytest.raises(ContractError):
        ReasonerEvidence(
            prepared,
            HERMIT_CAPABILITIES,
            CheckStatus.UNKNOWN,
            None,
            profile_findings=[],
        )


def test_isolated_runner_rejects_foreign_evidence(constructor, monkeypatch):
    snapshot = (
        OntologySystem(constructor.preparer.context)
        .begin_candidate(Proposal((entity(),)))
        .snapshot
    )
    prepared = constructor.preparer.prepare(snapshot).candidate
    foreign = UnavailableReasoner().evaluate(
        replace(prepared, token=replace(prepared.token, revision=2))
    )
    monkeypatch.setattr(IsolatedReasoner, "_run_worker", lambda *args: foreign)
    evidence = IsolatedReasoner(ReasonerKind.OWL_RL).evaluate(prepared)
    assert evidence.status is CheckStatus.ERROR
    assert evidence.feedback[0].code == "reasoner_protocol"


def test_construction_failure_does_not_execute_reasoner(constructor, monkeypatch):
    from krrood_ontology.construction import ConstructionResult
    from krrood_ontology.contracts import CheckReport, StageResult

    def failed(self, snapshot):
        return ConstructionResult(
            CheckReport(
                snapshot.token,
                (
                    StageResult(
                        CheckStage.CONSTRUCTION, CheckStatus.ERROR, "fault_fixture"
                    ),
                ),
            ),
            None,
        )

    monkeypatch.setattr(CandidateConstructor, "construct", failed)
    monkeypatch.setattr(
        UnavailableReasoner,
        "evaluate",
        lambda *args: pytest.fail("Reasoner executed after failed construction"),
    )
    _, result = review(constructor, UnavailableReasoner(), (entity(),))
    assert reasoner_stage(result).status is CheckStatus.UNKNOWN
    assert result.evidence is None


def test_real_hermit_unsupported_datatype_is_not_ignored(hermit):
    prepared = prepared_graph(
        (Path(__file__).parent / "fixtures/reasoner_datatype.ttl").read_text()
    )
    evidence = hermit.evaluate(prepared)
    assert evidence.status is CheckStatus.UNSUPPORTED
    assert evidence.consistent is None
    assert evidence.feedback[0].code == "reasoner_datatype_unsupported"


def test_unparsed_rdf_cannot_be_certified_consistent(hermit):
    prepared = prepared_graph(
        (Path(__file__).parent / "fixtures/reasoner_unparsed.ttl").read_text()
    )
    evidence = hermit.evaluate(prepared)
    assert evidence.status is CheckStatus.UNSUPPORTED
    assert evidence.consistent is None
    assert len(evidence.unparsed_triples) > 0
    assert any(f.code == "reasoner_unparsed_rdf" for f in evidence.feedback)


def test_reasoner_sees_axioms_from_resolved_local_imports(tmp_path, hermit):
    from krrood_ontology import LocalImport

    fixture = Path(__file__).parent / "fixtures/reasoner_imports"
    root = tmp_path / "generated"
    OntologyCompiler().compile(
        OntologyInput(fixture / "main.ttl"),
        CompilationOptions(
            root,
            "reasoner_import_model",
            structural_only=True,
            local_imports=(
                LocalImport(
                    "urn:reasoner-import:core", OntologyInput(fixture / "core.ttl")
                ),
            ),
        ),
    )
    sys.path.insert(0, str(root))
    try:
        model = importlib.import_module("reasoner_import_model").MODEL
        preparer = CandidatePreparer.from_manifest(
            model,
            root / "manifest.json",
            schema_id="urn:import:authority",
            base_revision="r0",
        )
        constructor = CandidateConstructor(preparer, root, "reasoner_import_model")
        _, result = review(
            constructor,
            hermit,
            (CreateEntity("create", "a", ("urn:reasoner-import:Impossible",)),),
        )
        assert reasoner_stage(result).status is CheckStatus.FAIL
        assert result.evidence.consistent is False
        assert not any(result.evidence.graph().triples((None, OWL.imports, None)))
    finally:
        sys.path.remove(str(root))
