"""Contract-level lifecycle tests; fixture checks perform no ontology reasoning."""

from dataclasses import FrozenInstanceError, dataclass
from threading import Event, Thread

import pytest

from krrood_ontology.contracts import (
    AcceptancePolicy,
    CandidateContext,
    CheckReport,
    CheckStage,
    CheckStatus,
    ContractError,
    CreateEntity,
    EntityReference,
    Feedback,
    FeedbackCategory,
    LiteralValue,
    ObjectAssertion,
    Proposal,
    Severity,
    StageRequirement,
    StageResult,
)
from krrood_ontology.sessions import (
    AcceptanceBlockedError,
    LifecycleError,
    OntologySystem,
    SessionState,
)

CONTEXT = CandidateContext("urn:test:schema", "a" * 64, "accepted-0")
REQUIREMENTS = tuple(
    StageRequirement(stage, "contract_fixture") for stage in CheckStage
)
POLICY = AcceptancePolicy(REQUIREMENTS)


def proposal(namespace="urn:example:"):
    return Proposal((CreateEntity("create-person", "person", (namespace + "Person",)),))


@dataclass
class FixtureChecker:
    """Only exercises orchestration; no consistency result about an ontology."""

    status: CheckStatus = CheckStatus.PASS

    def check(self, snapshot):
        return CheckReport(
            snapshot.token,
            tuple(StageResult(r.stage, self.status, r.profile) for r in REQUIREMENTS),
        )


def system(checker=None):
    return OntologySystem(CONTEXT, checker=checker, policy=POLICY)


def test_default_checker_explicitly_unsupported():
    session = system().begin_candidate(proposal())
    report = session.check()
    assert all(r.status is CheckStatus.UNSUPPORTED for r in report.results)
    assert session.state is SessionState.REVIEWED
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()


def test_review_and_acceptance_intent_without_committing():
    session = system(FixtureChecker()).begin_candidate(proposal())
    report = session.check()
    intent = session.prepare_acceptance()
    assert intent.token == report.token == session.snapshot.token
    assert intent.context == CONTEXT
    assert session.state is SessionState.READY_FOR_ACCEPTANCE
    assert not hasattr(session, "accept")


def test_edit_invalidates_review_and_old_intent():
    session = system(FixtureChecker()).begin_candidate(proposal())
    session.check()
    old = session.prepare_acceptance()
    updated = session.replace_proposal(proposal("urn:another:"), expected_revision=1)
    assert updated.token.revision == 2 and updated.token != old.token
    assert session.report is None and session.state is SessionState.DRAFT
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()
    session.check()
    assert session.prepare_acceptance().token == updated.token


def test_context_and_content_bound_digest():
    first = system().begin_candidate(proposal())
    same = system().begin_candidate(proposal())
    assert first.snapshot.token.digest == same.snapshot.token.digest
    assert first.snapshot.token.session_id != same.snapshot.token.session_id
    changed = OntologySystem(
        CandidateContext("urn:test:schema", "b" * 64, "accepted-0")
    ).begin_candidate(proposal())
    assert changed.snapshot.token.digest != first.snapshot.token.digest
    changed_base = OntologySystem(
        CandidateContext("urn:test:schema", "a" * 64, "accepted-1")
    ).begin_candidate(proposal())
    assert changed_base.snapshot.token.digest != first.snapshot.token.digest


def test_nested_contracts_are_immutable():
    p = proposal()
    with pytest.raises(FrozenInstanceError):
        p.operations[0].local_id = "changed"
    with pytest.raises(ContractError):
        Proposal(list(p.operations))
    with pytest.raises(ContractError):
        CreateEntity("x", "x", ["urn:Class"])


def test_local_and_existing_references_are_explicit():
    assert EntityReference(local_id="new").iri is None
    assert EntityReference(iri="urn:existing").local_id is None
    with pytest.raises(ContractError):
        EntityReference()
    with pytest.raises(ContractError):
        EntityReference(local_id="new", iri="urn:existing")


def test_rdf_literals_keep_lexical_and_language_metadata():
    assert (
        LiteralValue(
            "01", datatype_iri="http://www.w3.org/2001/XMLSchema#integer"
        ).lexical
        == "01"
    )
    assert LiteralValue("Hallo", language="de").language == "de"
    with pytest.raises(ContractError):
        LiteralValue("x", datatype_iri="urn:type", language="de")
    with pytest.raises(ContractError):
        LiteralValue(False)


def test_duplicate_request_and_entity_ids_are_rejected():
    create = CreateEntity("x", "person", ("urn:Person",))
    with pytest.raises(ContractError):
        Proposal((create, create))
    with pytest.raises(ContractError):
        Proposal((create, CreateEntity("y", "person", ("urn:Other",))))


def test_contract_does_not_claim_semantic_preflight():
    p = Proposal(
        (
            ObjectAssertion(
                "relation",
                EntityReference(local_id="missing"),
                "urn:unknown",
                EntityReference(iri="urn:other"),
            ),
        )
    )
    assert system().begin_candidate(p).snapshot.proposal == p


@pytest.mark.parametrize(
    "status",
    [CheckStatus.FAIL, CheckStatus.UNKNOWN, CheckStatus.UNSUPPORTED, CheckStatus.ERROR],
)
def test_nonpassing_outcomes_block_acceptance(status):
    session = system(FixtureChecker(status)).begin_candidate(proposal())
    session.check()
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()


def test_wrong_profile_and_missing_checks_block_acceptance():
    class IncompleteChecker:
        def check(self, s):
            return CheckReport(
                s.token,
                (
                    StageResult(
                        CheckStage.REASONER, CheckStatus.PASS, "different_profile"
                    ),
                ),
            )

    session = system(IncompleteChecker()).begin_candidate(proposal())
    session.check()
    with pytest.raises(AcceptanceBlockedError) as e:
        session.prepare_acceptance()
    assert any(f.code == "missing_check" for f in e.value.feedback)
    assert any(f.code == "profile_mismatch" for f in e.value.feedback)


def test_checker_exception_is_feedback_without_success():
    class BrokenChecker:
        def check(self, s):
            raise RuntimeError("private backend details")

    session = system(BrokenChecker()).begin_candidate(proposal())
    report = session.check()
    assert report.results == ()
    assert report.feedback[0].category is FeedbackCategory.EXECUTION
    assert "private backend details" not in report.feedback[0].message
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()


def test_foreign_review_is_rejected():
    foreign = system(FixtureChecker()).begin_candidate(proposal())
    result = foreign.check()

    class WrongChecker:
        def check(self, s):
            return result

    session = system(WrongChecker()).begin_candidate(proposal())
    with pytest.raises(LifecycleError, match="review_token"):
        session.check()
    assert session.report is None


def test_stale_async_review_cannot_replace_new_revision():
    entered, release = Event(), Event()

    class SlowChecker:
        def check(self, s):
            entered.set()
            assert release.wait(10)
            return FixtureChecker().check(s)

    session = system(SlowChecker()).begin_candidate(proposal())
    errors = []

    def worker():
        try:
            session.check()
        except LifecycleError as error:
            errors.append(error.code)

    thread = Thread(target=worker)
    thread.start()
    assert entered.wait(10)
    session.replace_proposal(proposal("urn:changed:"), expected_revision=1)
    release.set()
    thread.join(10)
    assert not thread.is_alive() and errors == ["stale_review"]
    assert session.report is None and session.snapshot.token.revision == 2


def test_revision_conflicts_and_closed_sessions_do_not_mutate():
    session = system().begin_candidate(proposal())
    before = session.snapshot
    with pytest.raises(LifecycleError):
        session.replace_proposal(proposal("urn:other:"), expected_revision=0)
    assert session.snapshot == before
    session.close()
    assert session.state is SessionState.CLOSED
    with pytest.raises(LifecycleError):
        session.check()
    with pytest.raises(LifecycleError):
        session.replace_proposal(proposal(), expected_revision=1)


def test_inconsistent_pass_report_is_rejected():
    error = Feedback(FeedbackCategory.REQUEST, Severity.ERROR, "bad", "Bad request")
    with pytest.raises(ContractError):
        StageResult(CheckStage.PREFLIGHT, CheckStatus.PASS, "fixture", (error,))


def test_protocol_version_and_json_ready_records():
    import json

    p = proposal()
    encoded = json.loads(json.dumps(p.to_dict()))
    assert encoded["contract_version"] == "1.0"
    assert encoded["operations"][0]["kind"] == "create_entity"
    with pytest.raises(ContractError):
        Proposal((), contract_version="2.0")


def test_manifest_context_verifies_snapshot_digest(tmp_path):
    import hashlib
    import json

    source = tmp_path / "source"
    source.mkdir()
    closure = source / "closure.ttl"
    closure.write_text("<urn:C> <urn:p> <urn:v> .")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime_ontology": "source/closure.ttl",
                "closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
            }
        )
    )
    context = CandidateContext.from_manifest(
        manifest, schema_id="urn:test:schema", base_revision="accepted-0"
    )
    assert context.schema_id == "urn:test:schema"
    closure.write_text("changed")
    with pytest.raises(ContractError):
        CandidateContext.from_manifest(
            manifest, schema_id="urn:test:schema", base_revision="accepted-0"
        )


def test_sessions_do_not_construct_native_symbols_or_relations():
    from krrood.symbol_graph.symbol_graph import SymbolGraph

    graph = SymbolGraph()
    before = (len(graph.wrapped_instances), len(list(graph.relations())))
    session = system().begin_candidate(proposal())
    session.check()
    session.replace_proposal(proposal("urn:other:"), expected_revision=1)
    session.close()
    assert (len(graph.wrapped_instances), len(list(graph.relations()))) == before


def test_acceptance_records_enforce_version_and_immutability():
    from krrood_ontology.contracts import AcceptanceRequest

    token = system().begin_candidate(proposal()).snapshot.token
    with pytest.raises(ContractError):
        AcceptanceRequest(token, CONTEXT, list(REQUIREMENTS))
    with pytest.raises(ContractError):
        AcceptanceRequest(token, CONTEXT, REQUIREMENTS, contract_version="2.0")


def test_new_check_invalidates_previous_success_on_failure():
    @dataclass
    class ToggleChecker:
        calls: int = 0

        def check(self, s):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("failed second check")
            return FixtureChecker().check(s)

    session = system(ToggleChecker()).begin_candidate(proposal())
    session.check()
    session.prepare_acceptance()
    result = session.check()
    assert result.feedback[0].code == "checker_failed"
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()
