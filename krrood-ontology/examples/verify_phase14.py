"""Run a configured ontology reasoner through the same generic construction bridge."""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from krrood.symbol_graph.symbol_graph import SymbolGraph
from verify_phase12 import demonstration_proposal

from krrood_ontology.construction import CandidateConstructor
from krrood_ontology.contracts import CheckStage, CheckStatus, ContractError, Proposal
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.reasoning import CandidateReasoner, IsolatedReasoner, ReasonerKind
from krrood_ontology.sessions import AcceptanceBlockedError, OntologySystem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("package")
    parser.add_argument(
        "--backend",
        choices=[k.value for k in ReasonerKind],
        default=ReasonerKind.HERMIT.value,
    )
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    directory = args.directory.resolve()
    sys.path.insert(0, str(directory))
    model = importlib.import_module(args.package).MODEL
    start = time.monotonic()
    native = SymbolGraph()
    before = tuple(native.wrapped_instances), tuple(native.relations())
    source = (directory / "source/closure.ttl").read_bytes()
    preparer = CandidatePreparer.from_manifest(
        model,
        directory / "manifest.json",
        schema_id="urn:example:ontology-system",
        base_revision="accepted-0",
    )
    constructor = CandidateConstructor(preparer, directory, args.package)
    checker = CandidateReasoner(
        constructor,
        IsolatedReasoner(ReasonerKind(args.backend), timeout_seconds=args.timeout),
    )
    proposal = demonstration_proposal(model)
    session = OntologySystem(preparer.context, checker=checker).begin_candidate(
        proposal
    )
    result = checker.review(session.snapshot)
    stages = {r.stage: r for r in result.report.results}
    assert (
        stages[CheckStage.PREFLIGHT].status is CheckStatus.PASS
    ), result.report.to_dict()
    assert (
        stages[CheckStage.CONSTRUCTION].status is CheckStatus.PASS
    ), result.report.to_dict()
    assert (
        stages[CheckStage.REASONER].status is not CheckStatus.ERROR
    ), result.report.to_dict()
    assert result.construction.roundtrip_equal is True
    assert result.evidence.prepared == result.construction.prepared
    assert len(result.evidence.prepared.additions) == 3
    result.evidence.require_current(session.snapshot)
    report = session.check()
    assert (
        next(r for r in report.results if r.stage is CheckStage.REASONER).status
        is stages[CheckStage.REASONER].status
    )
    try:
        session.prepare_acceptance()
    except AcceptanceBlockedError:
        acceptance_blocked = True
    else:
        raise AssertionError("Incomplete validation allowed acceptance")
    session.replace_proposal(Proposal(proposal.operations[:2]), expected_revision=1)
    try:
        result.evidence.require_current(session.snapshot)
    except ContractError as error:
        assert error.code == "stale_candidate"
    else:
        raise AssertionError("Old reasoner evidence survived an edit")
    assert before == (tuple(native.wrapped_instances), tuple(native.relations()))
    assert source == (directory / "source/closure.ttl").read_bytes()
    print(
        json.dumps(
            {
                "classes": len(model.classes),
                "properties": len(model.properties),
                "preflight": stages[CheckStage.PREFLIGHT].status.value,
                "construction": stages[CheckStage.CONSTRUCTION].status.value,
                "reasoner": stages[CheckStage.REASONER].status.value,
                "reasoner_profile": result.evidence.capabilities.profile,
                "consistency": result.evidence.consistent,
                "can_certify_consistency": result.evidence.capabilities.can_certify_consistency,
                "engine_version": result.evidence.engine_version,
                "engine_sha256": result.evidence.engine_sha256,
                "axioms_checked": result.evidence.axioms_checked,
                "profile_findings": [
                    {"code": f.code, "axiom_text": f.axiom_text, "axiom_id": f.axiom_id}
                    for f in result.evidence.profile_findings
                ],
                "unparsed_triples": list(result.evidence.unparsed_triples),
                "feedback_codes": [f.code for f in result.evidence.feedback],
                "candidate_added_triples": len(result.evidence.prepared.additions),
                "asserted_rdf_roundtrip_equal": True,
                "old_reasoner_evidence_invalidated": True,
                "local_validation": stages[CheckStage.LOCAL_VALIDATION].status.value,
                "parent_native_graph_unchanged": True,
                "accepted_rdf_unchanged": True,
                "inferences_committed": False,
                "facts_committed": False,
                "acceptance_blocked": acceptance_blocked,
                "runtime_check_seconds": round(time.monotonic() - start, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
