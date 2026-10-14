"""Exercise contracts only; fixture checks do not construct entities or run a reasoner."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from krrood.symbol_graph.symbol_graph import SymbolGraph

from krrood_ontology.contracts import (
    AcceptancePolicy,
    CandidateContext,
    CheckReport,
    CheckStage,
    CheckStatus,
    CreateEntity,
    EntityReference,
    ObjectAssertion,
    Proposal,
    StageRequirement,
    StageResult,
)
from krrood_ontology.sessions import AcceptanceBlockedError, OntologySystem


@dataclass(frozen=True)
class ContractFixtureChecker:
    """Checks orchestration paths only; its profiles explicitly identify fixture results."""

    def check(self, snapshot):
        return CheckReport(
            snapshot.token,
            tuple(
                StageResult(stage, CheckStatus.PASS, "contract_fixture")
                for stage in CheckStage
            ),
        )


def main():
    context = (
        CandidateContext.from_manifest(
            Path(sys.argv[1]) / "manifest.json",
            schema_id="urn:example:schema",
            base_revision="accepted-0",
        )
        if len(sys.argv) > 1
        else CandidateContext("urn:example:schema", "a" * 64, "accepted-0")
    )
    base = "https://example.org/university#"
    proposal = Proposal(
        (
            CreateEntity("create-student", "student", (base + "Student",)),
            CreateEntity("create-course", "course", (base + "Course",)),
            ObjectAssertion(
                "takes-course",
                EntityReference(local_id="student"),
                base + "takesCourse",
                EntityReference(local_id="course"),
            ),
        )
    )
    graph = SymbolGraph()
    before = (len(graph.wrapped_instances), len(list(graph.relations())))
    unavailable = OntologySystem(context).begin_candidate(proposal)
    report = unavailable.check()
    assert all(r.status is CheckStatus.UNSUPPORTED for r in report.results)
    try:
        unavailable.prepare_acceptance()
    except AcceptanceBlockedError:
        pass
    else:
        raise AssertionError("Unavailable checks allowed acceptance")
    policy = AcceptancePolicy(
        tuple(StageRequirement(stage, "contract_fixture") for stage in CheckStage)
    )
    session = OntologySystem(
        context, checker=ContractFixtureChecker(), policy=policy
    ).begin_candidate(proposal)
    session.check()
    first = session.prepare_acceptance()
    session.replace_proposal(Proposal(proposal.operations[:2]), expected_revision=1)
    assert session.report is None and session.snapshot.token != first.token
    session.check()
    current = session.prepare_acceptance()
    assert current.token.revision == 2
    after = (len(graph.wrapped_instances), len(list(graph.relations())))
    assert before == after
    print(
        json.dumps(
            {
                "contract_version": proposal.contract_version,
                "default_checks": "unsupported, acceptance blocked",
                "demonstration_checker": "contract_fixture; no ontology reasoning",
                "revision": current.token.revision,
                "old_review_invalidated": True,
                "acceptance_intent_prepared": True,
                "facts_committed": False,
                "native_symbols_created": after[0] - before[0],
                "native_relations_created": after[1] - before[1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
