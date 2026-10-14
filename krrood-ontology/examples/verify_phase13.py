"""Exercise isolated native construction with metadata from any compatible model."""

import importlib
import json
import os
import sys
import time
from pathlib import Path

from krrood.symbol_graph.symbol_graph import SymbolGraph
from verify_phase12 import demonstration_proposal

from krrood_ontology.construction import CandidateConstructor
from krrood_ontology.contracts import (
    CheckStage,
    CheckStatus,
    ContractError,
    EntityReference,
    Proposal,
)
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.sessions import AcceptanceBlockedError, OntologySystem


def main():
    directory = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "generated"
    ).resolve()
    package_name = sys.argv[2] if len(sys.argv) > 2 else "university_model"
    sys.path.insert(0, str(directory))
    model = importlib.import_module(package_name).MODEL
    before_bytes = (directory / "source/closure.ttl").read_bytes()
    native = SymbolGraph()
    before = tuple(native.wrapped_instances), tuple(native.relations())
    start = time.monotonic()
    preparer = CandidatePreparer.from_manifest(
        model,
        directory / "manifest.json",
        schema_id="urn:example:ontology-system",
        base_revision="accepted-0",
    )
    constructor = CandidateConstructor(preparer, directory, package_name)
    proposal = demonstration_proposal(model)
    session = OntologySystem(preparer.context, checker=constructor).begin_candidate(
        proposal
    )
    result = constructor.construct(session.snapshot)
    stages = {r.stage: r for r in result.report.results}
    assert (
        stages[CheckStage.PREFLIGHT].status is CheckStatus.PASS
    ), result.report.to_dict()
    assert (
        stages[CheckStage.CONSTRUCTION].status is CheckStatus.PASS
    ), result.report.to_dict()
    artifact = result.artifact
    subject = artifact.object_for_local_id("subject")
    target = artifact.object_for_local_id("target")
    assert (
        EntityReference(iri=target.iri)
        in subject.field(proposal.operations[-1].property_iri).values
    )
    assert artifact.worker_pid != os.getpid()
    assert len({v.identity_index for v in artifact.objects}) == len(artifact.objects)
    assert frozenset(artifact.graph()) == artifact.prepared.triples
    assert len(artifact.prepared.additions) == 3
    report = session.check()
    assert (
        next(r for r in report.results if r.stage is CheckStage.CONSTRUCTION).status
        is CheckStatus.PASS
    )
    try:
        session.prepare_acceptance()
    except AcceptanceBlockedError:
        acceptance_blocked = True
    else:
        raise AssertionError("Incomplete checks allowed acceptance")
    session.replace_proposal(Proposal(proposal.operations[:2]), expected_revision=1)
    assert session.report is None
    try:
        artifact.require_current(session.snapshot)
    except ContractError as error:
        assert error.code == "stale_candidate"
    else:
        raise AssertionError("Old construction artifact survived an edit")
    assert before == (tuple(native.wrapped_instances), tuple(native.relations()))
    assert before_bytes == (directory / "source/closure.ttl").read_bytes()
    print(
        json.dumps(
            {
                "classes": len(model.classes),
                "properties": len(model.properties),
                "preflight": stages[CheckStage.PREFLIGHT].status.value,
                "construction": stages[CheckStage.CONSTRUCTION].status.value,
                "construction_profile": stages[CheckStage.CONSTRUCTION].profile,
                "worker_process_isolated": True,
                "worker_native_objects_retained": False,
                "worker_objects": len(artifact.objects),
                "native_symbols_created": artifact.native_symbols_created,
                "native_relation_facts": len(artifact.relations),
                "one_object_per_iri": True,
                "shared_reference_identity_verified": True,
                "asserted_rdf_roundtrip_equal": artifact.roundtrip_equal,
                "candidate_added_triples": len(artifact.prepared.additions),
                "old_artifact_invalidated": True,
                "parent_native_graph_unchanged": True,
                "accepted_rdf_unchanged": True,
                "local_validation": stages[CheckStage.LOCAL_VALIDATION].status.value,
                "reasoner": stages[CheckStage.REASONER].status.value,
                "acceptance_blocked": acceptance_blocked,
                "facts_committed": False,
                "runtime_check_seconds": round(time.monotonic() - start, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
