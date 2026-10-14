"""Exercise candidate preflight with any compatible generated ontology package."""

import importlib
import json
import sys
import time
from pathlib import Path

from krrood.symbol_graph.symbol_graph import SymbolGraph
from rdflib import RDF, Graph, URIRef

from krrood_ontology.contracts import (
    CheckStatus,
    CreateEntity,
    EntityReference,
    ObjectAssertion,
    Proposal,
)
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.sessions import AcceptanceBlockedError, OntologySystem


def demonstration_proposal(model):
    """Choose a small operational proposal using only generated registry metadata."""
    # Select an operational object-property signature from metadata, without
    # ontology-specific class names. No consistency claim follows from this choice.
    signature = None
    for property_iri, mapping in sorted(model.properties.items()):
        if mapping.kind != "object":
            continue
        target_type = mapping.range_iri or next(iter(sorted(model.classes)))
        if target_type not in model.classes:
            continue
        for subject_type in sorted(model.classes):
            if model.mapping_for(property_iri, subject_type) is not None and (
                not mapping.domain_iri
                or issubclass(
                    model.classes[subject_type], model.classes[mapping.domain_iri]
                )
            ):
                signature = subject_type, property_iri, target_type
                break
        if signature:
            break
    assert signature, "This demonstration needs an operational object property"
    subject_type, predicate, target_type = signature
    return Proposal(
        (
            CreateEntity("create-subject", "subject", (subject_type,)),
            CreateEntity("create-target", "target", (target_type,)),
            ObjectAssertion(
                "link",
                EntityReference(local_id="subject"),
                predicate,
                EntityReference(local_id="target"),
            ),
        )
    )


def main():
    directory = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "generated"
    )
    package = sys.argv[2] if len(sys.argv) > 2 else "university_model"
    sys.path.insert(0, str(directory.resolve()))
    model = importlib.import_module(package).MODEL
    manifest = directory / "manifest.json"
    baseline = Graph()
    baseline.add(
        (
            URIRef("urn:phase12-demo:existing"),
            RDF.type,
            URIRef(next(iter(sorted(model.classes)))),
        )
    )
    before_rdf = frozenset(baseline)
    native = SymbolGraph()
    before = len(native.wrapped_instances), len(list(native.relations()))
    start = time.monotonic()
    preparer = CandidatePreparer.from_manifest(
        model,
        manifest,
        schema_id="urn:example:ontology-system",
        base_revision="accepted-0",
        accepted_graph=baseline,
    )
    proposal = demonstration_proposal(model)
    predicate = proposal.operations[-1].property_iri
    session = OntologySystem(preparer.context, checker=preparer).begin_candidate(
        proposal
    )
    first = preparer.prepare(session.snapshot)
    assert first.report.results[0].status is CheckStatus.PASS
    assert len(first.candidate.additions) == 3
    session.check()
    try:
        session.prepare_acceptance()
    except AcceptanceBlockedError:
        acceptance_blocked = True
    else:
        raise AssertionError("Default policy allowed incomplete checks")
    old_token = first.candidate.token
    session.replace_proposal(Proposal(proposal.operations[:2]), expected_revision=1)
    assert session.report is None
    current = preparer.prepare(session.snapshot)
    assert current.report.results[0].status is CheckStatus.PASS
    assert current.candidate.token != old_token
    assert current.candidate.iri_for("subject") == first.candidate.iri_for("subject")
    invalid = OntologySystem(preparer.context, checker=preparer).begin_candidate(
        Proposal(
            (
                ObjectAssertion(
                    "unresolved",
                    EntityReference(local_id="missing"),
                    predicate,
                    EntityReference(local_id="missing"),
                ),
            )
        )
    )
    rejected = preparer.prepare(invalid.snapshot)
    assert rejected.candidate is None
    assert rejected.report.results[0].feedback[0].code == "unresolved_reference"
    assert frozenset(baseline) == before_rdf
    after = len(native.wrapped_instances), len(list(native.relations()))
    assert before == after
    print(
        json.dumps(
            {
                "contract_version": proposal.contract_version,
                "classes": len(model.classes),
                "properties": len(model.properties),
                "preflight": first.report.results[0].status.value,
                "preflight_profile": first.report.results[0].profile,
                "candidate_entities": len(first.candidate.local_iris),
                "candidate_added_triples": len(first.candidate.additions),
                "invalid_reference_rejected": True,
                "revision": current.candidate.token.revision,
                "old_review_invalidated": True,
                "provisional_iris_stable_across_revision": True,
                "accepted_rdf_unchanged": True,
                "acceptance_blocked": acceptance_blocked,
                "reasoner": "unsupported; not executed",
                "facts_committed": False,
                "native_symbols_created": after[0] - before[0],
                "native_relations_created": after[1] - before[1],
                "runtime_check_seconds": round(time.monotonic() - start, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
