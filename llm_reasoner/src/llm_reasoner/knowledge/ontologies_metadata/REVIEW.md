# Modelling review and change record

> Historical review/catalog; its fixture versions and counts retain their original scope. Current coverage and consolidation findings: [0.21.0 review](review/coverage-0.21.0/REVIEW_REPORT.md).

## What was changed

1. Selected the larger supplied CRAM-AICOR ontology as baseline: the smaller AICOR adds no structurally unique assertions.
2. Collapsed repeated anonymous OWL expressions. This reduces the baseline by 408 triples and 77 restriction nodes. This is structural deduplication, not proof of elimination of all logically equivalent expressions.
3. Replaced 14 local `dul_*` placeholders with actual DUL IRIs and added the DUL import. Previously these placeholders were classes under owl:Thing; their names did not carry DUL semantics. Exact mappings are in reports/term_mapping.csv.
4. Gave the working candidate a distinct ontology identifier `urn:gpt-cram-ontologies:aicor-unified` and candidate version, retaining domain entity IRIs for traceability. Replaced header and placeholder metadata are retained in reports/replaced_source_metadata.ttl.
5. Removed five inherited parent axioms that introduced visible DUL disjointness clashes. Their mapped form is preserved in reports/removed_dul_parent_axioms.ttl.

| Domain class | Removed DUL parent | Reason |
|---|---|---|
| ComputationalMedium | InformationObject | Descendants are also Descriptions |
| ComputationalProcess | Process | Descendants are Descriptions, conflicting with Event ancestry |
| ReifiedTemporalEntity | InformationObject | EventType also inherits Description |
| ExecutiveCapability | Quality | Descendants are also Descriptions |
| TaskStatus | InformationObject | FluentType ancestry supplies Description |

Removing these axioms weakens the ontology; it is a conservative repair pending intentional reclassification. It does not establish that every remaining parent is appropriate. Reports record the changes so they can be reviewed or reversed.

## Which ontology suits the objective?

The supplied CRAM-AICOR is the better preservation baseline because it includes the smaller file's content. Neither input should automatically become the final generic action knowledge core: both combine implementation concepts and domain abstractions. This first candidate consolidates the inputs and makes their foundational commitments explicit.

The comparison concerns the uploaded files, not a certification that they match the current GitHub branch. Repository code alignment must be checked separately before claiming runtime coverage.

## Important distinctions to preserve in the next review

- An action description and an occurring action are different entities. A generated comprehensive designator should describe an action; it is not itself the physical event.
- Generic action knowledge, injected world facts and execution episodes need distinguishable identity and provenance.
- A shared representation can support multiple projections, but a projection needs an explicit transformation contract and missing-information policy. OWL subclass axioms alone do not implement projections.
- An LLM statement, a logical consequence, a measured fact and a simulation result need distinct evidential status. Importing DUL does not turn generated claims into facts.
- Reusable stereotypical knowledge should not require a robot, scene, controller or simulator instance.

## Unresolved modelling questions

Names alone do not justify equivalence. Local Action, Entity, PhysicalObject, soma_Action, Designator and CRAMDesignator have not been indiscriminately collapsed. In particular, local Process belongs to CouplingType, whereas DUL Process is an event category. Local Task is modelled as a description; directly identifying it with DUL Task would require review. CRAMDesignator inherits Designator and adds restrictions, which can be intentional specialization rather than duplication.

Vessel under Representation and the description-versus-occurrence meaning of ForceDynamicEvent deserve explicit review. Existing restrictions on ActionDesignator and other shared terms were preserved; preservation is not an endorsement of each existential requirement. Review the 18 changed subjects in reports/shared_term_changes.json before mapping runtime objects.

## Recommended next step within this scope

Review the retained generic/implementation distinctions and five weakened parents, then obtain full named-class satisfiability results. Only after that should we extract a generic DUL-based action description core and a separate CRAM implementation bridge. Frame, phase, force-dynamic and other perspectives can subsequently annotate that shared core through explicit links. These later modules are not part of this deliverable.
