# CRAM-AICOR and DUL: classes that become unsatisfiable once aligned

A draft report for the maintainers of CRAM-AICOR (`http://aicor.knowledge/l2-ontology.owl`),
written 2026-10-26 from the copy pinned in this repository
(`llm_reasoner/src/llm_reasoner/knowledge/ontologies/linked_ontologies/cram-aicor.ttl`,
SHA-256 `a33fb7af8c28792825542b1c663f233102e19ae076bbf747fdd16e4297ef0cda`) and DUL 3.32.

## What happens

CRAM-AICOR declares 14 classes copying DUL's (`aicor:dul_Action`, `dul_Agent`,
`dul_Concept`, `dul_Description`, `dul_Entity`, `dul_Event`, `dul_InformationObject`,
`dul_PhysicalAgent`, `dul_PhysicalObject`, `dul_Process`, `dul_Quality`, `dul_Region`,
`dul_Relation`, `dul_Situation`) and does not relate them to DUL: the file holds no
statement in DUL's namespace. When each copy is stated equivalent to the DUL class of the
same name, DUL's disjointness axioms apply to CRAM-AICOR, and HermiT finds **425 of its
classes unsatisfiable**. Unaligned, the ontology is consistent and every class satisfiable.

## The five axioms they follow from

Each of these places a class into a DUL category disjoint with one its other ancestors
give it. Removing the five, and nothing else, leaves no CRAM-AICOR class unsatisfiable.

| Axiom | Clash | Classes it makes unsatisfiable |
|---|---|---|
| `ComputationalMedium ⊑ dul_InformationObject` (inherited through `Representation`) | with `dul_Description` | 214 |
| `ComputationalProcess ⊑ dul_Process` (an Event) | with `dul_Description` (an Object) | 18 |
| `ReifiedTemporalEntity ⊑ dul_InformationObject` | with `EventType` and `FluentType ⊑ dul_Description` | 40 |
| `ExecutiveCapability ⊑ dul_Quality` | with `ExecutiveMechanism` (Description) and `ComputationalProcess` | 133 |
| `TaskStatus ⊑ dul_InformationObject` | with `FluentType` | 20 |

The counts are those each removal repairs when they are removed one after another in this
order. Examples of affected classes: `Plan`, `Task`, `Intention`, `MotionPhase`,
`ManipulationPhase`, `EventType`, `FluentType`, `PlanExecutive`, `KnowledgeExecutive`,
`Monitoring`, `ForceDynamicEvent`.

## Reproducing

```text
python -m llm_reasoner.knowledge.admission.report
python -m llm_reasoner.knowledge.admission.report --leave-out test/llm_reasoner_test/linked_ontology_cases/upstream_root_causes.ttl
```

The first merges the pinned CRAM-AICOR and DUL, this repository's modules and its
alignment module (`llm_reasoner/src/llm_reasoner/knowledge/ontologies/alignment/dul-alignment.ttl`,
the 14 equivalences), and prints the unsatisfiable classes. The second leaves the five
axioms out.

## What this repository does meanwhile

Its own modules no longer give a class a DUL parent in a category CRAM-AICOR's parent
contradicts (2026-10-26). The open-world check of a generated hypothesis leaves the
alignment out until CRAM-AICOR is repaired, and the tests check the alignment with the five
axioms subtracted.
