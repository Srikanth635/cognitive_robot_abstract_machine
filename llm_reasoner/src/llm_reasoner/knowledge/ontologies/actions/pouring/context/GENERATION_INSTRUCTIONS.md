# Pouring generation instructions

Produce one proposed comprehensive pouring description using the existing
PouringDescription and Pouring action kind. The ontology is a schema, not a
populated knowledge base or evidence that an action occurred. The current bounded
pouring template describes partial transfer; do not require an empty source.

## Inputs and output

The caller separately provides the instruction, world/robot context, requested
knowledge aspects and required output serialization. This bundle supplies no
world state, physical numerical defaults or complete output JSON schema. If the
caller has not specified a serialization, use RDF Turtle with the supplied IRIs.
Generate a KnowledgeRecord with a version, an expressed action description,
consistent entity/parameter specifications and exact statement-level claims.
Use the symbolic template as a structural guide, not as already grounded facts.

Distinguish source container, destination container, material specification and
transferred portion. Reuse identities across fields, role bindings, conditions,
phases, monitoring, models and perspectives. World identifiers belong to injected
context/bindings; do not manufacture world instances. Use fresh record and mutable
annotation/context identities, with explicit versioned policy for shared reusable
specifications. Keep generated claims Proposed and retain their generation/source
provenance. Preserve incoming observation provenance without upgrading it.

## Conditions

Prerequisites use scoped TransferReady(source,destination,material), or explicit
PreState material-in-source containment and destination accessibility. Success
requires the intended transferred portion in the intended destination, delivered
amount >= target amount and spill <= tolerance. A goal may be weaker, but success
must establish every declared goal conjunct. Failure may use current TransferFault
for these endpoints or explicit post-state spill > the bound tolerance. Source
availability, target capacity and feasibility must be supplied or left unknown;
these contracts do not compute them. Tilting is not evidence of transfer success.

Use volume or mass and preserve quantity roles/units. The current bounded scalar
profile uses cubic metres or kilograms without implicit conversion; preserve the
original instruction quantity alongside any explicitly normalized value. Do not
invent mass, capacity, viscosity, friction, force limits, probabilities, flow rates,
controller settings or spill tolerances. Template scalar_default null is metadata,
not an RDF numeric value. Known equal numbers need not be the same specification.

Preserve PreState, CurrentState, PostState and TaskContext and any stage contexts.
Reject a conjunction containing an atom and its negation with identical arguments
and scope, or exact complementary comparisons with identical operands/scopes.
Different state scopes can legitimately disagree. Atomic roots stay atomic:
all-clauses describes combination, not a one-operand Conjunction constructor.

## Enrichment and boundaries

Add requested phases, sensory expectations, causal hypotheses, property/model
descriptions, control requirements, objectives, capabilities, diagnosis/recovery,
frames, force dynamics and image schemas to the same record. Missing values or
unsupported knowledge remain explicit unknowns; acknowledge unresolved references
or applicability and request evidence where needed. Imported vocabulary is
available for use, not a mandate to populate every aspect. Importing manipulation
or tool vocabulary does not mean a pour necessarily starts with pickup or requires
container contact. Navigation is outside this profile; request additional context
if base movement becomes necessary.

Qualitative causal hypotheses and model equations are not validated physical laws.
Keep proposed model/study descriptions separate from validation results. Supplied
criterion truth is not evidence acceptance, occurrence verification or execution
readiness. Fault absence is unknown unless explicitly assessed. No world belief
is published and no controller or action is executed by this context.

## Validation

Use the supplied SHACL graph with the supplied ontology for structural checks.
SHACL coverage is not complete semantic validation: predicate referents, phase
guards, bounded goal/success signatures, provenance and evidence acceptance still
need application checks. The original bounded inspectors remain in the full
development package. Do not evaluate arbitrary lexical expressions or execute
equation text. Validation failure calls for repair or abstention, not an assertion
that a real robot action failed. A consistent description is not physical proof.
