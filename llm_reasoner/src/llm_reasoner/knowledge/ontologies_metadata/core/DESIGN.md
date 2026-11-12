> Import architecture updated in release 0.6.0: see `MODULAR_IMPORTS.md` at the package root. Earlier import-chain/open-file instructions below are superseded. Vocabulary descriptions still apply.

# Action-description core design

## Intended architecture

The shared representation is a versioned KnowledgeRecord expressing an ActionDescription. Annotators contribute claims about components of that same description; projections select and reshape those claims for consumers. This supports the proposed common-representation architecture without putting every conceivable fact inside one literal or requiring every perspective to be populated.

The core imports DUL only. The optional CRAM bridge imports the core and the existing AICOR model. A reusable core should not inherit CRAM-specific runtime restrictions merely because CRAM is one consumer.

## Three distinct entities

| Entity | DUL alignment | Meaning |
|---|---|---|
| ActionDescription | Description | The conceptual meaning of an action template |
| KnowledgeRecord | InformationObject | A versioned artifact expressing that meaning and recording claims/evidence |
| Actual action occurrence | DUL Action, outside the core examples | An event in a particular world or episode |

DUL's `expresses` relation connects the information artifact to its meaning. `ActionKind` specializes DUL Task; it is a task concept, not an event. `ParticipantRole` specializes DUL Role. EntitySpecification is a Description, so a bottle specification is not inferred to be a physical bottle. ParameterSpecification is also a Description; a named slot such as grip-force is not automatically a physical quality, quantity region or measured value.

The core explicitly distinguishes descriptions and records from events and physical objects using disjointness. Do not conflate an ActionDescription and KnowledgeRecord into one RDF individual.

## Shared descriptive components

ActionDescription can link to:

- Participant roles and entity specifications, preserving shared identity across perspectives.
- Goals, preconditions, success/failure criteria and constraints.
- Parameters, objectives and abstract capability requirements.
- Phases, subaction descriptions and recovery descriptions.
- Causal, physical-model and control specifications.
- PerspectiveDescription extensions for future frames, force dynamics and image schemas.

These are optional semantic components. The core does not require a robot, object pose, actuator, numeric force limit or simulation. A field being absent means unknown or unspecified, not false. The SHACL ingestion profile requires one action kind for an accepted record's description; OWL itself permits incomplete descriptions.

Phase ordering is a directed relation, not a temporal planner. Goals and conditions are descriptive content, not statements that a state holds. Lexical condition expressions may be stored with an expression language; this release does not interpret equations, conjunction, negation or temporal logic. Physical and causal specifications are extension points, not fluid solvers or causal identification algorithms.

## Claims and epistemic status

A KnowledgeClaim links a component, a reified Statement, an epistemic status, optional evidence and optional applicability conditions. Statement subject and object reference conceptual entities; a literal object is also supported. The predicate is an absolute IRI encoded as an anyURI literal. This avoids accidentally asserting the statement while retaining a queryable triple-shaped payload.

For example, a proposal relating a transfer-goal specification to a receiving-container specification remains a proposal. The corresponding RDF triple is not automatically materialized. No generated probability is inserted.

Statuses are Proposed, Supported, Disputed, Rejected and Unknown. They are workflow categories, not probabilities or universal truth labels. Supported requires an evidence record in the SHACL profile, but the validator cannot establish evidence quality or that evidence entails the claim. EvidenceKind distinguishes curated examples, LLM generation, documents, logical derivation, simulation and observation. A simulation result must retain its assumptions and scope in evidence/applicability records; merely attaching it does not validate deployment safety.

Direct ontology/template links describe the structure of a proposed model. They are not physical-world assertions. If alternative incompatible physical claims exist, retain separate claim/statement nodes with statuses; do not assert all alternatives as accepted physical facts.

## Annotators and updates

An AnnotationRecord identifies its producer, target description and contributed claims. Record-level SHACL checks prevent an annotation from contributing foreign claims or targeting another description. Annotation records describe contributions; the ontology does not execute annotators or provide a concurrent store.

Use a new KnowledgeRecord IRI for each revision and link it with previousVersion. Do not mutate an old record or its referenced claims/components after publishing a projection; otherwise projection provenance no longer identifies a reproducible snapshot. This immutability policy is an application responsibility, not an OWL guarantee. Shared reusable entities may remain shared; changed claim/component content needs new identifiers.

## Projections

ProjectionSpecification defines the projection key, optional output-schema IRI, required field declarations and missing-information policy. ProjectionArtifact links its source record and selected claims. SHACL rejects claims not present in the source record.

The included executable demo derives a goal view and a phase view from one pouring record. It preserves status/evidence references. It demonstrates selection and traceability, not a full Minsky frame schema or force-dynamic transformation.

ExecutionProjectionSpecification is an extension point. No generic projection is declared execution-ready. A future CoraPlex/SDT adapter must validate its own complete schema, bindings, capabilities, units, constraints and accepted evidence policy. Declared required fields and outputSchemaIRI are descriptive metadata in this release, not automatically enforced adapter contracts.

## World injection

BindingRecord optionally associates an EntitySpecification with a target world IRI and a context IRI. Keeping both as IRI-valued literals lets the core store external references without importing or instantiating a world ontology. The adapter must resolve these references; URI existence, context freshness and object identity are not checked here. No example contains a binding or actual robot/world entity.

## Action independence and examples

Pouring and grasping use the same generic schema. Each has its own action-kind concept, roles, goals and unspecified parameters. They are conceptual individuals (ABox examples describing stereotypes), not factual execution episodes. Their claims are Proposed and their evidence explicitly says they are synthetic schema examples.

A future action-domain module can define reusable specializations and stereotypical content on this core. No Pouring-specific class or property occurs in the core itself. An additional action need not modify the core unless it requires a genuinely new generic concept.

## Competency questions supported now

| Question | Representation/query route |
|---|---|
| What action kind does this record describe? | expressesDescription / hasActionKind |
| Which roles and entity specifications participate? | hasRoleSpecification / specifiesRole / hasEntitySpecification |
| What goals, phases and parameters are described? | component links and typed components |
| Which claims are proposed, disputed or supported? | hasClaim / hasStatus |
| What evidence and applicability accompany a claim? | hasEvidence / hasApplicability |
| Which annotator contributed a claim? | hasAnnotation / contributesClaim / producerIdentifier |
| Where did this projection obtain its claims? | derivedFromRecord / includesClaim |
| Which parameters remain unspecified? | parameterKey with no numericValue or bounds |

These support a foundation for queries at different competence levels. The ontology does not itself implement Bloom's taxonomy, prove a causal explanation, execute an action or answer arbitrary counterfactuals.

## Boundaries for the next increment

Next work should define one real perspective schema and its projection contract, then an extraction/annotation contract preserving sources, unknowns and claim identity. EQL integration, LLM transport, SDT bindings, simulation evidence acquisition, runtime readiness validation and full perspective ontologies remain future work. Do not treat this generic core as their implementation.
