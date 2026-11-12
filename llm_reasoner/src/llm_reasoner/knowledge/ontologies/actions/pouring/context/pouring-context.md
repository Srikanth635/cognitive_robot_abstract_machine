# Self-contained pouring action context — 0.23.0

Use this document as schema/generation context. The task instruction and world/robot snapshot must be supplied separately. No imports or other attachments are required to read its ontology and template.

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

## Pouring symbolic template

```json
{
  "bundle_version": "0.23.0",
  "source_template_version": "0.22.0",
  "focus_description_class": "http://aicor.knowledge/ontologies/material-transfer.owl#PouringDescription",
  "artifact_kind": "symbolic generation aid",
  "complete_output_schema": false,
  "new_record_identity_policy": "Use a fresh record and mutable annotation/context identities; reuse immutable descriptions/specifications only through an explicit versioned sharing policy.",
  "unknown_value_policy": "Leave physical values unspecified; scalar_default null is template metadata, never an RDF numeric literal.",
  "epistemic_policy": "Generated claims remain Proposed; no observations, world bindings or accepted beliefs are produced.",
  "template": {
    "action": "pour",
    "action_kind": "http://aicor.knowledge/ontologies/material-transfer.owl#Pouring",
    "entity_roles": [
      "destination",
      "material",
      "source",
      "transferred_portion"
    ],
    "parameters": {
      "delivered_amount": {
        "unit_iri": "http://qudt.org/vocab/unit/M3",
        "scalar_default": null
      },
      "target_amount": {
        "unit_iri": "http://qudt.org/vocab/unit/M3",
        "scalar_default": null
      },
      "spill_amount": {
        "unit_iri": "http://qudt.org/vocab/unit/M3",
        "scalar_default": null
      },
      "spill_tolerance": {
        "unit_iri": "http://qudt.org/vocab/unit/M3",
        "scalar_default": null
      }
    },
    "conditions": {
      "precondition": {
        "combination": "all-clauses",
        "root_expression_type": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#AtomicCondition",
        "atoms": [
          {
            "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#PreState",
            "relation": "http://aicor.knowledge/ontologies/material-transfer.owl#TransferReady",
            "arguments": [
              "entity:source",
              "entity:destination",
              "entity:material"
            ],
            "negated": false
          }
        ],
        "comparisons": []
      },
      "goal": {
        "combination": "all-clauses",
        "root_expression_type": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#Conjunction",
        "atoms": [
          {
            "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#PostState",
            "relation": "http://aicor.knowledge/ontologies/participants-relations.owl#ContainedIn",
            "arguments": [
              "entity:transferred_portion",
              "entity:destination"
            ],
            "negated": false
          }
        ],
        "comparisons": [
          {
            "operator": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#GreaterOrEqual",
            "left": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#PostState",
              "parameter": "parameter:delivered_amount"
            },
            "right": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#TaskContext",
              "parameter": "parameter:target_amount"
            }
          },
          {
            "operator": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#LessOrEqual",
            "left": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#PostState",
              "parameter": "parameter:spill_amount"
            },
            "right": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#TaskContext",
              "parameter": "parameter:spill_tolerance"
            }
          }
        ]
      },
      "success": {
        "combination": "all-clauses",
        "root_expression_type": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#Conjunction",
        "atoms": [
          {
            "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#PostState",
            "relation": "http://aicor.knowledge/ontologies/participants-relations.owl#ContainedIn",
            "arguments": [
              "entity:transferred_portion",
              "entity:destination"
            ],
            "negated": false
          }
        ],
        "comparisons": [
          {
            "operator": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#GreaterOrEqual",
            "left": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#PostState",
              "parameter": "parameter:delivered_amount"
            },
            "right": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#TaskContext",
              "parameter": "parameter:target_amount"
            }
          },
          {
            "operator": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#LessOrEqual",
            "left": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#PostState",
              "parameter": "parameter:spill_amount"
            },
            "right": {
              "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#TaskContext",
              "parameter": "parameter:spill_tolerance"
            }
          }
        ]
      },
      "fault": {
        "combination": "all-clauses",
        "root_expression_type": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#AtomicCondition",
        "atoms": [
          {
            "scope": "http://aicor.knowledge/ontologies/conditions-outcomes.owl#CurrentState",
            "relation": "http://aicor.knowledge/ontologies/material-transfer.owl#TransferFault",
            "arguments": [
              "entity:source",
              "entity:destination"
            ],
            "negated": false
          }
        ],
        "comparisons": []
      }
    }
  }
}
```

## Complete ontology reference

The following is reference data with original shared concept IRIs. It is not a world snapshot or a populated pouring episode.

```turtle
@prefix ac: <http://aicor.knowledge/ontologies/action-description-core.owl#> .
@prefix mt: <http://aicor.knowledge/ontologies/material-transfer.owl#> .
@prefix n0: <http://aicor.knowledge/ontologies/> .
@prefix n1: <http://aicor.knowledge/ontologies/capabilities-strategies.owl#> .
@prefix n10: <http://aicor.knowledge/ontologies/image-schemas.owl#> .
@prefix n11: <http://aicor.knowledge/ontologies/knowledge-context.owl#> .
@prefix n12: <http://aicor.knowledge/ontologies/model-contracts.owl#> .
@prefix n13: <http://aicor.knowledge/ontologies/monitoring-requirements.owl#> .
@prefix n14: <http://aicor.knowledge/ontologies/object-manipulation.owl#> .
@prefix n15: <http://aicor.knowledge/ontologies/objectives-tradeoffs.owl#> .
@prefix n16: <http://aicor.knowledge/ontologies/participants-relations.owl#> .
@prefix n17: <http://aicor.knowledge/ontologies/phases-transitions.owl#> .
@prefix n18: <http://aicor.knowledge/ontologies/physical-models.owl#> .
@prefix n19: <http://aicor.knowledge/ontologies/robot-motion.owl#> .
@prefix n2: <http://aicor.knowledge/ontologies/causal-mechanisms.owl#> .
@prefix n20: <http://aicor.knowledge/ontologies/sensorimotor-events.owl#> .
@prefix n21: <http://aicor.knowledge/ontologies/tool-interaction.owl#> .
@prefix n22: <http://aicor.knowledge/ontologies/transfer-contracts.owl#> .
@prefix n23: <http://www.ontologydesignpatterns.org/ont/dul/> .
@prefix n24: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> .
@prefix n3: <http://aicor.knowledge/ontologies/conditions-outcomes.owl#> .
@prefix n4: <http://aicor.knowledge/ontologies/diagnostic-causal-bridge.owl#> .
@prefix n5: <http://aicor.knowledge/ontologies/dynamics-control.owl#> .
@prefix n6: <http://aicor.knowledge/ontologies/entities-materials-affordances.owl#> .
@prefix n7: <http://aicor.knowledge/ontologies/failure-diagnosis.owl#> .
@prefix n8: <http://aicor.knowledge/ontologies/force-dynamics.owl#> .
@prefix n9: <http://aicor.knowledge/ontologies/frame-perspective.owl#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ac:ExecutionProjectionSpecification a owl:Class ;
    rdfs:label "Execution Projection Specification"@en ;
    rdfs:comment "Contract intended for an execution adapter; readiness must be separately checked." ;
    rdfs:subClassOf ac:ProjectionSpecification .

ac:aboutComponent a owl:ObjectProperty ;
    rdfs:label "About Component"@en ;
    rdfs:domain ac:KnowledgeClaim ;
    rdfs:range ac:ActionComponent ;
    rdfs:subPropertyOf n24:isAbout .

ac:annotatesDescription a owl:ObjectProperty ;
    rdfs:label "Annotates Description"@en ;
    rdfs:domain ac:AnnotationRecord ;
    rdfs:range ac:ActionDescription ;
    rdfs:subPropertyOf n24:isAbout .

ac:bindingContextIRI a owl:DatatypeProperty ;
    rdfs:label "Binding Context IRI"@en ;
    rdfs:domain ac:BindingRecord ;
    rdfs:range xsd:anyURI .

ac:bindingTargetIRI a owl:DatatypeProperty ;
    rdfs:label "Binding Target IRI"@en ;
    rdfs:domain ac:BindingRecord ;
    rdfs:range xsd:anyURI .

ac:bindsSpecification a owl:ObjectProperty ;
    rdfs:label "Binds Specification"@en ;
    rdfs:domain ac:BindingRecord ;
    rdfs:range ac:EntitySpecification .

ac:conformsToSpecification a owl:ObjectProperty ;
    rdfs:label "Conforms To Specification"@en ;
    rdfs:domain ac:ProjectionArtifact ;
    rdfs:range ac:ProjectionSpecification .

ac:contributesClaim a owl:ObjectProperty ;
    rdfs:label "Contributes Claim"@en ;
    rdfs:domain ac:AnnotationRecord ;
    rdfs:range ac:KnowledgeClaim .

ac:createdAt a owl:DatatypeProperty ;
    rdfs:label "Created At"@en ;
    rdfs:domain ac:AnnotationRecord ;
    rdfs:range xsd:dateTime .

ac:derivedFromRecord a owl:ObjectProperty ;
    rdfs:label "Derived From Record"@en ;
    rdfs:domain ac:ProjectionArtifact ;
    rdfs:range ac:KnowledgeRecord .

ac:evidenceKind a owl:ObjectProperty ;
    rdfs:label "Evidence Kind"@en ;
    rdfs:domain ac:EvidenceRecord ;
    rdfs:range ac:EvidenceKind .

ac:evidenceText a owl:DatatypeProperty ;
    rdfs:label "Evidence Text"@en ;
    rdfs:domain ac:EvidenceRecord ;
    rdfs:range xsd:string .

ac:expressesDescription a owl:ObjectProperty ;
    rdfs:label "Expresses Description"@en ;
    rdfs:domain ac:KnowledgeRecord ;
    rdfs:range ac:ActionDescription ;
    rdfs:subPropertyOf n24:expresses .

ac:expressionLanguage a owl:DatatypeProperty ;
    rdfs:label "Expression Language"@en ;
    rdfs:domain ac:ConditionSpecification ;
    rdfs:range xsd:string .

ac:hasActionKind a owl:ObjectProperty ;
    rdfs:label "Has Action Kind"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:ActionKind ;
    rdfs:subPropertyOf n24:defines .

ac:hasAnnotation a owl:ObjectProperty ;
    rdfs:label "Has Annotation"@en ;
    rdfs:domain ac:KnowledgeRecord ;
    rdfs:range ac:AnnotationRecord .

ac:hasApplicability a owl:ObjectProperty ;
    rdfs:label "Has Applicability"@en ;
    rdfs:domain ac:KnowledgeClaim ;
    rdfs:range ac:ApplicabilitySpecification .

ac:hasBinding a owl:ObjectProperty ;
    rdfs:label "Has Binding"@en ;
    rdfs:domain ac:KnowledgeRecord ;
    rdfs:range ac:BindingRecord .

ac:hasCapabilityRequirement a owl:ObjectProperty ;
    rdfs:label "Has Capability Requirement"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:CapabilityRequirement ;
    rdfs:subPropertyOf ac:hasComponent,
        ac:requiresCapability .

ac:hasCausalModel a owl:ObjectProperty ;
    rdfs:label "Has Causal Model"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:CausalSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasClaim a owl:ObjectProperty ;
    rdfs:label "Has Claim"@en ;
    rdfs:domain ac:KnowledgeRecord ;
    rdfs:range ac:KnowledgeClaim .

ac:hasConstraint a owl:ObjectProperty ;
    rdfs:label "Has Constraint"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:ConstraintSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasControl a owl:ObjectProperty ;
    rdfs:label "Has Control"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:ControlSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasEntitySpecification a owl:ObjectProperty ;
    rdfs:label "Has Entity Specification"@en ;
    rdfs:domain ac:RoleSpecification ;
    rdfs:range ac:EntitySpecification .

ac:hasEvidence a owl:ObjectProperty ;
    rdfs:label "Has Evidence"@en ;
    rdfs:domain ac:KnowledgeClaim ;
    rdfs:range ac:EvidenceRecord .

ac:hasFailureCriterion a owl:ObjectProperty ;
    rdfs:label "Has Failure Criterion"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:FailureCriterion ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasGoal a owl:ObjectProperty ;
    rdfs:label "Has Goal"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:GoalSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasObjective a owl:ObjectProperty ;
    rdfs:label "Has Objective"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:ObjectiveSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasParameter a owl:ObjectProperty ;
    rdfs:label "Has Parameter"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:ParameterSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasPerspective a owl:ObjectProperty ;
    rdfs:label "Has Perspective"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:PerspectiveDescription ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasPhase a owl:ObjectProperty ;
    rdfs:label "Has Phase"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:PhaseSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasPhysicalModel a owl:ObjectProperty ;
    rdfs:label "Has Physical Model"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:PhysicalModelSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasPrecondition a owl:ObjectProperty ;
    rdfs:label "Has Precondition"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:PreconditionSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasRecovery a owl:ObjectProperty ;
    rdfs:label "Has Recovery"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:RecoverySpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasRoleSpecification a owl:ObjectProperty ;
    rdfs:label "Has Role Specification"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:RoleSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

ac:hasStatement a owl:ObjectProperty ;
    rdfs:label "Has Statement"@en ;
    rdfs:domain ac:KnowledgeClaim ;
    rdfs:range ac:Statement .

ac:hasStatus a owl:ObjectProperty ;
    rdfs:label "Has Status"@en ;
    rdfs:domain ac:KnowledgeClaim ;
    rdfs:range ac:EpistemicStatus .

ac:hasSubActionDescription a owl:ObjectProperty ;
    rdfs:label "Has Sub Action Description"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:ActionDescription .

ac:hasSuccessCriterion a owl:ObjectProperty ;
    rdfs:label "Has Success Criterion"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:SuccessCriterion ;
    rdfs:subPropertyOf ac:hasComponent .

ac:includesClaim a owl:ObjectProperty ;
    rdfs:label "Includes Claim"@en ;
    rdfs:domain ac:ProjectionArtifact ;
    rdfs:range ac:KnowledgeClaim .

ac:lexicalExpression a owl:DatatypeProperty ;
    rdfs:label "Lexical Expression"@en ;
    rdfs:domain ac:ConditionSpecification ;
    rdfs:range xsd:string .

ac:lowerBound a owl:DatatypeProperty ;
    rdfs:label "Lower Bound"@en ;
    rdfs:domain ac:ParameterSpecification ;
    rdfs:range xsd:decimal .

ac:missingInformationPolicy a owl:DatatypeProperty ;
    rdfs:label "Missing Information Policy"@en ;
    rdfs:domain ac:ProjectionSpecification ;
    rdfs:range xsd:string .

ac:numericValue a owl:DatatypeProperty ;
    rdfs:label "Numeric Value"@en ;
    rdfs:domain ac:ParameterSpecification ;
    rdfs:range xsd:decimal .

ac:objectiveDirection a owl:DatatypeProperty ;
    rdfs:label "Objective Direction"@en ;
    rdfs:domain ac:ObjectiveSpecification ;
    rdfs:range xsd:string .

ac:outputSchemaIRI a owl:DatatypeProperty ;
    rdfs:label "Output Schema IRI"@en ;
    rdfs:domain ac:ProjectionSpecification ;
    rdfs:range xsd:anyURI .

ac:parameterKey a owl:DatatypeProperty ;
    rdfs:label "Parameter Key"@en ;
    rdfs:domain ac:ParameterSpecification ;
    rdfs:range xsd:string .

ac:precedesPhase a owl:ObjectProperty ;
    rdfs:label "Precedes Phase"@en ;
    rdfs:domain ac:PhaseSpecification ;
    rdfs:range ac:PhaseSpecification .

ac:previousVersion a owl:ObjectProperty ;
    rdfs:label "Previous Version"@en ;
    rdfs:domain ac:KnowledgeRecord ;
    rdfs:range ac:KnowledgeRecord .

ac:priority a owl:DatatypeProperty ;
    rdfs:label "Priority"@en ;
    rdfs:domain ac:ObjectiveSpecification ;
    rdfs:range xsd:integer .

ac:producerIdentifier a owl:DatatypeProperty ;
    rdfs:label "Producer Identifier"@en ;
    rdfs:domain ac:AnnotationRecord ;
    rdfs:range xsd:string .

ac:projectionKey a owl:DatatypeProperty ;
    rdfs:label "Projection Key"@en ;
    rdfs:domain ac:ProjectionSpecification ;
    rdfs:range xsd:string .

ac:recordVersion a owl:DatatypeProperty ;
    rdfs:label "Record Version"@en ;
    rdfs:domain ac:KnowledgeRecord ;
    rdfs:range xsd:string .

ac:requiredField a owl:DatatypeProperty ;
    rdfs:label "Required Field"@en ;
    rdfs:domain ac:ProjectionSpecification ;
    rdfs:range xsd:string .

ac:sourceIRI a owl:DatatypeProperty ;
    rdfs:label "Source IRI"@en ;
    rdfs:domain ac:EvidenceRecord ;
    rdfs:range xsd:anyURI .

ac:specifiesRole a owl:ObjectProperty ;
    rdfs:label "Specifies Role"@en ;
    rdfs:domain ac:RoleSpecification ;
    rdfs:range ac:ParticipantRole .

ac:statementLiteral a owl:DatatypeProperty ;
    rdfs:label "Statement Literal"@en ;
    rdfs:domain ac:Statement ;
    rdfs:range rdfs:Literal .

ac:statementObject a owl:ObjectProperty ;
    rdfs:label "Statement Object"@en ;
    rdfs:domain ac:Statement ;
    rdfs:range n24:Entity .

ac:statementPredicateIRI a owl:DatatypeProperty ;
    rdfs:label "Statement Predicate IRI"@en ;
    rdfs:domain ac:Statement ;
    rdfs:range xsd:anyURI .

ac:statementSubject a owl:ObjectProperty ;
    rdfs:label "Statement Subject"@en ;
    rdfs:domain ac:Statement ;
    rdfs:range n24:Entity .

ac:unitIRI a owl:DatatypeProperty ;
    rdfs:label "Unit IRI"@en ;
    rdfs:domain ac:ParameterSpecification ;
    rdfs:range xsd:anyURI .

ac:upperBound a owl:DatatypeProperty ;
    rdfs:label "Upper Bound"@en ;
    rdfs:domain ac:ParameterSpecification ;
    rdfs:range xsd:decimal .

n1:Met a n1:AssessmentOutcome,
        owl:NamedIndividual ;
    rdfs:label "Met"@en .

n1:Undetermined a n1:AssessmentOutcome,
        owl:NamedIndividual ;
    rdfs:label "Undetermined"@en .

n1:Unmet a n1:AssessmentOutcome,
        owl:NamedIndividual ;
    rdfs:label "Unmet"@en .

n1:appliesDuring a owl:ObjectProperty ;
    rdfs:label "Applies During"@en ;
    rdfs:comment "Optional reference to a core phase specification."@en ;
    rdfs:domain n1:StrategyStep ;
    rdfs:range ac:PhaseSpecification .

n1:assessesRequirement a owl:ObjectProperty ;
    rdfs:label "Assesses Requirement"@en ;
    rdfs:comment "Requirement under assessment."@en ;
    rdfs:domain n1:CapabilityAssessment ;
    rdfs:range ac:CapabilityRequirement .

n1:assessmentContext a owl:ObjectProperty ;
    rdfs:label "Assessment Context"@en ;
    rdfs:comment "Scope of the assessment."@en ;
    rdfs:domain n1:CapabilityAssessment ;
    rdfs:range ac:ApplicabilitySpecification .

n1:assessmentEvidence a owl:ObjectProperty ;
    rdfs:label "Assessment Evidence"@en ;
    rdfs:comment "Evidence for a met or unmet outcome."@en ;
    rdfs:domain n1:CapabilityAssessment ;
    rdfs:range ac:EvidenceRecord .

n1:assessmentOutcome a owl:ObjectProperty ;
    rdfs:label "Assessment Outcome"@en ;
    rdfs:comment "Context-specific assessment outcome."@en ;
    rdfs:domain n1:CapabilityAssessment ;
    rdfs:range n1:AssessmentOutcome .

n1:completionCondition a owl:ObjectProperty ;
    rdfs:label "Completion Condition"@en ;
    rdfs:comment "Descriptive completion requirement."@en ;
    rdfs:domain n1:StrategyStep ;
    rdfs:range ac:ConditionSpecification .

n1:describesAction a owl:ObjectProperty ;
    rdfs:label "Describes Action"@en ;
    rdfs:comment "Conceptual action performed by the step."@en ;
    rdfs:domain n1:StrategyStep ;
    rdfs:range ac:ActionDescription .

n1:fulfillsRequirement a owl:ObjectProperty ;
    rdfs:label "Fulfills Requirement"@en ;
    rdfs:comment "Capability requirement this interface is intended to fulfill."@en ;
    rdfs:domain n1:ProcessModuleRequirement ;
    rdfs:range ac:CapabilityRequirement .

n1:hasCapabilityProfile a owl:ObjectProperty ;
    rdfs:label "Has Capability Profile"@en ;
    rdfs:comment "Capability requirements for this action."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n1:CapabilityProfile ;
    rdfs:subPropertyOf ac:hasComponent .

n1:hasStep a owl:ObjectProperty ;
    rdfs:label "Has Step"@en ;
    rdfs:comment "Steps owned by a strategy."@en ;
    rdfs:domain n1:ExecutionStrategy ;
    rdfs:range n1:StrategyStep .

n1:hasStrategy a owl:ObjectProperty ;
    rdfs:label "Has Strategy"@en ;
    rdfs:comment "Candidate strategy."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n1:ExecutionStrategy ;
    rdfs:subPropertyOf ac:hasComponent .

n1:inputParameter a owl:ObjectProperty ;
    rdfs:label "Input Parameter"@en ;
    rdfs:comment "Required abstract input."@en ;
    rdfs:domain n1:ProcessModuleRequirement ;
    rdfs:range ac:ParameterSpecification .

n1:interfaceKey a owl:DatatypeProperty ;
    rdfs:label "Interface Key"@en ;
    rdfs:comment "Stable abstract interface identifier, not a runtime endpoint."@en ;
    rdfs:domain n1:ProcessModuleRequirement ;
    rdfs:range xsd:string .

n1:outputParameter a owl:ObjectProperty ;
    rdfs:label "Output Parameter"@en ;
    rdfs:comment "Declared abstract output."@en ;
    rdfs:domain n1:ProcessModuleRequirement ;
    rdfs:range ac:ParameterSpecification .

n1:precedesStep a owl:ObjectProperty ;
    rdfs:label "Precedes Step"@en ;
    rdfs:comment "Partial-order precedence, not a timing guarantee."@en ;
    rdfs:domain n1:StrategyStep ;
    rdfs:range n1:StrategyStep .

n1:precondition a owl:ObjectProperty ;
    rdfs:label "Precondition"@en ;
    rdfs:comment "Descriptive step precondition."@en ;
    rdfs:domain n1:StrategyStep ;
    rdfs:range ac:ConditionSpecification .

n1:requiresCapability a owl:ObjectProperty ;
    rdfs:label "Requires Capability"@en ;
    rdfs:comment "Required abstract capability."@en ;
    rdfs:domain n1:CapabilityProfile ;
    rdfs:range ac:CapabilityRequirement ;
    rdfs:subPropertyOf ac:requiresCapability .

n1:requiresProcess a owl:ObjectProperty ;
    rdfs:label "Requires Process"@en ;
    rdfs:comment "Required process interface."@en ;
    rdfs:domain n1:StrategyStep ;
    rdfs:range n1:ProcessModuleRequirement .

n2:antecedent a owl:ObjectProperty ;
    rdfs:label "Antecedent"@en ;
    rdfs:comment "Described cause condition."@en ;
    rdfs:domain n2:CausalLink ;
    rdfs:range ac:ConditionSpecification .

n2:applicability a owl:ObjectProperty ;
    rdfs:label "Applicability"@en ;
    rdfs:comment "Domain of applicability, not a universal guarantee."@en ;
    rdfs:domain n2:MechanismSpecification ;
    rdfs:range ac:ApplicabilitySpecification .

n2:appliesDuring a owl:ObjectProperty ;
    rdfs:label "Applies During"@en ;
    rdfs:comment "Optional phase context."@en ;
    rdfs:domain n2:CausalLink ;
    rdfs:range ac:PhaseSpecification .

n2:assumption a owl:ObjectProperty ;
    rdfs:label "Assumption"@en ;
    rdfs:comment "Assumption of this mechanism."@en ;
    rdfs:domain n2:MechanismSpecification ;
    rdfs:range ac:ConditionSpecification .

n2:consequent a owl:ObjectProperty ;
    rdfs:label "Consequent"@en ;
    rdfs:comment "Described effect condition."@en ;
    rdfs:domain n2:CausalLink ;
    rdfs:range ac:ConditionSpecification .

n2:enablingCondition a owl:ObjectProperty ;
    rdfs:label "Enabling Condition"@en ;
    rdfs:comment "Explicit enabling conditions."@en ;
    rdfs:domain n2:CausalLink ;
    rdfs:range n2:EnablingCondition .

n2:hasLink a owl:ObjectProperty ;
    rdfs:label "Has Link"@en ;
    rdfs:comment "Links belonging to this proposed model."@en ;
    rdfs:domain n2:CausalModel ;
    rdfs:range n2:CausalLink .

n2:hasSuppression a owl:ObjectProperty ;
    rdfs:label "Has Suppression"@en ;
    rdfs:comment "Suppression hypotheses in the model."@en ;
    rdfs:domain n2:CausalModel ;
    rdfs:range n2:SuppressionSpecification .

n2:intendedOutcome a owl:ObjectProperty ;
    rdfs:label "Intended Outcome"@en ;
    rdfs:comment "Desired intervention outcome."@en ;
    rdfs:domain n2:InterventionSpecification ;
    rdfs:range n3:OutcomeSpecification .

n2:intervention a owl:ObjectProperty ;
    rdfs:label "Intervention"@en ;
    rdfs:comment "Proposed suppressing intervention."@en ;
    rdfs:domain n2:SuppressionSpecification ;
    rdfs:range n2:InterventionSpecification .

n2:mechanism a owl:ObjectProperty ;
    rdfs:label "Mechanism"@en ;
    rdfs:comment "Proposed explanatory mechanism."@en ;
    rdfs:domain n2:CausalLink ;
    rdfs:range n2:MechanismSpecification .

n2:precedesLink a owl:ObjectProperty ;
    rdfs:label "Precedes Link"@en ;
    rdfs:comment "Explanatory chain order; does not entail transitive causation."@en ;
    rdfs:domain n2:CausalLink ;
    rdfs:range n2:CausalLink .

n2:suppressedLink a owl:ObjectProperty ;
    rdfs:label "Suppressed Link"@en ;
    rdfs:comment "Target causal hypothesis."@en ;
    rdfs:domain n2:SuppressionSpecification ;
    rdfs:range n2:CausalLink .

n3:CurrentState a n3:StateScope,
        owl:NamedIndividual ;
    rdfs:label "Current State"@en .

n3:Equal a n3:ComparisonOperator,
        owl:NamedIndividual ;
    rdfs:label "Equal"@en .

n3:GreaterOrEqual a n3:ComparisonOperator,
        owl:NamedIndividual ;
    rdfs:label "Greater Or Equal"@en .

n3:GreaterThan a n3:ComparisonOperator,
        owl:NamedIndividual ;
    rdfs:label "Greater Than"@en .

n3:LessOrEqual a n3:ComparisonOperator,
        owl:NamedIndividual ;
    rdfs:label "Less Or Equal"@en .

n3:LessThan a n3:ComparisonOperator,
        owl:NamedIndividual ;
    rdfs:label "Less Than"@en .

n3:NotEqual a n3:ComparisonOperator,
        owl:NamedIndividual ;
    rdfs:label "Not Equal"@en .

n3:PostState a n3:StateScope,
        owl:NamedIndividual ;
    rdfs:label "Post State"@en .

n3:PreState a n3:StateScope,
        owl:NamedIndividual ;
    rdfs:label "Pre State"@en .

n3:TaskContext a n3:StateScope,
        owl:NamedIndividual ;
    rdfs:label "Task Context"@en .

n3:afterCondition a owl:ObjectProperty ;
    rdfs:label "After Condition"@en ;
    rdfs:comment "Condition after the described transition."@en ;
    rdfs:domain n3:OutcomeSpecification ;
    rdfs:range ac:ConditionSpecification .

n3:beforeCondition a owl:ObjectProperty ;
    rdfs:label "Before Condition"@en ;
    rdfs:comment "Condition before the described transition."@en ;
    rdfs:domain n3:OutcomeSpecification ;
    rdfs:range ac:ConditionSpecification .

n3:decimalValue a owl:DatatypeProperty ;
    rdfs:label "Decimal Value"@en ;
    rdfs:comment "Constant decimal value."@en ;
    rdfs:domain n3:QuantityConstant ;
    rdfs:range xsd:decimal .

n3:hasExpression a owl:ObjectProperty ;
    rdfs:label "Has Expression"@en ;
    rdfs:comment "Root expression for a condition specification."@en ;
    rdfs:domain ac:ConditionSpecification ;
    rdfs:range n3:BooleanExpression .

n3:hasOperand a owl:ObjectProperty ;
    rdfs:label "Has Operand"@en ;
    rdfs:comment "Child expression. Arity is constrained by the concrete expression type."@en ;
    rdfs:domain n3:BooleanExpression ;
    rdfs:range n3:BooleanExpression .

n3:hasOutcome a owl:ObjectProperty ;
    rdfs:label "Has Outcome"@en ;
    rdfs:comment "Transition specification for an action."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n3:OutcomeSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

n3:hasPattern a owl:ObjectProperty ;
    rdfs:label "Has Pattern"@en ;
    rdfs:comment "Relation pattern whose truth is requested."@en ;
    rdfs:domain n3:AtomicCondition ;
    rdfs:range n16:RelationPattern .

n3:inScope a owl:ObjectProperty ;
    rdfs:label "In Scope"@en ;
    rdfs:comment "Scope of an atomic condition or parameter reference."@en ;
    rdfs:domain n3:Expression ;
    rdfs:range n3:StateScope .

n3:leftOperand a owl:ObjectProperty ;
    rdfs:label "Left Operand"@en ;
    rdfs:comment "Left-hand numeric expression."@en ;
    rdfs:domain n3:Comparison ;
    rdfs:range n3:ValueExpression .

n3:operator a owl:ObjectProperty ;
    rdfs:label "Operator"@en ;
    rdfs:comment "Numeric comparison operator."@en ;
    rdfs:domain n3:Comparison ;
    rdfs:range n3:ComparisonOperator .

n3:parameter a owl:ObjectProperty ;
    rdfs:label "Parameter"@en ;
    rdfs:comment "Referenced parameter specification."@en ;
    rdfs:domain n3:ParameterReference ;
    rdfs:range ac:ParameterSpecification .

n3:rightOperand a owl:ObjectProperty ;
    rdfs:label "Right Operand"@en ;
    rdfs:comment "Right-hand numeric expression."@en ;
    rdfs:domain n3:Comparison ;
    rdfs:range n3:ValueExpression .

n3:unitIRI a owl:DatatypeProperty ;
    rdfs:label "Unit IRI"@en ;
    rdfs:comment "Unit identifier. No implicit conversion."@en ;
    rdfs:domain n3:QuantityConstant ;
    rdfs:range xsd:anyURI .

n4:applicability a owl:ObjectProperty ;
    rdfs:label "Applicability"@en ;
    rdfs:domain n4:DiagnosticCausalAssociation ;
    rdfs:range ac:ApplicabilitySpecification .

n4:causalLink a owl:ObjectProperty ;
    rdfs:label "Causal link"@en ;
    rdfs:domain n4:DiagnosticCausalAssociation ;
    rdfs:range n2:CausalLink .

n4:causeCondition a owl:ObjectProperty ;
    rdfs:label "Cause condition"@en ;
    rdfs:domain n4:DiagnosticCausalAssociation ;
    rdfs:range ac:ConditionSpecification .

n4:hypothesis a owl:ObjectProperty ;
    rdfs:label "Hypothesis"@en ;
    rdfs:domain n4:DiagnosticCausalAssociation ;
    rdfs:range n7:DiagnosticHypothesis .

n5:Disturbance a n5:VariableRole,
        owl:NamedIndividual ;
    rdfs:label "Disturbance"@en .

n5:FeedbackControl a n5:ControllerKind,
        owl:NamedIndividual ;
    rdfs:label "Feedback Control"@en .

n5:Input a n5:VariableRole,
        owl:NamedIndividual ;
    rdfs:label "Input"@en .

n5:ModelPredictiveControl a n5:ControllerKind,
        owl:NamedIndividual ;
    rdfs:label "Model Predictive Control"@en .

n5:Output a n5:VariableRole,
        owl:NamedIndividual ;
    rdfs:label "Output"@en .

n5:State a n5:VariableRole,
        owl:NamedIndividual ;
    rdfs:label "State"@en .

n5:controllerKind a owl:ObjectProperty ;
    rdfs:label "Controller Kind"@en ;
    rdfs:comment "Declared controller family."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range n5:ControllerKind .

n5:controlsModel a owl:ObjectProperty ;
    rdfs:label "Controls Model"@en ;
    rdfs:comment "Model used by this proposed controller."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range n5:DynamicalModelSpecification .

n5:describesPlant a owl:ObjectProperty ;
    rdfs:label "Describes Plant"@en ;
    rdfs:comment "Existing physical model refined by this dynamic specification."@en ;
    rdfs:domain n5:DynamicalModelSpecification ;
    rdfs:range n18:PhysicalModelSpecification .

n5:dimension a owl:DatatypeProperty ;
    rdfs:label "Dimension"@en ;
    rdfs:comment "Number of scalar components in this entry."@en ;
    rdfs:domain n5:VariableAssignment ;
    rdfs:range xsd:positiveInteger .

n5:evolutionEquation a owl:ObjectProperty ;
    rdfs:label "Evolution Equation"@en ;
    rdfs:comment "State-evolution equation, also declared as a model equation."@en ;
    rdfs:domain n5:DynamicalModelSpecification ;
    rdfs:range n18:ModelEquation .

n5:hasAdaptation a owl:ObjectProperty ;
    rdfs:label "Has Adaptation"@en ;
    rdfs:comment "Proposed parameter adaptation."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range n5:AdaptationSpecification .

n5:hasAssignment a owl:ObjectProperty ;
    rdfs:label "Has Assignment"@en ;
    rdfs:comment "Vector entries of this model."@en ;
    rdfs:domain n5:DynamicalModelSpecification ;
    rdfs:range n5:VariableAssignment .

n5:inputConstraint a owl:ObjectProperty ;
    rdfs:label "Input Constraint"@en ;
    rdfs:comment "Input constraint specification."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range ac:ConstraintSpecification .

n5:objectiveSet a owl:ObjectProperty ;
    rdfs:label "Objective Set"@en ;
    rdfs:comment "Objective and hard-constraint configuration."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range n15:ObjectiveSet .

n5:outputEquation a owl:ObjectProperty ;
    rdfs:label "Output Equation"@en ;
    rdfs:comment "Output equation, also declared as a model equation."@en ;
    rdfs:domain n5:DynamicalModelSpecification ;
    rdfs:range n18:ModelEquation .

n5:position a owl:DatatypeProperty ;
    rdfs:label "Position"@en ;
    rdfs:comment "One-based position among vector entries."@en ;
    rdfs:domain n5:VariableAssignment ;
    rdfs:range xsd:positiveInteger .

n5:predictionHorizon a owl:ObjectProperty ;
    rdfs:label "Prediction Horizon"@en ;
    rdfs:comment "Optional time parameter."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range ac:ParameterSpecification .

n5:role a owl:ObjectProperty ;
    rdfs:label "Role"@en ;
    rdfs:comment "Role of the vector entry."@en ;
    rdfs:domain n5:VariableAssignment ;
    rdfs:range n5:VariableRole .

n5:samplingPeriod a owl:ObjectProperty ;
    rdfs:label "Sampling Period"@en ;
    rdfs:comment "Time parameter; no numeric default."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range ac:ParameterSpecification .

n5:stateConstraint a owl:ObjectProperty ;
    rdfs:label "State Constraint"@en ;
    rdfs:comment "State constraint specification."@en ;
    rdfs:domain n5:ControllerSpecification ;
    rdfs:range ac:ConstraintSpecification .

n5:updatesParameter a owl:ObjectProperty ;
    rdfs:label "Updates Parameter"@en ;
    rdfs:comment "Parameter to estimate or update."@en ;
    rdfs:domain n5:AdaptationSpecification ;
    rdfs:range ac:ParameterSpecification .

n5:validWhen a owl:ObjectProperty ;
    rdfs:label "Valid When"@en ;
    rdfs:comment "Applicability guard of adaptation."@en ;
    rdfs:domain n5:AdaptationSpecification ;
    rdfs:range ac:ConditionSpecification .

n5:variable a owl:ObjectProperty ;
    rdfs:label "Variable"@en ;
    rdfs:comment "Variable reused from the physical-model vocabulary."@en ;
    rdfs:domain n5:VariableAssignment ;
    rdfs:range n18:ModelVariable .

n6:Gas a n6:MaterialState,
        owl:NamedIndividual ;
    rdfs:label "Gas"@en .

n6:Granular a n6:MaterialState,
        owl:NamedIndividual ;
    rdfs:label "Granular"@en .

n6:Liquid a n6:MaterialState,
        owl:NamedIndividual ;
    rdfs:label "Liquid"@en .

n6:Solid a n6:MaterialState,
        owl:NamedIndividual ;
    rdfs:label "Solid"@en .

n6:UnknownState a n6:MaterialState,
        owl:NamedIndividual ;
    rdfs:label "Unknown State"@en .

n6:affordsActionKind a owl:ObjectProperty ;
    rdfs:label "Affords Action Kind"@en ;
    rdfs:domain n6:AffordanceSpecification ;
    rdfs:range ac:ActionKind .

n6:assetIRI a owl:DatatypeProperty ;
    rdfs:label "Asset IRI"@en ;
    rdfs:domain n6:GeometrySpecification ;
    rdfs:range xsd:anyURI .

n6:assetVersion a owl:DatatypeProperty ;
    rdfs:label "Asset Version"@en ;
    rdfs:domain n6:GeometrySpecification ;
    rdfs:range xsd:string .

n6:constituent a owl:ObjectProperty ;
    rdfs:label "Constituent"@en ;
    rdfs:domain n6:MaterialSpecification ;
    rdfs:range n6:MaterialSpecification .

n6:describesEntity a owl:ObjectProperty ;
    rdfs:label "Describes Entity"@en ;
    rdfs:domain n6:EntityDescription ;
    rdfs:range ac:EntitySpecification .

n6:dimensionParameter a owl:ObjectProperty ;
    rdfs:label "Dimension Parameter"@en ;
    rdfs:domain n6:GeometrySpecification ;
    rdfs:range ac:ParameterSpecification .

n6:hasAffordance a owl:ObjectProperty ;
    rdfs:label "Has Affordance"@en ;
    rdfs:domain n6:EntityDescription ;
    rdfs:range n6:AffordanceSpecification .

n6:hasGeometry a owl:ObjectProperty ;
    rdfs:label "Has Geometry"@en ;
    rdfs:domain n6:EntityDescription ;
    rdfs:range n6:GeometrySpecification .

n6:hasMaterial a owl:ObjectProperty ;
    rdfs:label "Has Material"@en ;
    rdfs:domain n6:EntityDescription ;
    rdfs:range n6:MaterialSpecification .

n6:hasOpening a owl:ObjectProperty ;
    rdfs:label "Has Opening"@en ;
    rdfs:domain n6:GeometrySpecification ;
    rdfs:range n6:OpeningSpecification .

n6:materialState a owl:ObjectProperty ;
    rdfs:label "Material State"@en ;
    rdfs:domain n6:MaterialSpecification ;
    rdfs:range n6:MaterialState .

n6:physicalProperty a owl:ObjectProperty ;
    rdfs:label "Physical Property"@en ;
    rdfs:domain n6:EntityDescription ;
    rdfs:range n18:PhysicalPropertySpecification .

n6:requiresCapability a owl:ObjectProperty ;
    rdfs:label "Requires Capability"@en ;
    rdfs:domain n6:AffordanceSpecification ;
    rdfs:range ac:CapabilityRequirement ;
    rdfs:subPropertyOf ac:requiresCapability .

n6:validWhen a owl:ObjectProperty ;
    rdfs:label "Valid When"@en ;
    rdfs:domain n6:AffordanceSpecification ;
    rdfs:range ac:ConditionSpecification .

n7:Collision a n7:FailureKind,
        owl:NamedIndividual ;
    rdfs:label "Collision"@en .

n7:LossOfControl a n7:FailureKind,
        owl:NamedIndividual ;
    rdfs:label "Loss Of Control"@en .

n7:Spillage a n7:FailureKind,
        owl:NamedIndividual ;
    rdfs:label "Spillage"@en .

n7:UnmetPrecondition a n7:FailureKind,
        owl:NamedIndividual ;
    rdfs:label "Unmet Precondition"@en .

n7:addressesFailure a owl:ObjectProperty ;
    rdfs:label "Addresses Failure"@en ;
    rdfs:comment "Failure addressed by a proposed recovery."@en ;
    rdfs:domain n7:RecoveryOption ;
    rdfs:range n7:FailureSpecification .

n7:alternativeTo a owl:ObjectProperty ;
    rdfs:label "Alternative To"@en ;
    rdfs:comment "Alternative explanation; not logical negation."@en ;
    rdfs:domain n7:DiagnosticHypothesis ;
    rdfs:range n7:DiagnosticHypothesis .

n7:candidateCause a owl:ObjectProperty ;
    rdfs:label "Candidate Cause"@en ;
    rdfs:comment "Proposed cause condition."@en ;
    rdfs:domain n7:DiagnosticHypothesis ;
    rdfs:range ac:ConditionSpecification .

n7:detectedBy a owl:ObjectProperty ;
    rdfs:label "Detected By"@en ;
    rdfs:comment "Criterion describing failure detection."@en ;
    rdfs:domain n7:FailureSpecification ;
    rdfs:range ac:FailureCriterion .

n7:explainsFailure a owl:ObjectProperty ;
    rdfs:label "Explains Failure"@en ;
    rdfs:comment "Failure explained by this candidate hypothesis."@en ;
    rdfs:domain n7:DiagnosticHypothesis ;
    rdfs:range n7:FailureSpecification .

n7:failureKind a owl:ObjectProperty ;
    rdfs:label "Failure Kind"@en ;
    rdfs:comment "Failure category."@en ;
    rdfs:domain n7:FailureSpecification ;
    rdfs:range n7:FailureKind .

n7:guardedBy a owl:ObjectProperty ;
    rdfs:label "Guarded By"@en ;
    rdfs:comment "Applicability guard."@en ;
    rdfs:domain n7:RecoveryOption ;
    rdfs:range ac:ConditionSpecification .

n7:hasFailure a owl:ObjectProperty ;
    rdfs:label "Has Failure"@en ;
    rdfs:comment "Failure modes in this profile."@en ;
    rdfs:domain n7:FailureProfile ;
    rdfs:range n7:FailureSpecification .

n7:hasFailureProfile a owl:ObjectProperty ;
    rdfs:label "Has Failure Profile"@en ;
    rdfs:comment "Potential failures for this action."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n7:FailureProfile ;
    rdfs:subPropertyOf ac:hasComponent .

n7:hasHypothesis a owl:ObjectProperty ;
    rdfs:label "Has Hypothesis"@en ;
    rdfs:comment "Candidate explanations."@en ;
    rdfs:domain n7:FailureProfile ;
    rdfs:range n7:DiagnosticHypothesis .

n7:hasRecoveryOption a owl:ObjectProperty ;
    rdfs:label "Has Recovery Option"@en ;
    rdfs:comment "Potential recovery options."@en ;
    rdfs:domain n7:FailureProfile ;
    rdfs:range n7:RecoveryOption .

n7:refutingEvidence a owl:ObjectProperty ;
    rdfs:label "Refuting Evidence"@en ;
    rdfs:comment "Evidence offered against the hypothesis."@en ;
    rdfs:domain n7:DiagnosticHypothesis ;
    rdfs:range ac:EvidenceRecord .

n7:requiresCapability a owl:ObjectProperty ;
    rdfs:label "Requires Capability"@en ;
    rdfs:comment "Required recovery capability."@en ;
    rdfs:domain n7:RecoveryOption ;
    rdfs:range ac:CapabilityRequirement ;
    rdfs:subPropertyOf ac:requiresCapability .

n7:restoresCondition a owl:ObjectProperty ;
    rdfs:label "Restores Condition"@en ;
    rdfs:comment "Intended restored condition, not a guaranteed effect."@en ;
    rdfs:domain n7:RecoveryOption ;
    rdfs:range ac:ConditionSpecification .

n7:retryBudget a owl:ObjectProperty ;
    rdfs:label "Retry Budget"@en ;
    rdfs:comment "Optional explicit dimensionless retry budget; no default inferred."@en ;
    rdfs:domain n7:RecoveryOption ;
    rdfs:range ac:ParameterSpecification .

n7:supportingEvidence a owl:ObjectProperty ;
    rdfs:label "Supporting Evidence"@en ;
    rdfs:comment "Evidence offered in support; not automatically accepted."@en ;
    rdfs:domain n7:DiagnosticHypothesis ;
    rdfs:range ac:EvidenceRecord .

n7:usesStrategy a owl:ObjectProperty ;
    rdfs:label "Uses Strategy"@en ;
    rdfs:comment "Optional recovery strategy binding, without importing strategies into the base failure module."@en ;
    rdfs:domain n7:RecoveryOption ;
    rdfs:range n1:ExecutionStrategy .

n8:AgonistStronger a n8:StrengthRelation,
        owl:NamedIndividual ;
    rdfs:label "Agonist Stronger"@en .

n8:AntagonistStronger a n8:StrengthRelation,
        owl:NamedIndividual ;
    rdfs:label "Antagonist Stronger"@en .

n8:Balanced a n8:StrengthRelation,
        owl:NamedIndividual ;
    rdfs:label "Balanced"@en .

n8:Motion a n8:Resultant,
        owl:NamedIndividual ;
    rdfs:label "Motion"@en .

n8:Rest a n8:Resultant,
        owl:NamedIndividual ;
    rdfs:label "Rest"@en .

n8:TowardMotion a n8:Tendency,
        owl:NamedIndividual ;
    rdfs:label "Toward Motion"@en .

n8:TowardRest a n8:Tendency,
        owl:NamedIndividual ;
    rdfs:label "Toward Rest"@en .

n8:UnknownResult a n8:Resultant,
        owl:NamedIndividual ;
    rdfs:label "Unknown Result"@en .

n8:UnknownStrength a n8:StrengthRelation,
        owl:NamedIndividual ;
    rdfs:label "Unknown Strength"@en .

n8:UnknownTendency a n8:Tendency,
        owl:NamedIndividual ;
    rdfs:label "Unknown Tendency"@en .

n8:agonist a owl:ObjectProperty ;
    rdfs:label "Agonist"@en ;
    rdfs:domain n8:ForceDynamicDescription ;
    rdfs:range n8:ForceParticipant .

n8:antagonist a owl:ObjectProperty ;
    rdfs:label "Antagonist"@en ;
    rdfs:domain n8:ForceDynamicDescription ;
    rdfs:range n8:ForceParticipant .

n8:applicability a owl:ObjectProperty ;
    rdfs:label "Applicability"@en ;
    rdfs:domain n8:ForceDynamicDescription ;
    rdfs:range ac:ApplicabilitySpecification .

n8:atPhase a owl:ObjectProperty ;
    rdfs:label "At Phase"@en ;
    rdfs:domain n8:ForceDynamicDescription ;
    rdfs:range ac:PhaseSpecification .

n8:describedResult a owl:ObjectProperty ;
    rdfs:label "Described Result"@en ;
    rdfs:domain n8:ForceDynamicDescription ;
    rdfs:range n8:Resultant .

n8:entity a owl:ObjectProperty ;
    rdfs:label "Entity"@en ;
    rdfs:domain n8:ForceParticipant ;
    rdfs:range ac:EntitySpecification .

n8:relativeStrength a owl:ObjectProperty ;
    rdfs:label "Relative Strength"@en ;
    rdfs:domain n8:ForceDynamicDescription ;
    rdfs:range n8:StrengthRelation .

n8:tendency a owl:ObjectProperty ;
    rdfs:label "Tendency"@en ;
    rdfs:domain n8:ForceParticipant ;
    rdfs:range n8:Tendency .

n9:allowedStatus a owl:ObjectProperty ;
    rdfs:label "Allowed Status"@en ;
    rdfs:domain n9:FrameProjectionSpecification ;
    rdfs:range ac:EpistemicStatus .

n9:defaultEntity a owl:ObjectProperty ;
    rdfs:label "Default Entity"@en ;
    rdfs:domain n9:SlotDefinition ;
    rdfs:range n24:Entity .

n9:defaultLiteral a owl:DatatypeProperty ;
    rdfs:label "Default Literal"@en ;
    rdfs:domain n9:SlotDefinition ;
    rdfs:range rdfs:Literal .

n9:entityValue a owl:ObjectProperty ;
    rdfs:label "Entity Value"@en ;
    rdfs:domain n9:FrameSlotValue ;
    rdfs:range n24:Entity .

n9:evidencePolicy a owl:DatatypeProperty ;
    rdfs:label "Evidence Policy"@en ;
    rdfs:domain n9:FrameProjectionSpecification ;
    rdfs:range xsd:string .

n9:fillsSlot a owl:ObjectProperty ;
    rdfs:label "Fills Slot"@en ;
    rdfs:domain n9:FrameSlotValue ;
    rdfs:range n9:SlotDefinition .

n9:frameDefinition a owl:ObjectProperty ;
    rdfs:label "Frame Definition"@en ;
    rdfs:domain n9:FrameProjectionSpecification ;
    rdfs:range n9:FrameDefinition .

n9:hasMapping a owl:ObjectProperty ;
    rdfs:label "Has Mapping"@en ;
    rdfs:domain n9:SlotDefinition ;
    rdfs:range n9:SlotMapping .

n9:hasOverride a owl:ObjectProperty ;
    rdfs:label "Has Override"@en ;
    rdfs:domain n9:FrameDefinition ;
    rdfs:range n9:SlotOverride .

n9:hasPathStep a owl:ObjectProperty ;
    rdfs:label "Has Path Step"@en ;
    rdfs:domain n9:SlotMapping ;
    rdfs:range n9:PathStep .

n9:hasSlot a owl:ObjectProperty ;
    rdfs:label "Has Slot"@en ;
    rdfs:domain n9:FrameDefinition ;
    rdfs:range n9:SlotDefinition ;
    rdfs:subPropertyOf n24:defines .

n9:hasSlotValue a owl:ObjectProperty ;
    rdfs:label "Has Slot Value"@en ;
    rdfs:domain n9:FrameProjectionArtifact ;
    rdfs:range n9:FrameSlotValue .

n9:literalValue a owl:DatatypeProperty ;
    rdfs:label "Literal Value"@en ;
    rdfs:domain n9:FrameSlotValue ;
    rdfs:range rdfs:Literal .

n9:maximumCount a owl:DatatypeProperty ;
    rdfs:label "Maximum Count"@en ;
    rdfs:domain n9:SlotDefinition ;
    rdfs:range xsd:nonNegativeInteger .

n9:minimumCount a owl:DatatypeProperty ;
    rdfs:label "Minimum Count"@en ;
    rdfs:domain n9:SlotDefinition ;
    rdfs:range xsd:nonNegativeInteger .

n9:overridesSlot a owl:ObjectProperty ;
    rdfs:label "Overrides Slot"@en ;
    rdfs:domain n9:SlotOverride ;
    rdfs:range n9:SlotDefinition .

n9:parentFrame a owl:ObjectProperty ;
    rdfs:label "Parent Frame"@en ;
    rdfs:domain n9:FrameDefinition ;
    rdfs:range n9:FrameDefinition .

n9:position a owl:DatatypeProperty ;
    rdfs:label "Position"@en ;
    rdfs:domain n9:PathStep ;
    rdfs:range xsd:positiveInteger .

n9:predicateIRI a owl:DatatypeProperty ;
    rdfs:label "Predicate IRI"@en ;
    rdfs:domain n9:PathStep ;
    rdfs:range xsd:anyURI .

n9:projectedFrame a owl:ObjectProperty ;
    rdfs:label "Projected Frame"@en ;
    rdfs:domain n9:FrameProjectionArtifact ;
    rdfs:range n9:FrameDefinition .

n9:replacementLiteral a owl:DatatypeProperty ;
    rdfs:label "Replacement Literal"@en ;
    rdfs:domain n9:SlotOverride ;
    rdfs:range rdfs:Literal .

n9:slotKey a owl:DatatypeProperty ;
    rdfs:label "Slot Key"@en ;
    rdfs:domain n9:SlotDefinition ;
    rdfs:range xsd:string .

n9:sourceClaim a owl:ObjectProperty ;
    rdfs:label "Source Claim"@en ;
    rdfs:domain n9:FrameSlotValue ;
    rdfs:range ac:KnowledgeClaim .

n9:suppressDefault a owl:DatatypeProperty ;
    rdfs:label "Suppress Default"@en ;
    rdfs:domain n9:SlotOverride ;
    rdfs:range xsd:boolean .

n9:valueOrigin a owl:DatatypeProperty ;
    rdfs:label "Value Origin"@en ;
    rdfs:domain n9:FrameSlotValue ;
    rdfs:range xsd:string .

n10:container a owl:ObjectProperty ;
    rdfs:label "Container"@en ;
    rdfs:domain n10:ContainmentSchema ;
    rdfs:range ac:EntitySpecification .

n10:content a owl:ObjectProperty ;
    rdfs:label "Content"@en ;
    rdfs:domain n10:ContainmentSchema ;
    rdfs:range ac:EntitySpecification .

n10:firstContactEntity a owl:ObjectProperty ;
    rdfs:label "First Contact Entity"@en ;
    rdfs:domain n10:ContactSchema ;
    rdfs:range ac:EntitySpecification .

n10:goal a owl:ObjectProperty ;
    rdfs:label "Goal"@en ;
    rdfs:domain n10:SourcePathGoalSchema ;
    rdfs:range ac:EntitySpecification .

n10:hasPathElement a owl:ObjectProperty ;
    rdfs:label "Has Path Element"@en ;
    rdfs:domain n10:SourcePathGoalSchema ;
    rdfs:range n10:PathElement .

n10:location a owl:ObjectProperty ;
    rdfs:label "Location"@en ;
    rdfs:domain n10:PathElement ;
    rdfs:range ac:EntitySpecification .

n10:position a owl:DatatypeProperty ;
    rdfs:label "Position"@en ;
    rdfs:domain n10:PathElement ;
    rdfs:range xsd:positiveInteger .

n10:secondContactEntity a owl:ObjectProperty ;
    rdfs:label "Second Contact Entity"@en ;
    rdfs:domain n10:ContactSchema ;
    rdfs:range ac:EntitySpecification .

n10:source a owl:ObjectProperty ;
    rdfs:label "Source"@en ;
    rdfs:domain n10:SourcePathGoalSchema ;
    rdfs:range ac:EntitySpecification .

n10:supported a owl:ObjectProperty ;
    rdfs:label "Supported"@en ;
    rdfs:domain n10:SupportSchema ;
    rdfs:range ac:EntitySpecification .

n10:supporter a owl:ObjectProperty ;
    rdfs:label "Supporter"@en ;
    rdfs:domain n10:SupportSchema ;
    rdfs:range ac:EntitySpecification .

n10:theme a owl:ObjectProperty ;
    rdfs:label "Theme"@en ;
    rdfs:domain n10:SourcePathGoalSchema ;
    rdfs:range ac:EntitySpecification .

n11:assessedAt a owl:DatatypeProperty ;
    rdfs:label "Assessed At"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range xsd:dateTime .

n11:assessedStatus a owl:ObjectProperty ;
    rdfs:label "Assessed Status"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range ac:EpistemicStatus .

n11:assessesClaim a owl:ObjectProperty ;
    rdfs:label "Assesses Claim"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range ac:KnowledgeClaim .

n11:contextKey a owl:DatatypeProperty ;
    rdfs:label "Context Key"@en ;
    rdfs:domain n11:KnowledgeContext ;
    rdfs:range xsd:string .

n11:exactExcerpt a owl:DatatypeProperty ;
    rdfs:label "Exact Excerpt"@en ;
    rdfs:domain n11:SourceSnapshot ;
    rdfs:range xsd:string .

n11:externalContextIRI a owl:DatatypeProperty ;
    rdfs:label "External Context I R I"@en ;
    rdfs:domain n11:KnowledgeContext ;
    rdfs:range xsd:anyURI .

n11:inContext a owl:ObjectProperty ;
    rdfs:label "In Context"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range n11:KnowledgeContext .

n11:inRecord a owl:ObjectProperty ;
    rdfs:label "In Record"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range ac:KnowledgeRecord .

n11:observedAt a owl:DatatypeProperty ;
    rdfs:label "Observed At"@en ;
    rdfs:domain n11:ObservationEvidence ;
    rdfs:range xsd:dateTime .

n11:revisesAssessment a owl:ObjectProperty ;
    rdfs:label "Revises Assessment"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range n11:ClaimAssessment .

n11:scopedDescription a owl:ObjectProperty ;
    rdfs:label "Scoped Description"@en ;
    rdfs:domain n11:KnowledgeContext ;
    rdfs:range ac:ActionDescription .

n11:selector a owl:DatatypeProperty ;
    rdfs:label "Selector"@en ;
    rdfs:domain n11:SourceSnapshot ;
    rdfs:range xsd:string .

n11:serializationProfile a owl:DatatypeProperty ;
    rdfs:label "Serialization Profile"@en ;
    rdfs:domain n11:RecordSnapshot ;
    rdfs:range xsd:string .

n11:sha256 a owl:DatatypeProperty ;
    rdfs:label "Sha256"@en ;
    rdfs:domain n24:InformationObject ;
    rdfs:range xsd:string .

n11:snapshotOf a owl:ObjectProperty ;
    rdfs:label "Snapshot Of"@en ;
    rdfs:domain n11:RecordSnapshot ;
    rdfs:range ac:KnowledgeRecord .

n11:sourceIRI a owl:DatatypeProperty ;
    rdfs:label "Source I R I"@en ;
    rdfs:domain n11:SourceSnapshot ;
    rdfs:range xsd:anyURI .

n11:sourceSnapshot a owl:ObjectProperty ;
    rdfs:label "Source Snapshot"@en ;
    rdfs:domain ac:EvidenceRecord ;
    rdfs:range n11:SourceSnapshot .

n11:sourceVersion a owl:DatatypeProperty ;
    rdfs:label "Source Version"@en ;
    rdfs:domain n11:SourceSnapshot ;
    rdfs:range xsd:string .

n11:usesEvidence a owl:ObjectProperty ;
    rdfs:label "Uses Evidence"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range ac:EvidenceRecord .

n11:validFrom a owl:DatatypeProperty ;
    rdfs:label "Valid From"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range xsd:dateTime .

n11:validUntil a owl:DatatypeProperty ;
    rdfs:label "Valid Until"@en ;
    rdfs:domain n11:ClaimAssessment ;
    rdfs:range xsd:dateTime .

mt:CompleteMaterialTransfer a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Complete material transfer"@en .

mt:CompleteTransferDescription a owl:Class ;
    rdfs:label "Complete transfer description"@en ;
    rdfs:comment "Transfer with explicit residual-source amount and tolerance criteria."@en ;
    rdfs:subClassOf mt:MaterialTransferDescription .

mt:PartialMaterialTransfer a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Partial material transfer"@en .

mt:PartialTransferDescription a owl:Class ;
    rdfs:label "Partial transfer description"@en ;
    rdfs:comment "Transfer a requested amount without requiring an empty source."@en ;
    rdfs:subClassOf mt:MaterialTransferDescription .

mt:TransferFault a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Transfer fault"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition mt:TransferFaultArgument1,
        mt:TransferFaultArgument2 ;
    rdfs:comment "Context-qualified transfer condition requiring explicit evidence."@en .

mt:TransferReady a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Transfer ready"@en ;
    n16:arity "3"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition mt:TransferReadyArgument1,
        mt:TransferReadyArgument2,
        mt:TransferReadyArgument3 ;
    rdfs:comment "Context-qualified transfer condition requiring explicit evidence."@en .

mt:residualAmount a owl:ObjectProperty ;
    rdfs:label "Residual amount"@en ;
    rdfs:domain mt:MaterialTransferDescription ;
    rdfs:range ac:ParameterSpecification .

mt:residualTolerance a owl:ObjectProperty ;
    rdfs:label "Residual tolerance"@en ;
    rdfs:domain mt:MaterialTransferDescription ;
    rdfs:range ac:ParameterSpecification .

mt:usesToolInteraction a owl:ObjectProperty ;
    rdfs:label "Uses tool interaction"@en ;
    rdfs:domain mt:PouringDescription ;
    rdfs:range n21:ToolUseDescription .

n12:coefficient a owl:DatatypeProperty ;
    rdfs:label "Coefficient"@en ;
    rdfs:domain n12:EquationTerm ;
    rdfs:range xsd:decimal .

n12:dimensionVector a owl:DatatypeProperty ;
    rdfs:label "Dimension vector"@en ;
    rdfs:domain n12:QuantityContract ;
    rdfs:range xsd:string .

n12:equation a owl:ObjectProperty ;
    rdfs:label "Equation"@en ;
    rdfs:domain n12:AdditiveEquationContract ;
    rdfs:range n18:ModelEquation .

n12:hasTerm a owl:ObjectProperty ;
    rdfs:label "Has term"@en ;
    rdfs:domain n12:AdditiveEquationContract ;
    rdfs:range n12:EquationTerm .

n12:parameter a owl:ObjectProperty ;
    rdfs:label "Parameter"@en ;
    rdfs:domain n12:QuantityContract ;
    rdfs:range ac:ParameterSpecification .

n12:resultParameter a owl:ObjectProperty ;
    rdfs:label "Result parameter"@en ;
    rdfs:domain n12:AdditiveEquationContract ;
    rdfs:range ac:ParameterSpecification .

n12:scope a owl:ObjectProperty ;
    rdfs:label "Scope"@en ;
    rdfs:domain n12:AdditiveEquationContract ;
    rdfs:range ac:ApplicabilitySpecification .

n12:termParameter a owl:ObjectProperty ;
    rdfs:label "Term parameter"@en ;
    rdfs:domain n12:EquationTerm ;
    rdfs:range ac:ParameterSpecification .

n12:unitIRI a owl:DatatypeProperty ;
    rdfs:label "Unit i r i"@en ;
    rdfs:domain n12:QuantityContract ;
    rdfs:range xsd:anyURI .

n13:DuringPhase a n13:MonitoringPoint,
        owl:NamedIndividual ;
    rdfs:label "During Phase"@en .

n13:PhaseEntry a n13:MonitoringPoint,
        owl:NamedIndividual ;
    rdfs:label "Phase Entry"@en .

n13:PhaseExit a n13:MonitoringPoint,
        owl:NamedIndividual ;
    rdfs:label "Phase Exit"@en .

n13:ReportSatisfied a n13:MonitoringResponse,
        owl:NamedIndividual ;
    rdfs:label "Report Satisfied"@en .

n13:ReportUnknown a n13:MonitoringResponse,
        owl:NamedIndividual ;
    rdfs:label "Report Unknown"@en .

n13:ReportUnsatisfied a n13:MonitoringResponse,
        owl:NamedIndividual ;
    rdfs:label "Report Unsatisfied"@en .

n13:ReportViolation a n13:MonitoringResponse,
        owl:NamedIndividual ;
    rdfs:label "Report Violation"@en .

n13:RequestEvidence a n13:MonitoringResponse,
        owl:NamedIndividual ;
    rdfs:label "Request Evidence"@en .

n13:RequestRecovery a n13:MonitoringResponse,
        owl:NamedIndividual ;
    rdfs:label "Request Recovery"@en .

n13:RequestSuspension a n13:MonitoringResponse,
        owl:NamedIndividual ;
    rdfs:label "Request Suspension"@en .

n13:TransitionCheck a n13:MonitoringPoint,
        owl:NamedIndividual ;
    rdfs:label "Transition Check"@en .

n13:atPoint a owl:ObjectProperty ;
    rdfs:label "At Point"@en ;
    rdfs:comment "Phase entry, during phase, phase exit or transition check."@en ;
    rdfs:domain n13:MonitoringRequirement ;
    rdfs:range n13:MonitoringPoint .

n13:hasMonitoringRequirement a owl:ObjectProperty ;
    rdfs:label "Has Monitoring Requirement"@en ;
    rdfs:comment "Monitor attached to a phase or transition specification."@en ;
    rdfs:domain ac:ActionComponent ;
    rdfs:range n13:MonitoringRequirement .

n13:inScope a owl:ObjectProperty ;
    rdfs:label "In Scope"@en ;
    rdfs:comment "Scope matching the referenced expression leaf."@en ;
    rdfs:domain n13:ObservationRequirement ;
    rdfs:range n3:StateScope .

n13:monitorsCondition a owl:ObjectProperty ;
    rdfs:label "Monitors Condition"@en ;
    rdfs:comment "Condition evaluated by this requirement."@en ;
    rdfs:domain n13:MonitoringRequirement ;
    rdfs:range ac:ConditionSpecification .

n13:observesParameter a owl:ObjectProperty ;
    rdfs:label "Observes Parameter"@en ;
    rdfs:comment "Parameter whose quantity is needed."@en ;
    rdfs:domain n13:QuantityObservation ;
    rdfs:range ac:ParameterSpecification .

n13:observesPattern a owl:ObjectProperty ;
    rdfs:label "Observes Pattern"@en ;
    rdfs:comment "Symbolic relation whose truth is needed."@en ;
    rdfs:domain n13:RelationObservation ;
    rdfs:range n16:RelationPattern .

n13:requiresCapability a owl:ObjectProperty ;
    rdfs:label "Requires Capability"@en ;
    rdfs:comment "Optional abstract sensing/estimation capability; independent of robot or sensor APIs."@en ;
    rdfs:domain n13:ObservationRequirement ;
    rdfs:range ac:CapabilityRequirement ;
    rdfs:subPropertyOf ac:requiresCapability .

n13:requiresObservation a owl:ObjectProperty ;
    rdfs:label "Requires Observation"@en ;
    rdfs:comment "Required evidence inputs; validated against the expression dependencies."@en ;
    rdfs:domain n13:MonitoringRequirement ;
    rdfs:range n13:ObservationRequirement .

n13:whenFalse a owl:ObjectProperty ;
    rdfs:label "When False"@en ;
    rdfs:comment "Response to explicitly false evaluation."@en ;
    rdfs:domain n13:MonitoringRequirement ;
    rdfs:range n13:MonitoringResponse .

n13:whenTrue a owl:ObjectProperty ;
    rdfs:label "When True"@en ;
    rdfs:comment "Response to explicitly true evaluation."@en ;
    rdfs:domain n13:MonitoringRequirement ;
    rdfs:range n13:MonitoringResponse .

n13:whenUnknown a owl:ObjectProperty ;
    rdfs:label "When Unknown"@en ;
    rdfs:comment "Response to missing or incompatible evidence; cannot treat unknown as a satisfied condition."@en ;
    rdfs:domain n13:MonitoringRequirement ;
    rdfs:range n13:MonitoringResponse .

n14:AtTarget a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "At target"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:AtTargetArgument1,
        n14:AtTargetArgument2 ;
    rdfs:comment "Entity is within the intended region under an explicitly supplied tolerance/frame interpretation."@en .

n14:ClearOfSupport a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Clear of support"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:ClearOfSupportArgument1,
        n14:ClearOfSupportArgument2 ;
    rdfs:comment "Object has cleared the specified support under a supplied clearance criterion."@en .

n14:Fixating a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Fixating"@en .

n14:GraspAcquisition a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Grasp acquisition"@en .

n14:GraspAcquisitionDescription a owl:Class ;
    rdfs:label "Grasp acquisition description"@en ;
    rdfs:comment "Acquire an explicitly assessed hold and contact; no lifting is implied."@en ;
    rdfs:subClassOf n14:ObjectManipulationDescription .

n14:GraspFailed a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Grasp failed"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:GraspFailedArgument1,
        n14:GraspFailedArgument2 ;
    rdfs:comment "Grasp attempt is explicitly reported failed; absence of success evidence is insufficient."@en .

n14:HoldLost a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Hold lost"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:HoldLostArgument1,
        n14:HoldLostArgument2 ;
    rdfs:comment "An expected hold was lost during the supplied evaluation interval."@en .

n14:HoldMaintained a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Hold maintained"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:HoldMaintainedArgument1,
        n14:HoldMaintainedArgument2 ;
    rdfs:comment "Hold was maintained throughout the supplied evaluation interval, not merely at its endpoint."@en .

n14:Holding a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Holding"@en .

n14:HoldingDescription a owl:Class ;
    rdfs:label "Holding description"@en ;
    rdfs:comment "Maintain a hold over an explicit externally evaluated interval."@en ;
    rdfs:subClassOf n14:ObjectManipulationDescription .

n14:Holds a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Holds"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:HoldsArgument1,
        n14:HoldsArgument2 ;
    rdfs:comment "Manipulator has an assessed hold on the object under the supplied context; gripper closure alone is insufficient."@en .

n14:ManipulatorClear a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Manipulator clear"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:ManipulatorClearArgument1,
        n14:ManipulatorClearArgument2 ;
    rdfs:comment "Manipulator is clear of the object after a placement under a supplied clearance criterion."@en .

n14:PickingUp a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Picking up"@en .

n14:Placing a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Placing"@en .

n14:PoseMaintained a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Pose maintained"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:PoseMaintainedArgument1,
        n14:PoseMaintainedArgument2 ;
    rdfs:comment "Object pose was maintained relative to the specified frame throughout the evaluation interval."@en .

n14:Reaching a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Reaching"@en .

n14:ReachingDescription a owl:Class ;
    rdfs:label "Reaching description"@en ;
    rdfs:comment "Reach an intended target region; establishes no grasp."@en ;
    rdfs:subClassOf n14:ObjectManipulationDescription .

n14:ReleaseBlocked a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Release blocked"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:ReleaseBlockedArgument1,
        n14:ReleaseBlockedArgument2 ;
    rdfs:comment "Release attempt is explicitly reported blocked."@en .

n14:ReleaseDescription a owl:Class ;
    rdfs:label "Release description"@en ;
    rdfs:comment "Cease holding; support or placement is not implied."@en ;
    rdfs:subClassOf n14:ObjectManipulationDescription .

n14:Releasing a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Releasing"@en .

n14:StableOnSupport a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Stable on support"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:StableOnSupportArgument1,
        n14:StableOnSupportArgument2 ;
    rdfs:comment "Object is assessed as stably supported by the specified support; mere contact is insufficient."@en .

n14:SupportUnstable a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Support unstable"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:SupportUnstableArgument1,
        n14:SupportUnstableArgument2 ;
    rdfs:comment "Object/support instability is explicitly reported."@en .

n14:UnexpectedContact a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Unexpected contact"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n14:UnexpectedContactArgument1,
        n14:UnexpectedContactArgument2 ;
    rdfs:comment "Unexpected manipulator/object contact is reported under the supplied contact policy."@en .

n14:anchorFrame a owl:ObjectProperty ;
    rdfs:label "Anchor frame"@en ;
    rdfs:domain n14:FixationDescription ;
    rdfs:range n19:FrameSpecification .

n14:motionControlledEntity a owl:ObjectProperty ;
    rdfs:label "Motion controlled entity"@en ;
    rdfs:domain n14:ManipulationMotionRequirement ;
    rdfs:range ac:EntitySpecification .

n14:motionKind a owl:ObjectProperty ;
    rdfs:label "Motion kind"@en ;
    rdfs:domain n14:ManipulationMotionRequirement ;
    rdfs:range ac:ActionKind .

n14:motionTarget a owl:ObjectProperty ;
    rdfs:label "Motion target"@en ;
    rdfs:domain n14:ManipulationMotionRequirement ;
    rdfs:range n14:TargetRegionSpecification .

n14:poseSpecification a owl:ObjectProperty ;
    rdfs:label "Pose specification"@en ;
    rdfs:domain n14:TargetRegionSpecification ;
    rdfs:range n19:PoseTarget .

n14:requiresMotion a owl:ObjectProperty ;
    rdfs:label "Requires motion"@en ;
    rdfs:domain n14:ObjectManipulationDescription ;
    rdfs:range n14:ManipulationMotionRequirement ;
    rdfs:subPropertyOf ac:hasComponent .

n15:Lexicographic a n15:AggregationPolicy,
        owl:NamedIndividual ;
    rdfs:label "Lexicographic"@en .

n15:Maximize a n15:OptimizationDirection,
        owl:NamedIndividual ;
    rdfs:label "Maximize"@en .

n15:Minimize a n15:OptimizationDirection,
        owl:NamedIndividual ;
    rdfs:label "Minimize"@en .

n15:Pareto a n15:AggregationPolicy,
        owl:NamedIndividual ;
    rdfs:label "Pareto"@en .

n15:WeightedSum a n15:AggregationPolicy,
        owl:NamedIndividual ;
    rdfs:label "Weighted Sum"@en .

n15:aggregationPolicy a owl:ObjectProperty ;
    rdfs:label "Aggregation Policy"@en ;
    rdfs:comment "Weighted sum, lexicographic or Pareto policy."@en ;
    rdfs:domain n15:ObjectiveSet ;
    rdfs:range n15:AggregationPolicy .

n15:direction a owl:ObjectProperty ;
    rdfs:label "Direction"@en ;
    rdfs:comment "Optimization direction."@en ;
    rdfs:domain n15:ObjectiveTerm ;
    rdfs:range n15:OptimizationDirection .

n15:hardConstraint a owl:ObjectProperty ;
    rdfs:label "Hard Constraint"@en ;
    rdfs:comment "A condition that cannot be traded off as a soft objective."@en ;
    rdfs:domain n15:ObjectiveSet ;
    rdfs:range ac:ConstraintSpecification .

n15:hasObjectiveSet a owl:ObjectProperty ;
    rdfs:label "Has Objective Set"@en ;
    rdfs:comment "Objective configuration for this action."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n15:ObjectiveSet ;
    rdfs:subPropertyOf ac:hasComponent .

n15:hasTerm a owl:ObjectProperty ;
    rdfs:label "Has Term"@en ;
    rdfs:comment "Terms in this set."@en ;
    rdfs:domain n15:ObjectiveSet ;
    rdfs:range n15:ObjectiveTerm .

n15:metricParameter a owl:ObjectProperty ;
    rdfs:label "Metric Parameter"@en ;
    rdfs:comment "Metric whose value is optimized."@en ;
    rdfs:domain n15:ObjectiveTerm ;
    rdfs:range ac:ParameterSpecification .

n15:normalizedBy a owl:ObjectProperty ;
    rdfs:label "Normalized By"@en ;
    rdfs:comment "Normalization required for weighted sums."@en ;
    rdfs:domain n15:ObjectiveTerm ;
    rdfs:range n15:NormalizationSpecification .

n15:rank a owl:DatatypeProperty ;
    rdfs:label "Rank"@en ;
    rdfs:comment "Lexicographic rank; smaller values take precedence."@en ;
    rdfs:domain n15:ObjectiveTerm ;
    rdfs:range xsd:nonNegativeInteger .

n15:scaleParameter a owl:ObjectProperty ;
    rdfs:label "Scale Parameter"@en ;
    rdfs:comment "Positive reference scale; can remain unspecified."@en ;
    rdfs:domain n15:NormalizationSpecification ;
    rdfs:range ac:ParameterSpecification .

n15:weightParameter a owl:ObjectProperty ;
    rdfs:label "Weight Parameter"@en ;
    rdfs:comment "Optional explicit dimensionless weight; no default is inferred."@en ;
    rdfs:domain n15:ObjectiveTerm ;
    rdfs:range ac:ParameterSpecification .

n16:Accessible a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Accessible"@en ;
    n16:arity "1"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n16:AccessibleArgument1 ;
    rdfs:comment "Accessibility under the externally supplied context."@en .

n16:AgentRole a owl:Class ;
    rdfs:label "Agent Role"@en ;
    rdfs:comment "Agent role, without requiring any particular robot. Role subclasses are not mutually disjoint: a conceptual role may serve multiple functions."@en ;
    rdfs:subClassOf ac:ParticipantRole .

n16:ContainedIn a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Contained In"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n16:ContainedInArgument1,
        n16:ContainedInArgument2 ;
    rdfs:comment "The specified content is contained in the specified container; amount is not implied."@en .

n16:Controls a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Controls"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n16:ControlsArgument1,
        n16:ControlsArgument2 ;
    rdfs:comment "A controlled hold or other control relation, with exact operational meaning supplied by an adapter."@en .

n16:DestinationRole a owl:Class ;
    rdfs:label "Destination Role"@en ;
    rdfs:comment "Intended recipient or destination. Role subclasses are not mutually disjoint: a conceptual role may serve multiple functions."@en ;
    rdfs:subClassOf ac:ParticipantRole .

n16:Graspable a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Graspable"@en ;
    n16:arity "1"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n16:GraspableArgument1 ;
    rdfs:comment "Graspability under the externally supplied context; not an intrinsic universal property."@en .

n16:InContactWith a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "In Contact With"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n16:InContactWithArgument1,
        n16:InContactWithArgument2 ;
    rdfs:comment "Contact between the two entities; no force or stability is implied."@en .

n16:InstrumentRole a owl:Class ;
    rdfs:label "Instrument Role"@en ;
    rdfs:comment "Means used to carry out an action. Role subclasses are not mutually disjoint: a conceptual role may serve multiple functions."@en ;
    rdfs:subClassOf ac:ParticipantRole .

n16:SourceRole a owl:Class ;
    rdfs:label "Source Role"@en ;
    rdfs:comment "Origin in a transfer or transformation. Role subclasses are not mutually disjoint: a conceptual role may serve multiple functions."@en ;
    rdfs:subClassOf ac:ParticipantRole .

n16:SupportRole a owl:Class ;
    rdfs:label "Support Role"@en ;
    rdfs:comment "Supporting entity role. Role subclasses are not mutually disjoint: a conceptual role may serve multiple functions."@en ;
    rdfs:subClassOf ac:ParticipantRole .

n16:Supports a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Supports"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n16:SupportsArgument1,
        n16:SupportsArgument2 ;
    rdfs:comment "The first entity supports the second under the stated context."@en .

n16:ThemeRole a owl:Class ;
    rdfs:label "Theme Role"@en ;
    rdfs:comment "Entity acted on or transferred. Role subclasses are not mutually disjoint: a conceptual role may serve multiple functions."@en ;
    rdfs:subClassOf ac:ParticipantRole .

n16:argumentPosition a owl:DatatypeProperty ;
    rdfs:label "Argument Position"@en ;
    rdfs:comment "One-based position in a relation signature."@en ;
    rdfs:domain n16:ArgumentDefinition ;
    rdfs:range xsd:positiveInteger .

n16:arity a owl:DatatypeProperty ;
    rdfs:label "Arity"@en ;
    rdfs:comment "Number of arguments in the signature."@en ;
    rdfs:domain n16:RelationDefinition ;
    rdfs:range xsd:positiveInteger .

n16:bindsArgument a owl:ObjectProperty ;
    rdfs:label "Binds Argument"@en ;
    rdfs:comment "Signature position assigned by this binding."@en ;
    rdfs:domain n16:ArgumentBinding ;
    rdfs:range n16:ArgumentDefinition .

n16:bindsEntity a owl:ObjectProperty ;
    rdfs:label "Binds Entity"@en ;
    rdfs:comment "Conceptual entity specification filling this position."@en ;
    rdfs:domain n16:ArgumentBinding ;
    rdfs:range ac:EntitySpecification .

n16:hasArgumentBinding a owl:ObjectProperty ;
    rdfs:label "Has Argument Binding"@en ;
    rdfs:comment "Argument bindings of a symbolic relation pattern."@en ;
    rdfs:domain n16:RelationPattern ;
    rdfs:range n16:ArgumentBinding .

n16:hasArgumentDefinition a owl:ObjectProperty ;
    rdfs:label "Has Argument Definition"@en ;
    rdfs:comment "Ordered signature entries."@en ;
    rdfs:domain n16:RelationDefinition ;
    rdfs:range n16:ArgumentDefinition .

n16:hasRelationPattern a owl:ObjectProperty ;
    rdfs:label "Has Relation Pattern"@en ;
    rdfs:comment "Relation pattern available in this action description."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n16:RelationPattern ;
    rdfs:subPropertyOf ac:hasComponent .

n16:usesRelation a owl:ObjectProperty ;
    rdfs:label "Uses Relation"@en ;
    rdfs:comment "Predicate used by a symbolic relation pattern."@en ;
    rdfs:domain n16:RelationPattern ;
    rdfs:range n16:RelationDefinition .

n17:RecoveryPhase a owl:Class ;
    rdfs:label "Recovery Phase"@en ;
    rdfs:comment "Phase describing recovery, without executing a recovery controller."@en ;
    rdfs:subClassOf ac:PhaseSpecification .

n17:entryCondition a owl:ObjectProperty ;
    rdfs:label "Entry Condition"@en ;
    rdfs:comment "Declared phase-entry requirement; guard assessment alone does not enforce it."@en ;
    rdfs:domain ac:PhaseSpecification ;
    rdfs:range ac:ConditionSpecification .

n17:exitCondition a owl:ObjectProperty ;
    rdfs:label "Exit Condition"@en ;
    rdfs:comment "Declared phase-completion requirement."@en ;
    rdfs:domain ac:PhaseSpecification ;
    rdfs:range ac:ConditionSpecification .

n17:expectedOutcome a owl:ObjectProperty ;
    rdfs:label "Expected Outcome"@en ;
    rdfs:comment "Intended outcome using the existing conditions/outcomes module."@en ;
    rdfs:domain ac:PhaseSpecification ;
    rdfs:range n3:OutcomeSpecification .

n17:fromPhase a owl:ObjectProperty ;
    rdfs:label "From Phase"@en ;
    rdfs:comment "Source phase."@en ;
    rdfs:domain n17:PhaseTransition ;
    rdfs:range ac:PhaseSpecification .

n17:guardCondition a owl:ObjectProperty ;
    rdfs:label "Guard Condition"@en ;
    rdfs:comment "Condition whose truth makes this transition a candidate; no priority or execution is implied."@en ;
    rdfs:domain n17:PhaseTransition ;
    rdfs:range ac:ConditionSpecification .

n17:hasMemberPhase a owl:ObjectProperty ;
    rdfs:label "Has Member Phase"@en ;
    rdfs:comment "Phases in this graph; shared phase specifications are permitted."@en ;
    rdfs:domain n17:PhaseStructure ;
    rdfs:range ac:PhaseSpecification .

n17:hasPhaseStructure a owl:ObjectProperty ;
    rdfs:label "Has Phase Structure"@en ;
    rdfs:comment "Links a shared action description to a phase graph."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n17:PhaseStructure ;
    rdfs:subPropertyOf ac:hasComponent .

n17:hasTransition a owl:ObjectProperty ;
    rdfs:label "Has Transition"@en ;
    rdfs:comment "Transitions owned by this graph."@en ;
    rdfs:domain n17:PhaseStructure ;
    rdfs:range n17:PhaseTransition .

n17:initialPhase a owl:ObjectProperty ;
    rdfs:label "Initial Phase"@en ;
    rdfs:comment "Exactly one initial phase in the validation profile."@en ;
    rdfs:domain n17:PhaseStructure ;
    rdfs:range ac:PhaseSpecification .

n17:invariantCondition a owl:ObjectProperty ;
    rdfs:label "Invariant Condition"@en ;
    rdfs:comment "Condition intended to hold during a phase; observations do not prove continuous validity."@en ;
    rdfs:domain ac:PhaseSpecification ;
    rdfs:range ac:ConditionSpecification .

n17:toPhase a owl:ObjectProperty ;
    rdfs:label "To Phase"@en ;
    rdfs:comment "Target phase."@en ;
    rdfs:domain n17:PhaseTransition ;
    rdfs:range ac:PhaseSpecification .

n18:Capacity a n18:PropertyKind,
        owl:NamedIndividual ;
    rdfs:label "Capacity"@en .

n18:Density a n18:PropertyKind,
        owl:NamedIndividual ;
    rdfs:label "Density"@en .

n18:FrictionCoefficient a n18:PropertyKind,
        owl:NamedIndividual ;
    rdfs:label "Friction Coefficient"@en .

n18:Geometry a n18:PropertyKind,
        owl:NamedIndividual ;
    rdfs:label "Geometry"@en .

n18:Mass a n18:PropertyKind,
        owl:NamedIndividual ;
    rdfs:label "Mass"@en .

n18:Viscosity a n18:PropertyKind,
        owl:NamedIndividual ;
    rdfs:label "Viscosity"@en .

n18:applicability a owl:ObjectProperty ;
    rdfs:label "Applicability"@en ;
    rdfs:comment "Model validity scope."@en ;
    rdfs:domain n18:PhysicalModelSpecification ;
    rdfs:range ac:ApplicabilitySpecification .

n18:artifactIRI a owl:DatatypeProperty ;
    rdfs:label "Artifact IRI"@en ;
    rdfs:comment "External artifact identifier."@en ;
    rdfs:domain n18:ModelArtifact ;
    rdfs:range xsd:anyURI .

n18:artifactVersion a owl:DatatypeProperty ;
    rdfs:label "Artifact Version"@en ;
    rdfs:comment "Version or content digest."@en ;
    rdfs:domain n18:ModelArtifact ;
    rdfs:range xsd:string .

n18:assumption a owl:ObjectProperty ;
    rdfs:label "Assumption"@en ;
    rdfs:comment "Physical-model assumption."@en ;
    rdfs:domain n18:PhysicalModelSpecification ;
    rdfs:range ac:ConditionSpecification .

n18:describesEntity a owl:ObjectProperty ;
    rdfs:label "Describes Entity"@en ;
    rdfs:comment "Conceptual entity whose property is described."@en ;
    rdfs:domain n18:PhysicalPropertySpecification ;
    rdfs:range ac:EntitySpecification .

n18:expressionLanguage a owl:DatatypeProperty ;
    rdfs:label "Expression Language"@en ;
    rdfs:comment "Language used by the equation."@en ;
    rdfs:domain n18:ModelEquation ;
    rdfs:range xsd:string .

n18:expressionText a owl:DatatypeProperty ;
    rdfs:label "Expression Text"@en ;
    rdfs:comment "Equation text; no symbolic execution."@en ;
    rdfs:domain n18:ModelEquation ;
    rdfs:range xsd:string .

n18:hasArtifact a owl:ObjectProperty ;
    rdfs:label "Has Artifact"@en ;
    rdfs:comment "External implementation or geometry artifact."@en ;
    rdfs:domain n18:PhysicalModelSpecification ;
    rdfs:range n18:ModelArtifact .

n18:hasEquation a owl:ObjectProperty ;
    rdfs:label "Has Equation"@en ;
    rdfs:comment "Equation specifications."@en ;
    rdfs:domain n18:PhysicalModelSpecification ;
    rdfs:range n18:ModelEquation .

n18:hasPhysicalProperty a owl:ObjectProperty ;
    rdfs:label "Has Physical Property"@en ;
    rdfs:comment "Property descriptions available to this action."@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n18:PhysicalPropertySpecification ;
    rdfs:subPropertyOf ac:hasComponent .

n18:hasVariable a owl:ObjectProperty ;
    rdfs:label "Has Variable"@en ;
    rdfs:comment "Declared model variables."@en ;
    rdfs:domain n18:PhysicalModelSpecification ;
    rdfs:range n18:ModelVariable .

n18:parameter a owl:ObjectProperty ;
    rdfs:label "Parameter"@en ;
    rdfs:comment "Reusable parameter represented by the variable."@en ;
    rdfs:domain n18:ModelVariable ;
    rdfs:range ac:ParameterSpecification .

n18:propertyKind a owl:ObjectProperty ;
    rdfs:label "Property Kind"@en ;
    rdfs:comment "Property concept."@en ;
    rdfs:domain n18:PhysicalPropertySpecification ;
    rdfs:range n18:PropertyKind .

n18:realizesMechanism a owl:ObjectProperty ;
    rdfs:label "Realizes Mechanism"@en ;
    rdfs:comment "Model intended to represent this mechanism; not a proof."@en ;
    rdfs:domain n18:PhysicalModelSpecification ;
    rdfs:range n2:MechanismSpecification .

n18:studiesModel a owl:ObjectProperty ;
    rdfs:label "Studies Model"@en ;
    rdfs:comment "Model proposed for study."@en ;
    rdfs:domain n18:SimulationStudySpecification ;
    rdfs:range n18:PhysicalModelSpecification .

n18:symbol a owl:DatatypeProperty ;
    rdfs:label "Symbol"@en ;
    rdfs:comment "Variable symbol, unique within a model."@en ;
    rdfs:domain n18:ModelVariable ;
    rdfs:range xsd:string .

n18:testsOutcome a owl:ObjectProperty ;
    rdfs:label "Tests Outcome"@en ;
    rdfs:comment "Outcome to investigate."@en ;
    rdfs:domain n18:SimulationStudySpecification ;
    rdfs:range n3:OutcomeSpecification .

n18:usesVariable a owl:ObjectProperty ;
    rdfs:label "Uses Variable"@en ;
    rdfs:comment "Declared variables used by an equation."@en ;
    rdfs:domain n18:ModelEquation ;
    rdfs:range n18:ModelVariable .

n18:validatesModel a owl:ObjectProperty ;
    rdfs:label "Validates Model"@en ;
    rdfs:comment "Model evaluated by the external artifact."@en ;
    rdfs:domain n18:ModelValidationRecord ;
    rdfs:range n18:PhysicalModelSpecification .

n18:validationArtifactIRI a owl:DatatypeProperty ;
    rdfs:label "Validation Artifact IRI"@en ;
    rdfs:comment "External validation record or run identifier."@en ;
    rdfs:domain n18:ModelValidationRecord ;
    rdfs:range xsd:anyURI .

n18:validationScope a owl:ObjectProperty ;
    rdfs:label "Validation Scope"@en ;
    rdfs:comment "Scope of the recorded evaluation."@en ;
    rdfs:domain n18:ModelValidationRecord ;
    rdfs:range ac:ApplicabilitySpecification .

n18:valueParameter a owl:ObjectProperty ;
    rdfs:label "Value Parameter"@en ;
    rdfs:comment "Parameter containing optional values, bounds and units."@en ;
    rdfs:domain n18:PhysicalPropertySpecification ;
    rdfs:range ac:ParameterSpecification .

n0:pouring-action.owl a owl:Ontology ;
    rdfs:label "Pouring action knowledge profile"@en ;
    rdfs:comment "Action-specific entry ontology for comprehensive pouring descriptions. Reuses the existing material-transfer PouringDescription and Pouring action kind. Imports expose representation vocabulary; they do not assert that every pour requires pickup, tool contact, every aspect, or a physical controller. This profile remains world-independent and its current bounded pouring template describes partial transfer."@en ;
    rdfs:seeAlso mt:Pouring,
        mt:PouringDescription ;
    owl:versionInfo "0.23.0" .

n19:CartesianPositioning a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Cartesian positioning"@en .

n19:GripperApertureChange a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Gripper aperture change"@en .

n19:GripperConfigurationChange a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Gripper configuration change"@en .

n19:JointRepositioning a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Joint repositioning"@en .

n19:PosePathFollowing a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Pose path following"@en .

n19:Stowing a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Stowing"@en .

n19:accelerationLimit a owl:ObjectProperty ;
    rdfs:label "Acceleration limit"@en ;
    rdfs:domain n19:MotionBounds ;
    rdfs:range ac:ParameterSpecification .

n19:bounds a owl:ObjectProperty ;
    rdfs:label "Bounds"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range n19:MotionBounds ;
    rdfs:subPropertyOf ac:hasComponent .

n19:configurationTarget a owl:ObjectProperty ;
    rdfs:label "Configuration target"@en ;
    rdfs:domain n19:ConfigurationMotionDescription ;
    rdfs:range n19:ConfigurationTarget ;
    rdfs:subPropertyOf ac:hasComponent .

n19:controlledEntity a owl:ObjectProperty ;
    rdfs:label "Controlled entity"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range ac:EntitySpecification .

n19:coordinate a owl:ObjectProperty ;
    rdfs:label "Coordinate"@en ;
    rdfs:domain n19:CoordinateTarget ;
    rdfs:range n19:CoordinateSpecification .

n19:coordinateKey a owl:DatatypeProperty ;
    rdfs:label "Coordinate key"@en ;
    rdfs:domain n19:CoordinateSpecification ;
    rdfs:range xsd:string .

n19:coordinateKind a owl:DatatypeProperty ;
    rdfs:label "Coordinate kind"@en ;
    rdfs:domain n19:CoordinateSpecification ;
    rdfs:range xsd:string .

n19:endpointErrorParameter a owl:ObjectProperty ;
    rdfs:label "Endpoint error parameter"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range ac:ParameterSpecification .

n19:endpointToleranceParameter a owl:ObjectProperty ;
    rdfs:label "Endpoint tolerance parameter"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range ac:ParameterSpecification .

n19:errorParameter a owl:ObjectProperty ;
    rdfs:label "Error parameter"@en ;
    rdfs:domain n19:CoordinateTarget ;
    rdfs:range ac:ParameterSpecification .

n19:hasCoordinateTarget a owl:ObjectProperty ;
    rdfs:label "Has coordinate target"@en ;
    rdfs:domain n19:ConfigurationTarget ;
    rdfs:range n19:CoordinateTarget .

n19:hasWaypoint a owl:ObjectProperty ;
    rdfs:label "Has waypoint"@en ;
    rdfs:domain n19:PathSpecification ;
    rdfs:range n19:WaypointSpecification .

n19:orientation a owl:ObjectProperty ;
    rdfs:label "Orientation"@en ;
    rdfs:domain n19:PoseTarget ;
    rdfs:range n19:OrientationTarget .

n19:orientationConvention a owl:DatatypeProperty ;
    rdfs:label "Orientation convention"@en ;
    rdfs:domain n19:OrientationTarget ;
    rdfs:range xsd:string .

n19:orientationErrorParameter a owl:ObjectProperty ;
    rdfs:label "Orientation error parameter"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range ac:ParameterSpecification .

n19:orientationToleranceParameter a owl:ObjectProperty ;
    rdfs:label "Orientation tolerance parameter"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range ac:ParameterSpecification .

n19:pathTarget a owl:ObjectProperty ;
    rdfs:label "Path target"@en ;
    rdfs:domain n19:PathFollowingDescription ;
    rdfs:range n19:PathSpecification ;
    rdfs:subPropertyOf ac:hasComponent .

n19:poseTarget a owl:ObjectProperty ;
    rdfs:label "Pose target"@en ;
    rdfs:domain n19:CartesianMotionDescription ;
    rdfs:range n19:PoseTarget ;
    rdfs:subPropertyOf ac:hasComponent .

n19:position a owl:ObjectProperty ;
    rdfs:label "Position"@en ;
    rdfs:domain n19:PoseTarget ;
    rdfs:range n19:PositionTarget .

n19:positionIndex a owl:DatatypeProperty ;
    rdfs:label "Position index"@en ;
    rdfs:domain n19:WaypointSpecification ;
    rdfs:range xsd:positiveInteger .

n19:qwParameter a owl:ObjectProperty ;
    rdfs:label "Qw parameter"@en ;
    rdfs:domain n19:OrientationTarget ;
    rdfs:range ac:ParameterSpecification .

n19:qxParameter a owl:ObjectProperty ;
    rdfs:label "Qx parameter"@en ;
    rdfs:domain n19:OrientationTarget ;
    rdfs:range ac:ParameterSpecification .

n19:qyParameter a owl:ObjectProperty ;
    rdfs:label "Qy parameter"@en ;
    rdfs:domain n19:OrientationTarget ;
    rdfs:range ac:ParameterSpecification .

n19:qzParameter a owl:ObjectProperty ;
    rdfs:label "Qz parameter"@en ;
    rdfs:domain n19:OrientationTarget ;
    rdfs:range ac:ParameterSpecification .

n19:referenceFrame a owl:ObjectProperty ;
    rdfs:label "Reference frame"@en ;
    rdfs:domain n19:PoseTarget ;
    rdfs:range n19:FrameSpecification .

n19:speedLimit a owl:ObjectProperty ;
    rdfs:label "Speed limit"@en ;
    rdfs:domain n19:MotionBounds ;
    rdfs:range ac:ParameterSpecification .

n19:targetParameter a owl:ObjectProperty ;
    rdfs:label "Target parameter"@en ;
    rdfs:domain n19:CoordinateTarget ;
    rdfs:range ac:ParameterSpecification .

n19:toleranceParameter a owl:ObjectProperty ;
    rdfs:label "Tolerance parameter"@en ;
    rdfs:domain n19:CoordinateTarget ;
    rdfs:range ac:ParameterSpecification .

n19:trackingErrorParameter a owl:ObjectProperty ;
    rdfs:label "Tracking error parameter"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range ac:ParameterSpecification .

n19:trackingToleranceParameter a owl:ObjectProperty ;
    rdfs:label "Tracking tolerance parameter"@en ;
    rdfs:domain n19:RobotMotionDescription ;
    rdfs:range ac:ParameterSpecification .

n19:waypointPose a owl:ObjectProperty ;
    rdfs:label "Waypoint pose"@en ;
    rdfs:domain n19:WaypointSpecification ;
    rdfs:range n19:PoseTarget .

n19:xParameter a owl:ObjectProperty ;
    rdfs:label "X parameter"@en ;
    rdfs:domain n19:PositionTarget ;
    rdfs:range ac:ParameterSpecification .

n19:yParameter a owl:ObjectProperty ;
    rdfs:label "Y parameter"@en ;
    rdfs:domain n19:PositionTarget ;
    rdfs:range ac:ParameterSpecification .

n19:zParameter a owl:ObjectProperty ;
    rdfs:label "Z parameter"@en ;
    rdfs:domain n19:PositionTarget ;
    rdfs:range ac:ParameterSpecification .

n20:Auditory a n20:SensoryModality,
        owl:NamedIndividual ;
    rdfs:label "Auditory"@en .

n20:ForceTorque a n20:SensoryModality,
        owl:NamedIndividual ;
    rdfs:label "Force Torque"@en .

n20:Proprioceptive a n20:SensoryModality,
        owl:NamedIndividual ;
    rdfs:label "Proprioceptive"@en .

n20:Tactile a n20:SensoryModality,
        owl:NamedIndividual ;
    rdfs:label "Tactile"@en .

n20:Visual a n20:SensoryModality,
        owl:NamedIndividual ;
    rdfs:label "Visual"@en .

n20:afterPhase a owl:ObjectProperty ;
    rdfs:label "After Phase"@en ;
    rdfs:domain n20:EventBoundarySpecification ;
    rdfs:range ac:PhaseSpecification .

n20:atBoundary a owl:ObjectProperty ;
    rdfs:label "At Boundary"@en ;
    rdfs:domain n20:ControlPointSpecification ;
    rdfs:range n20:EventBoundarySpecification .

n20:beforePhase a owl:ObjectProperty ;
    rdfs:label "Before Phase"@en ;
    rdfs:domain n20:EventBoundarySpecification ;
    rdfs:range ac:PhaseSpecification .

n20:boundaryCondition a owl:ObjectProperty ;
    rdfs:label "Boundary Condition"@en ;
    rdfs:domain n20:EventBoundarySpecification ;
    rdfs:range ac:ConditionSpecification .

n20:discrepancyCondition a owl:ObjectProperty ;
    rdfs:label "Discrepancy Condition"@en ;
    rdfs:domain n20:MismatchSpecification ;
    rdfs:range ac:ConditionSpecification .

n20:expectedCondition a owl:ObjectProperty ;
    rdfs:label "Expected Condition"@en ;
    rdfs:domain n20:SensoryExpectation ;
    rdfs:range ac:ConditionSpecification .

n20:hasExpectation a owl:ObjectProperty ;
    rdfs:label "Has Expectation"@en ;
    rdfs:domain n20:ControlPointSpecification ;
    rdfs:range n20:SensoryExpectation .

n20:mismatchCriterion a owl:ObjectProperty ;
    rdfs:label "Mismatch Criterion"@en ;
    rdfs:domain n20:ControlPointSpecification ;
    rdfs:range n20:MismatchSpecification .

n20:modality a owl:ObjectProperty ;
    rdfs:label "Modality"@en ;
    rdfs:domain n20:SensoryExpectation ;
    rdfs:range n20:SensoryModality .

n20:observationRequirement a owl:ObjectProperty ;
    rdfs:label "Observation Requirement"@en ;
    rdfs:domain n20:ControlPointSpecification ;
    rdfs:range ac:CapabilityRequirement ;
    rdfs:subPropertyOf ac:requiresCapability .

n20:requiresCapability a owl:ObjectProperty ;
    rdfs:label "Requires Capability"@en ;
    rdfs:domain n20:SensoryExpectation ;
    rdfs:range ac:CapabilityRequirement ;
    rdfs:subPropertyOf ac:requiresCapability .

n21:DisengagingTool a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Disengaging tool"@en .

n21:EffectObserved a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Effect observed"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n21:EffectObservedArgument1,
        n21:EffectObservedArgument2 ;
    rdfs:comment "Externally assessed, context-qualified EffectObserved condition; no physical guarantee inferred."@en .

n21:EngagingTool a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Engaging tool"@en .

n21:PositioningTool a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Positioning tool"@en .

n21:ToolAligned a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Tool aligned"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n21:ToolAlignedArgument1,
        n21:ToolAlignedArgument2 ;
    rdfs:comment "Externally assessed, context-qualified ToolAligned condition; no physical guarantee inferred."@en .

n21:ToolClear a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Tool clear"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n21:ToolClearArgument1,
        n21:ToolClearArgument2 ;
    rdfs:comment "Externally assessed, context-qualified ToolClear condition; no physical guarantee inferred."@en .

n21:ToolDisengagementDescription a owl:Class ;
    rdfs:label "Tool disengagement description"@en ;
    rdfs:comment "Separate and clear a tool from its target; completion does not establish task effect."@en ;
    rdfs:subClassOf n21:ToolInteractionDescription .

n21:ToolEngagementDescription a owl:Class ;
    rdfs:label "Tool engagement description"@en ;
    rdfs:comment "Establish a held tool contact under a declared contact policy."@en ;
    rdfs:subClassOf n21:ToolInteractionDescription .

n21:ToolFault a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Tool fault"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n21:ToolFaultArgument1,
        n21:ToolFaultArgument2 ;
    rdfs:comment "Externally assessed, context-qualified ToolFault condition; no physical guarantee inferred."@en .

n21:ToolFunctional a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Tool functional"@en ;
    n16:arity "1"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n21:ToolFunctionalArgument1 ;
    rdfs:comment "Externally assessed, context-qualified ToolFunctional condition; no physical guarantee inferred."@en .

n21:ToolHeld a n16:RelationDefinition,
        owl:NamedIndividual ;
    rdfs:label "Tool held"@en ;
    n16:arity "2"^^xsd:positiveInteger ;
    n16:hasArgumentDefinition n21:ToolHeldArgument1,
        n21:ToolHeldArgument2 ;
    rdfs:comment "Externally assessed, context-qualified ToolHeld condition; no physical guarantee inferred."@en .

n21:ToolPositioningDescription a owl:Class ;
    rdfs:label "Tool positioning description"@en ;
    rdfs:comment "Align a tool working part to its target; no contact or effect implied."@en ;
    rdfs:subClassOf n21:ToolInteractionDescription .

n21:UsingTool a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Using tool"@en .

n21:contactPolicy a owl:DatatypeProperty ;
    rdfs:label "Contact policy"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range xsd:string .

n21:evaluationContext a owl:ObjectProperty ;
    rdfs:label "Evaluation context"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range n11:KnowledgeContext .

n21:forceLimit a owl:ObjectProperty ;
    rdfs:label "Force limit"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range ac:ParameterSpecification .

n21:intendedEffect a owl:ObjectProperty ;
    rdfs:label "Intended effect"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range ac:EntitySpecification .

n21:operator a owl:ObjectProperty ;
    rdfs:label "Operator"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range ac:EntitySpecification .

n21:referenceFrame a owl:ObjectProperty ;
    rdfs:label "Reference frame"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range n19:FrameSpecification .

n21:target a owl:ObjectProperty ;
    rdfs:label "Target"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range ac:EntitySpecification .

n21:tool a owl:ObjectProperty ;
    rdfs:label "Tool"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range ac:EntitySpecification .

n21:workingPart a owl:ObjectProperty ;
    rdfs:label "Working part"@en ;
    rdfs:domain n21:ToolInteractionDescription ;
    rdfs:range ac:EntitySpecification .

n21:workingPartOf a owl:ObjectProperty ;
    rdfs:label "Working part of"@en ;
    rdfs:domain ac:EntitySpecification ;
    rdfs:range ac:EntitySpecification .

n22:deliveredAmount a owl:ObjectProperty ;
    rdfs:label "Delivered amount"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:ParameterSpecification .

n22:destination a owl:ObjectProperty ;
    rdfs:label "Destination"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:EntitySpecification .

n22:source a owl:ObjectProperty ;
    rdfs:label "Source"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:EntitySpecification .

n22:spillAmount a owl:ObjectProperty ;
    rdfs:label "Spill amount"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:ParameterSpecification .

n22:spillTolerance a owl:ObjectProperty ;
    rdfs:label "Spill tolerance"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:ParameterSpecification .

n22:substance a owl:ObjectProperty ;
    rdfs:label "Substance"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:EntitySpecification .

n22:successCriterion a owl:ObjectProperty ;
    rdfs:label "Success criterion"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:SuccessCriterion .

n22:targetAmount a owl:ObjectProperty ;
    rdfs:label "Target amount"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:ParameterSpecification .

n22:transferMode a owl:DatatypeProperty ;
    rdfs:label "Transfer mode"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range xsd:string .

n22:transferredPortion a owl:ObjectProperty ;
    rdfs:label "Transferred portion"@en ;
    rdfs:domain n22:TransferContract ;
    rdfs:range ac:EntitySpecification .

n24:Amount a owl:Class ;
    rdfs:label "Amount"@en,
        "Quantità"@it ;
    rdfs:comment "A quantity, independently from how it is measured, computed, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Region ;
    owl:disjointWith n24:PhysicalAttribute,
        n24:SpaceRegion,
        n24:TimeInterval .

n24:ChemicalObject a owl:Class ;
    rdfs:label "Chemical object"@en ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:PhysicalBody .

n24:Classification a owl:Class ;
    rdfs:label "Classification"@en,
        "Classificazione"@it ;
    rdfs:comment """A special kind of Situation that allows to include time indexing for the classifies relation in situations. For example, if a Situation s 'my old cradle is used in these days as a flower pot' isSettingFor the entity 'my old cradle' and the TimeIntervals '8June2007' and '10June2007', and we know that s satisfies a functional Description for aesthetic objects, which defines the Concepts 'flower pot' and 'flower', then we also need to know what concept classifies 'my old cradle' at what time.
In order to solve this issue, we need to create a sub-situation s' for the classification time: 'my old cradle is a flower pot in 8June2007'. Such sub-situation s' isPartOf s.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:isSettingFor ;
            owl:someValuesFrom n24:Concept ],
        [ a owl:Restriction ;
            owl:onProperty n24:isSettingFor ;
            owl:someValuesFrom n24:TimeInterval ],
        n24:Situation .

n24:Collective a owl:Class ;
    rdfs:label "Collective"@en,
        "Collettivo"@it ;
    rdfs:comment """A Collection whose members are agents, e.g. "the nurses", "the Italian rockabilly fans".
Collectives, facon de parler, can act as agents, although they are not assumed here to be agents (they are even disjoint from the class SocialAgent). This is represented by admitting collectives in the range of the relations having Agent in their domain or range.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Agent ;
            owl:onProperty n24:hasMember ],
        n24:Collection .

n24:Community a owl:Class ;
    rdfs:label "Community"@en,
        "Comunità"@it ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:CollectiveAgent .

n24:Configuration a owl:Class ;
    rdfs:label "Configuration"@en,
        "Configurazione"@it ;
    rdfs:comment """A collection whose members are 'unified', i.e. organized according to a certain schema that can be represented by a Description.
Typically, a configuration is the collection that emerges out of a composed entity: an industrial artifact, a plan, a discourse, etc.  
E.g. a physical book has a configuration provided by the part-whole schema that holds together its cover, pages, ink. That schema, based on the individual relations between the book and its parts, can be represented in a reified way by means of a (structural) description, which is said to 'unify' the book configuration.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Collection .

n24:Contract a owl:Class ;
    rdfs:label "Contract"@en,
        "Contratto"@it ;
    rdfs:comment "(The content of) an agreement between at least two agents that play a Party Role, about some contract object (a Task to be executed)." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:DesignedSubstance a owl:Class ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:DesignedArtifact,
        n24:FunctionalSubstance .

n24:Diagnosis a owl:Class ;
    rdfs:label "Diagnosis"@en,
        "Diagnosi"@it ;
    rdfs:comment "A Description of the Situation of a system, usually applied in order to control a normal behaviour, or to explain a notable behavior (e.g. a functional breakdown)." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:Group a owl:Class ;
    rdfs:label "Group"@en,
        "Gruppo"@it ;
    rdfs:comment "A CollectiveAgent whose acting agents conceptualize a same SocialRelation ." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:isDescribedBy ;
            owl:someValuesFrom n24:Plan ],
        n24:CollectiveAgent .

n24:LocalConcept a owl:Class ;
    rdfs:label "Local concept"@en ;
    rdfs:comment "A Concept that isDefinedIn exactly 1 Description. For example, the Concept 'coffee' in a 'preparesCoffee' relation can be defined in that relation, and for all other Description(s) that use it, the isConceptUsedIn property should be applied. Notice therefore that not necessarily all Concept(s) isDefinedIn exactly 1 Description." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Concept .

n24:Method a owl:Class ;
    rdfs:label "Method"@en,
        "Metodo"@it ;
    rdfs:comment """A method is a Description that defines or uses concepts in order to guide carrying out actions aimed at a solution with respect to a problem. 
It is different from a Plan, because plans could be carried out in order to follow a method, but a method can be followed by executing alternative plans.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:Narrative a owl:Class ;
    rdfs:label "Narrative"@en ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:NaturalPerson a owl:Class ;
    rdfs:label "Natural person"@en,
        "Persona fisica"@it ;
    rdfs:comment "A person in the physical commonsense intuition: 'have you seen that person walking down the street?'" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Person,
        n24:PhysicalAgent .

n24:Norm a owl:Class ;
    rdfs:label "Norm"@en,
        "Norma"@it ;
    rdfs:comment "A social norm." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:Organism a owl:Class ;
    rdfs:label "Organism"@en,
        "Organismo"@it ;
    rdfs:comment "A physical objects with biological characteristics, typically that organisms can self-reproduce." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:BiologicalObject,
        n24:PhysicalAgent .

n24:Organization a owl:Class ;
    rdfs:label "Organization"@en,
        "Organizzazione"@it ;
    rdfs:comment "An internally structured, conventionally created SocialAgent, needing a specific Role and Agent that plays it, in order to act."@en,
        "Un agente sociale strutturato internamente e creato convenzionalmente. Per agire, ha bisogno di ruoli e agenti che li ricoprano."@it ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:SocialAgent .

n24:Pattern a owl:Class ;
    rdfs:label "Pattern"@en ;
    rdfs:comment """Any invariance detected from a dataset, or from observation; also, any invariance proposed based on top-down considerations.
E.g. patterns detected and abstracted by an organism, by pattern recognition algorithms, by machine learning techniques, etc.
An occurrence of a pattern is an 'observable', or detected Situation""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Relation .

n24:Personification a owl:Class ;
    rdfs:label "Personification"@en ;
    rdfs:comment "A social entity with agentive features, but whose status is the result of a cultural transformation from e.g. a PhysicalObject, an Event, an Abstract, another SocialObject, etc. For example: the holy grail, deus ex machina, gods, magic wands, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:SocialAgent .

n24:PhysicalPlace a owl:Class ;
    rdfs:label "Physical place"@en,
        "Luogo fisico"@it ;
    rdfs:comment "A physical object that is inherently located; for example, a water area." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:PhysicalObject .

n24:Place a owl:Class ;
    rdfs:label "Place"@en,
        "Luogo"@it ;
    rdfs:comment """A location, in a very generic sense: a political geographic entity (Roma, Lesotho), a non-material location determined by the presence of other entities ("the area close to Roma"), pivot events or signs ("the area where the helicopter fell"), complements of other entities ("the area under the table"), etc. 
In this generic sense, a Place is an "approximate" location. For an "absolute" location, see the class SpaceRegion""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:minCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:isLocationOf ],
        n24:SocialObject .

n24:PlanExecution a owl:Class ;
    rdfs:label "Plan execution"@en,
        "Esecuzione di piano"@it ;
    rdfs:comment "Plan executions are situations that proactively satisfy a plan. Subplan executions are proper parts of the whole plan execution." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Situation ;
    owl:equivalentClass [ a owl:Restriction ;
            owl:onProperty n24:satisfies ;
            owl:someValuesFrom n24:Plan ] .

n24:Project a owl:Class ;
    rdfs:label "Project"@en,
        "Progetto"@it ;
    rdfs:comment "A Plan that defines Role(s), Task(s), and a specific structure for tasks to be executed in relation to goals to be achieved, in order to achieve the main goal of the project. In other words, a project is a plan with a subgoal structure and multiple roles and tasks." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:definesRole ;
            owl:someValuesFrom n24:Role ],
        [ a owl:Restriction ;
            owl:onProperty n24:definesTask ;
            owl:someValuesFrom n24:Task ],
        n24:Plan .

n24:Right a owl:Class ;
    rdfs:label "Right"@en,
        "Diritto"@it ;
    rdfs:comment "A legal position by which an Agent is entitled to obtain something from another Agent , under specified circumstances, through an enforcement explicited either in a Law, Contract , etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:minCardinality "2"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:definesRole ],
        [ a owl:Restriction ;
            owl:minCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:definesTask ],
        n24:Description .

n24:Set a owl:Class ;
    rdfs:label "Insieme {it}",
        "Set"@en ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:FormalEntity .

n24:SocialObjectAttribute a owl:Class ;
    rdfs:label "Social attribute"@en,
        "Caratteristica sociale"@it ;
    rdfs:comment "Any Region in a dimensional space that is used to represent some characteristic of a SocialObject, e.g. judgment values, social scalars, statistical attributes over a collection of entities, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:SocialObject ;
            owl:onProperty n24:isRegionFor ],
        n24:Region .

n24:SocialPerson a owl:Class ;
    rdfs:label "Social person"@en,
        "Persona sociale"@it ;
    rdfs:comment "A SocialAgent that needs the existence of a specific NaturalPerson in order to act (but the lifetime of the NaturalPerson has only to overlap that of the SocialPerson)." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:cardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:actsThrough ],
        n24:Person,
        n24:SocialAgent ;
    owl:versionInfo "Formerly: Person (changed to avoid confusion with commonsense intuition)" .

n24:SocialRelation a owl:Class ;
    rdfs:label "Social relation"@en,
        "Relazione sociale"@it ;
    rdfs:comment "Any social relationship" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Relation .

n24:SpatioTemporalRegion a owl:Class ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:hasConstituent ;
            owl:someValuesFrom n24:SpaceRegion ],
        [ a owl:Restriction ;
            owl:onProperty n24:hasConstituent ;
            owl:someValuesFrom n24:TimeInterval ],
        n24:Region .

n24:Theory a owl:Class ;
    rdfs:label "Theory"@en,
        "Teoria"@it ;
    rdfs:comment """A Theory is a Description that represents a set of assumptions for describing something, usually general. Scientific, philosophical, and commonsense theories can be included here.
This class can also be used to act as 'naturalized reifications' of logical theories (of course, they will be necessarily incomplete in this case, because second-order entities are represented as first-order ones).""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:hasComponent ;
            owl:someValuesFrom n24:Relation ],
        n24:Description .

n24:Transition a owl:Class ;
    rdfs:label "Transition"@en,
        "Transizione"@it ;
    rdfs:comment """A transition is a Situation that creates a context for three TimeInterval(s), two additional different Situation(s), one Event, one Process, and at least one Object: the Event is observed as the cause for the transition, one Situation is the state before the transition, the second Situation is the state after the transition, the Process is the invariance under some different transitions (including the one represented here), in which at least one Object is situated. Finally, the time intervals position the situations and the transitional event in time.
This class of situations partly encodes the ontology underlying typical engineering algebras for processes, e.g. Petri Nets. 
A full representation of the transition ontology is outside the expressivity of OWL, because we would need qualified cardinality restrictions,  coreference, property equivalence, and property composition.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:minCardinality "2"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:includesEvent ],
        [ a owl:Restriction ;
            owl:minCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:includesObject ],
        [ a owl:Restriction ;
            owl:minCardinality "3"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:includesTime ],
        [ a owl:Restriction ;
            owl:onProperty n24:isSettingFor ;
            owl:someValuesFrom n24:Situation ],
        [ a owl:Restriction ;
            owl:onProperty n24:isSettingFor ;
            owl:someValuesFrom n24:Process ],
        n24:Situation .

n24:TypeCollection a owl:Class ;
    rdfs:label "Type collection"@en,
        "Collezione di un tipo"@it ;
    rdfs:comment """A Collection whose members are the maximal set of individuals that share the same (named) type, e.g. "the gem stones", "the Italians".
This class is very useful to apply a variety of the so-called "ClassesAsValues" design pattern, when it is used to talk about the extensional aspect of a class. An alternative variety of the pattern applies to the intensional aspect of a class, and the class Concept should be used instead.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Collection .

n24:UnitOfMeasure a owl:Class ;
    rdfs:label "Unit of measure"@en,
        "Unità di misura"@it ;
    rdfs:comment "Units of measure are conceptualized here as parameters on regions, which can be valued as datatype values." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:parametrizes ;
            owl:someValuesFrom n24:Region ],
        n24:Parameter .

n24:WorkflowExecution a owl:Class ;
    rdfs:label "Workflow execution"@en,
        "Esecuzione di workflow"@it ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Situation ;
    owl:equivalentClass [ a owl:Restriction ;
            owl:onProperty n24:satisfies ;
            owl:someValuesFrom n24:Workflow ] .

n24:actsFor a owl:ObjectProperty ;
    rdfs:label "acts for"@en,
        "agisce per"@it ;
    rdfs:comment "The relation holding between any Agent, and a SocialAgent. In principle, a SocialAgent requires at least one PhysicalAgent in order to act, but this dependency can be 'delegated'; e.g. a university can be acted for by a department, which on its turm is acted for by physical agents." ;
    rdfs:domain n24:Agent ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:SocialAgent ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:actsThrough .

n24:characterizes a owl:ObjectProperty ;
    rdfs:label "characterizes"@en,
        "caratterizza"@it ;
    rdfs:comment """A relation between concepts and collections, where a Concept is said to characterize a Collection; it corresponds to a link between the (reified) intensional and extensional interpretations of a _proper subset of_ a (reified) class. This is different from covers, because it refers to an interpretation the entire reified class.
E.g. the collection of vintage saxophones is characterized by the Concept 'manufactured by hand', while it gets covered by the Concept 'Saxophone' with the Parameter 'Vintage'.""" ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Collection ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isCharacterizedBy .

n24:conceptualizes a owl:ObjectProperty ;
    rdfs:label "conceptualizes"@en,
        "concettualizza"@it ;
    rdfs:comment """A relation stating that an Agent is internally representing a SocialObject: situations, descriptions, concepts, etc. E.g., 'John believes in the conspiracy theory'; 'Niels Bohr created the solar-system metaphor for the atomic theory'; 'Jacques assumes all swans are white'; 'the task force members share the attack plan'.
Conceptualizations can be distinguished into different forms, primarily based on the type of SocialObject that is conceptualized. Descriptions and concepts can be 'assumed', situations can be 'believed' or 'known', plans can be 'adopted', etc. (see ontology: http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl.""" ;
    rdfs:domain n24:Agent ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:SocialObject ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isConceptualizedBy .

n24:concretelyExpresses a owl:ObjectProperty ;
    rdfs:label "concretely expresses"@en,
        "esprime concretamente"@it ;
    rdfs:comment "A relation between an InformationRealization and a Description, e.g. 'the printout of the Italian Constitution concretelyExpresses the Italian Constitution'. It should be supplied also with a rule stating that the InformationRealization realizes an InformationObject that expresses the Description" ;
    rdfs:domain n24:InformationRealization ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:SocialObject ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isConcretelyExpressedBy .

n24:covers a owl:ObjectProperty ;
    rdfs:label "covers"@en,
        "ricopre"@it ;
    rdfs:comment """A relation between concepts and collections, where a Concept is said to cover a Collection; it corresponds to a link between the (reified) intensional and extensional interpretations of a (reified) class.
E.g. the collection of vintage saxophones is covered by the Concept 'Saxophone' with the Parameter 'Vintage'.""" ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Collection ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isCoveredBy .

n24:describes a owl:ObjectProperty ;
    rdfs:label "describes"@en,
        "descrive"@it ;
    rdfs:comment """The relation between a Description and an Entity : a Description gives a unity to a Collection of parts (the components), or constituents, by assigning a Role to each of them in the context of a whole Object (the system).
A same Entity can be given different descriptions, for example, an old cradle can be given a unifying Description based on the original aesthetic design, the functionality it was built for, or a new aesthetic functionality in which it can be used as a flower pot.""" ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isDescribedBy .

n24:expands a owl:ObjectProperty ;
    rdfs:label "expands"@en,
        "espande"@it ;
    rdfs:comment """A partial order relation that holds between descriptions. It represents the proper part relation between a description and another description featuring the same properties as the former, with at least one additional one.
Descriptions can be expanded either by adding other descriptions as parts, or by refining concepts that are used by them. 
An 'intention' to expand must be present (unless purely formal theories are considered, but even in this case a criterion of relevance is usually active).""" ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:isRelatedToDescription ;
    owl:inverseOf n24:isExpandedIn .

n24:expressesConcept a owl:ObjectProperty ;
    rdfs:label "expresses concept"@en,
        "esprime il concetto"@it ;
    rdfs:comment "A relation between an InformationObject and a Concept , e.g. the term \"dog\" expresses the Concept \"dog\". For expressing a relational meaning, see the more general object property: expresses" ;
    rdfs:domain n24:InformationObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:expresses ;
    owl:inverseOf n24:isConceptExpressedBy .

n24:hasConstraint a owl:ObjectProperty ;
    rdfs:label "has constraint"@en,
        "ha vincolo"@it ;
    rdfs:comment """A relation between parameters and entities. It allows to assert generic constraints (encoded as parameters), e.g. MinimumAgeForDriving isConstraintFor John (where John is a legal subject under the TrafficLaw).
The intended semantics (not expressible in OWL) is that a Parameter isParameterFor a Concept that classifies an Entity; moreover, it entails that a Parameter parametrizes a Region that isRegionFor that Entity.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Parameter ;
    rdfs:subPropertyOf n24:isClassifiedBy ;
    owl:inverseOf n24:isConstraintFor .

n24:hasEventDate a owl:DatatypeProperty ;
    rdfs:label "has event date"@en,
        "evento ha data"@it ;
    rdfs:comment "A datatype property that encodes values from xsd:dateTime for an Event; a same Event can have more than one xsd:dateTime value: begin date, end date, date at which the interval holds, etc." ;
    rdfs:domain n24:Event ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range xsd:dateTime ;
    rdfs:subPropertyOf n24:hasDataValue .

n24:hasIntervalDate a owl:DatatypeProperty ;
    rdfs:label "has interval date"@en,
        "intervallo ha data"@it ;
    rdfs:comment "A datatype property that encodes values from xsd:dateTime for a TimeInterval; a same TimeInterval can have more than one xsd:dateTime value: begin date, end date, date at which the interval holds, etc." ;
    rdfs:domain n24:TimeInterval ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range xsd:dateTime ;
    rdfs:subPropertyOf n24:hasRegionDataValue .

n24:hasParameter a owl:ObjectProperty ;
    rdfs:label "has parameter"@en,
        "ha parametro"@it ;
    rdfs:comment "A Concept can have a Parameter that constrains the attributes that a classified Entity can have in a certain Situation, e.g. a 4WheelDriver Role definedIn the ItalianTrafficLaw has a MinimumAge parameter on the Amount 16." ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Parameter ;
    rdfs:subPropertyOf n24:isRelatedToConcept ;
    owl:inverseOf n24:isParameterFor .

n24:hasParameterDataValue a owl:DatatypeProperty ;
    rdfs:label "has parameter data value"@en,
        "ha valore"@it ;
    rdfs:comment """Parametrizes values from a datatype. For example, a Parameter MinimumAgeForDriving hasParameterDataValue 18 on datatype xsd:int, in the Italian traffic code. In this example, MinimumAgeForDriving isDefinedIn the Norm ItalianTrafficCodeAgeDriving.
More complex parametrization requires workarounds. E.g. AgeRangeForDrugUsage could parametrize data value: 14 to 50 on the datatype: xsd:int. Since complex datatypes are not allowed in OWL1.0, a solution to this can only work by creating two 'sub-parameters': MinimumAgeForDrugUsage (that hasParameterDataValue 14) and MaximumAgeForDrugUsage (that hasParameterDataValue 50), which are components of (cf. hasComponent) the main Parameter AgeRangeForDrugUsage.
Ordering on subparameters can be created by using or specializing the object property 'precedes'.""" ;
    rdfs:domain n24:Parameter ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subPropertyOf n24:hasDataValue .

n24:hasPostcondition a owl:ObjectProperty ;
    rdfs:label "has postcondition"@en,
        "ha postcondizione"@it ;
    rdfs:comment """Direct succession applied to situations. 
E.g., 'A postcondition of our Plan is to have things settled'.""" ;
    rdfs:domain [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:subPropertyOf n24:directlyPrecedes ;
    owl:inverseOf n24:isPostconditionOf .

n24:hasPrecondition a owl:ObjectProperty ;
    rdfs:label "has precondition"@en,
        "ha precondizione"@it ;
    rdfs:comment """Direct precedence applied to situations. 
E.g., 'A precondition to declare war against a foreign country is claiming to find nuclear weapons in it'.""" ;
    rdfs:domain [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:subPropertyOf n24:directlyFollows ;
    owl:inverseOf n24:isPreconditionOf .

n24:hasQuality a owl:ObjectProperty ;
    rdfs:label "has quality"@en,
        "ha qualità"@it ;
    rdfs:comment "A relation between entities and qualities, e.g. 'Dmitri's skin is yellowish'." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Quality ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isQualityOf .

n24:hasRole a owl:ObjectProperty ;
    rdfs:label "has role"@en,
        "ha ruolo"@it ;
    rdfs:comment "A relation between an object and a role, e.g. the person 'John' has role 'student'." ;
    rdfs:domain n24:Object ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Role ;
    rdfs:subPropertyOf n24:isClassifiedBy ;
    owl:inverseOf n24:isRoleOf .

n24:hasTask a owl:ObjectProperty ;
    rdfs:label "has task"@en,
        "ha come obiettivo"@it ;
    rdfs:comment "A relation between roles and tasks, e.g. 'students have the duty of giving exams' (i.e. the Role 'student' hasTask the Task 'giving exams')." ;
    rdfs:domain n24:Role ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Task ;
    rdfs:subPropertyOf n24:isRelatedToConcept ;
    owl:inverseOf n24:isTaskOf .

n24:includesAction a owl:ObjectProperty ;
    rdfs:label "includes action"@en,
        "include azione"@it ;
    rdfs:comment "A relation between situations and actions, e.g. 'this morning I've prepared my coffee and had my fingers burnt' (i.e.: the preparation of my coffee this morning included a burning of my fingers)." ;
    rdfs:domain n24:Situation ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Action ;
    rdfs:subPropertyOf n24:includesEvent ;
    owl:inverseOf n24:isActionIncludedIn .

n24:includesAgent a owl:ObjectProperty ;
    rdfs:label "includes agent"@en,
        "include l'agente"@it ;
    rdfs:comment "A relation between situations and persons, e.g. 'this morning I've prepared my coffee and had my fingers burnt' (i.e.: the preparation of my coffee this morning included me)." ;
    rdfs:domain n24:Situation ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Agent ;
    rdfs:subPropertyOf n24:includesObject ;
    owl:inverseOf n24:isAgentIncludedIn .

n24:introduces a owl:ObjectProperty ;
    rdfs:label "introduces"@en,
        "introduce"@it ;
    rdfs:comment "A relation between a Description and a SocialAgent, e.g. a Constitutional Charter introduces the SocialAgent 'PresidentOfRepublic'." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:SocialAgent ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isIntroducedBy .

n24:involvesAgent a owl:ObjectProperty ;
    rdfs:label "involves agent"@en,
        "coinvolge agente"@it ;
    rdfs:comment "Agent participation." ;
    rdfs:domain n24:Event ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Agent ;
    rdfs:subPropertyOf n24:hasParticipant ;
    owl:inverseOf n24:isAgentInvolvedIn .

n24:isObservableAt a owl:ObjectProperty ;
    rdfs:label "is observable at"@en,
        "è osservabile a"@it ;
    rdfs:comment """A relation to represent a (past, present or future) TimeInterval at which an Entity is observable.
In order to encode a specific time, a data value should be related to the TimeInterval. 
An alternative way of representing time is the datatype property: hasIntervalDate""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:TimeInterval ;
    rdfs:subPropertyOf n24:hasRegion ;
    owl:inverseOf n24:isTimeOfObservationOf .

n24:isParametrizedBy a owl:ObjectProperty ;
    rdfs:label "is parametrized by"@en,
        "è parametrizzato da"@it ;
    rdfs:comment "The relation between a Parameter, e.g. 'MajorAge', and a Region, e.g. '>17 year'." ;
    rdfs:domain n24:Region ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Parameter ;
    rdfs:subPropertyOf n24:isClassifiedBy ;
    owl:inverseOf n24:parametrizes .

n24:isRealizedBy a owl:ObjectProperty ;
    rdfs:label "is realized by"@en,
        "è realizzato da"@it ;
    rdfs:comment "A relation between an information realization and an information object, e.g. the paper copy of the Italian Constitution realizes the text of the Constitution." ;
    rdfs:domain n24:InformationObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:InformationRealization ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:realizes .

n24:isReferenceOfInformationRealizedBy a owl:ObjectProperty ;
    rdfs:label "is reference of information realized by"@en,
        "è riferimento dell'informazione realizzata da"@it ;
    rdfs:comment "The relation between entities and information realizations, e.g. between Italy and a paper copy of the text of the Italian Constitution." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:InformationRealization ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:realizesInformationAbout .

n24:isSatisfiedBy a owl:ObjectProperty ;
    rdfs:label "is satisfied by"@en,
        "è soddisfatta da"@it ;
    rdfs:comment "A relation between a Situation and a Description, e.g. the execution of a Plan satisfies that plan." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Situation ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:satisfies .

n24:isSpecializedBy a owl:ObjectProperty,
        owl:TransitiveProperty ;
    rdfs:label "is specialized by"@en,
        "è specializzato da"@it ;
    rdfs:comment """A partial order relation that holds between social objects. It represents the subsumption relation between e.g. a Concept and another Concept that is broader in extensional interpretation, but narrowe in intensional interpretation.
E.g. PhDStudent Role specializes Student Role""" ;
    rdfs:domain n24:SocialObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:SocialObject ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:specializes .

n24:isSubordinatedTo a owl:ObjectProperty ;
    rdfs:label "is subordinated to"@en,
        "è subordinato a"@it ;
    rdfs:comment "Direct succession applied to concepts. E.g. the role 'Officer' is subordinated to 'Director'." ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:directlyFollows,
        n24:isRelatedToConcept ;
    owl:inverseOf n24:isSuperordinatedTo .

n24:isUnifiedBy a owl:ObjectProperty ;
    rdfs:label "is unified by"@en,
        "è unificato da"@it ;
    rdfs:comment "A Collection has a unification criterion, provided by a Description; for example, a community of practice can be unified by a shared theory or interest, e.g. the community that makes research on mirror neurons shares some core knowledge about mirror neurons, which can be represented as a Description MirrorNeuronTheory that unifies the community. There can be several unifying descriptions." ;
    rdfs:domain n24:Collection ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:unifies .

rdfs:comment a owl:AnnotationProperty .

rdfs:label a owl:AnnotationProperty .

owl:Thing a owl:Class .

owl:versionInfo a owl:AnnotationProperty .

ac:CuratedExample a ac:EvidenceKind,
        owl:NamedIndividual ;
    rdfs:label "Curated Example"@en .

ac:Disputed a ac:EpistemicStatus,
        owl:NamedIndividual ;
    rdfs:label "Disputed"@en .

ac:DocumentSource a ac:EvidenceKind,
        owl:NamedIndividual ;
    rdfs:label "Document Source"@en .

ac:GoalSpecification a owl:Class ;
    rdfs:label "Goal Specification"@en ;
    rdfs:comment "Desired outcome specification; does not imply an actual agent or achieved state." ;
    rdfs:subClassOf ac:ConditionSpecification .

ac:LLMGeneration a ac:EvidenceKind,
        owl:NamedIndividual ;
    rdfs:label "LLM Generation"@en .

ac:LogicalDerivation a ac:EvidenceKind,
        owl:NamedIndividual ;
    rdfs:label "Logical Derivation"@en .

ac:Observation a ac:EvidenceKind,
        owl:NamedIndividual ;
    rdfs:label "Observation"@en .

ac:PreconditionSpecification a owl:Class ;
    rdfs:label "Precondition Specification"@en ;
    rdfs:comment "Condition required before an action or phase." ;
    rdfs:subClassOf ac:ConditionSpecification .

ac:Proposed a ac:EpistemicStatus,
        owl:NamedIndividual ;
    rdfs:label "Proposed"@en .

ac:Rejected a ac:EpistemicStatus,
        owl:NamedIndividual ;
    rdfs:label "Rejected"@en .

ac:SimulationResult a ac:EvidenceKind,
        owl:NamedIndividual ;
    rdfs:label "Simulation Result"@en .

ac:Supported a ac:EpistemicStatus,
        owl:NamedIndividual ;
    rdfs:label "Supported"@en .

ac:Unknown a ac:EpistemicStatus,
        owl:NamedIndividual ;
    rdfs:label "Unknown"@en .

n2:EnablingCondition a owl:Class ;
    rdfs:label "Enabling Condition"@en ;
    rdfs:comment "Condition enabling a proposed mechanism; not automatically sufficient."@en ;
    rdfs:subClassOf ac:ConditionSpecification .

n3:Conjunction a owl:Class ;
    rdfs:label "Conjunction"@en ;
    rdfs:comment "Logical AND of two or more operands under strong Kleene three-valued logic."@en ;
    rdfs:subClassOf n3:BooleanExpression .

n3:Disjunction a owl:Class ;
    rdfs:label "Disjunction"@en ;
    rdfs:comment "Logical OR of two or more operands under strong Kleene three-valued logic."@en ;
    rdfs:subClassOf n3:BooleanExpression .

n3:Negation a owl:Class ;
    rdfs:label "Negation"@en ;
    rdfs:comment "Logical NOT of one operand; unknown remains unknown. Not negation-as-failure."@en ;
    rdfs:subClassOf n3:BooleanExpression .

n6:OpeningSpecification a owl:Class ;
    rdfs:label "Opening Specification"@en ;
    rdfs:comment "Opening geometry; dimensions remain unspecified until provided."@en ;
    rdfs:subClassOf n6:GeometrySpecification .

n11:ObservationEvidence a owl:Class ;
    rdfs:label "Observation Evidence"@en ;
    rdfs:comment "Externally supplied observation evidence with a timestamp, not an occurring event."@en ;
    rdfs:subClassOf ac:EvidenceRecord .

mt:Pouring a ac:ActionKind,
        owl:NamedIndividual ;
    rdfs:label "Pouring"@en .

mt:TransferFaultArgument1 a n16:ArgumentDefinition ;
    rdfs:label "source"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

mt:TransferFaultArgument2 a n16:ArgumentDefinition ;
    rdfs:label "destination"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

mt:TransferReadyArgument1 a n16:ArgumentDefinition ;
    rdfs:label "source"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

mt:TransferReadyArgument2 a n16:ArgumentDefinition ;
    rdfs:label "destination"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

mt:TransferReadyArgument3 a n16:ArgumentDefinition ;
    rdfs:label "substance"@en ;
    n16:argumentPosition "3"^^xsd:positiveInteger .

mt:evaluationContext a owl:ObjectProperty ;
    rdfs:label "Evaluation context"@en ;
    rdfs:domain mt:MaterialTransferDescription ;
    rdfs:range n11:KnowledgeContext .

mt:quantityBasis a owl:DatatypeProperty ;
    rdfs:label "Quantity basis"@en ;
    rdfs:domain mt:MaterialTransferDescription ;
    rdfs:range xsd:string .

n14:AtTargetArgument1 a n16:ArgumentDefinition ;
    rdfs:label "entity"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:AtTargetArgument2 a n16:ArgumentDefinition ;
    rdfs:label "target region"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:ClearOfSupportArgument1 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:ClearOfSupportArgument2 a n16:ArgumentDefinition ;
    rdfs:label "initial support"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:FixationDescription a owl:Class ;
    rdfs:label "Fixation description"@en ;
    rdfs:comment "Maintain holding and an object pose constraint over an evaluation interval."@en ;
    rdfs:subClassOf n14:ObjectManipulationDescription .

n14:GraspFailedArgument1 a n16:ArgumentDefinition ;
    rdfs:label "manipulator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:GraspFailedArgument2 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:HoldLostArgument1 a n16:ArgumentDefinition ;
    rdfs:label "manipulator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:HoldLostArgument2 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:HoldMaintainedArgument1 a n16:ArgumentDefinition ;
    rdfs:label "manipulator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:HoldMaintainedArgument2 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:HoldsArgument1 a n16:ArgumentDefinition ;
    rdfs:label "manipulator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:HoldsArgument2 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:ManipulatorClearArgument1 a n16:ArgumentDefinition ;
    rdfs:label "manipulator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:ManipulatorClearArgument2 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:PickupDescription a owl:Class ;
    rdfs:label "Pickup description"@en ;
    rdfs:comment "Acquire or retain a hold and lift clear of an initial support."@en,
        "Bounded supported pickup profile: pre-state graspability and initial support must reference the intended object. Success requires a hold, separation from initial support, and clearance; the declared conjunctive goal must follow from these success conditions. Current hold-loss evidence references the same manipulator/object. A hold alone is not pickup; missing support evidence is not separation."@en ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n14:initialSupport ],
        n14:ObjectManipulationDescription .

n14:PlacementDescription a owl:Class ;
    rdfs:label "Placement description"@en ;
    rdfs:comment "Bounded supported-place-and-clear profile: pre-state holding references the intended manipulator/object. Success requires stable destination support, target-region arrival, explicit release and manipulator clearance; the declared conjunctive goal must follow from these success conditions. Current support-instability evidence references this object/destination support. Release alone is not placement."@en,
        "Place at a target with stable support, release and manipulator clearance."@en ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n14:targetRegion ],
        [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n14:destinationSupport ],
        n14:ObjectManipulationDescription .

n14:PoseMaintainedArgument1 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:PoseMaintainedArgument2 a n16:ArgumentDefinition ;
    rdfs:label "reference frame"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:ReleaseBlockedArgument1 a n16:ArgumentDefinition ;
    rdfs:label "manipulator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:ReleaseBlockedArgument2 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:StableOnSupportArgument1 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:StableOnSupportArgument2 a n16:ArgumentDefinition ;
    rdfs:label "destination support"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:SupportUnstableArgument1 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:SupportUnstableArgument2 a n16:ArgumentDefinition ;
    rdfs:label "support"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:UnexpectedContactArgument1 a n16:ArgumentDefinition ;
    rdfs:label "manipulator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n14:UnexpectedContactArgument2 a n16:ArgumentDefinition ;
    rdfs:label "object"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n14:destinationSupport a owl:ObjectProperty ;
    rdfs:label "Destination support"@en ;
    rdfs:domain n14:PlacementDescription ;
    rdfs:range ac:EntitySpecification .

n14:evaluationContext a owl:ObjectProperty ;
    rdfs:label "Evaluation context"@en ;
    rdfs:domain n14:ObjectManipulationDescription ;
    rdfs:range n11:KnowledgeContext .

n14:initialSupport a owl:ObjectProperty ;
    rdfs:label "Initial support"@en ;
    rdfs:domain n14:PickupDescription ;
    rdfs:range ac:EntitySpecification .

n14:manipulator a owl:ObjectProperty ;
    rdfs:label "Manipulator"@en ;
    rdfs:domain n14:ObjectManipulationDescription ;
    rdfs:range ac:EntitySpecification .

n14:object a owl:ObjectProperty ;
    rdfs:label "Object"@en ;
    rdfs:domain n14:ObjectManipulationDescription ;
    rdfs:range ac:EntitySpecification .

n14:targetRegion a owl:ObjectProperty ;
    rdfs:label "Target region"@en ;
    rdfs:domain n14:ObjectManipulationDescription ;
    rdfs:range n14:TargetRegionSpecification .

n16:AccessibleArgument1 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "entity"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n16:ContainedInArgument1 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "content"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n16:ContainedInArgument2 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "container"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n16:ControlsArgument1 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "controller"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n16:ControlsArgument2 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "controlled entity"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n16:GraspableArgument1 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "entity"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n16:InContactWithArgument1 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "first entity"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n16:InContactWithArgument2 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "second entity"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n16:SupportsArgument1 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "support"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n16:SupportsArgument2 a n16:ArgumentDefinition,
        owl:NamedIndividual ;
    rdfs:label "supported entity"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n17:FailedTerminalPhase a owl:Class ;
    rdfs:label "Failed Terminal Phase"@en ;
    rdfs:comment "Terminal disposition marked failed by the model; reaching it is not asserted."@en ;
    rdfs:subClassOf n17:TerminalPhase .

n17:FailureTransition a owl:Class ;
    rdfs:label "Failure Transition"@en ;
    rdfs:comment "Progression to a failure disposition; may depend on explicit negative evidence."@en ;
    rdfs:subClassOf n17:PhaseTransition .

n17:OrdinaryTransition a owl:Class ;
    rdfs:label "Ordinary Transition"@en ;
    rdfs:comment "Normal progression between phases."@en ;
    rdfs:subClassOf n17:PhaseTransition .

n17:RecoveryTransition a owl:Class ;
    rdfs:label "Recovery Transition"@en ;
    rdfs:comment "Recovery or retry path. No retry budget is inferred."@en ;
    rdfs:subClassOf n17:PhaseTransition .

n17:SuccessfulTerminalPhase a owl:Class ;
    rdfs:label "Successful Terminal Phase"@en ;
    rdfs:comment "Terminal disposition marked successful by the model; reaching it is not asserted."@en ;
    rdfs:subClassOf n17:TerminalPhase .

n19:CartesianMotionDescription a owl:Class ;
    rdfs:label "Cartesian motion description"@en ;
    rdfs:comment "Motion toward a pose specification in an explicit reference frame."@en ;
    rdfs:subClassOf n19:RobotMotionDescription .

n19:ConfigurationMotionDescription a owl:Class ;
    rdfs:label "Configuration motion description"@en ;
    rdfs:comment "Motion toward a configuration of explicitly identified coordinates."@en ;
    rdfs:subClassOf n19:RobotMotionDescription .

n19:PathFollowingDescription a owl:Class ;
    rdfs:label "Path following description"@en ;
    rdfs:comment "Motion following ordered pose waypoints; endpoint arrival alone does not establish tracking."@en ;
    rdfs:subClassOf n19:RobotMotionDescription .

n21:EffectObservedArgument1 a n16:ArgumentDefinition ;
    rdfs:label "target"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n21:EffectObservedArgument2 a n16:ArgumentDefinition ;
    rdfs:label "intended effect"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n21:ToolAlignedArgument1 a n16:ArgumentDefinition ;
    rdfs:label "working part"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n21:ToolAlignedArgument2 a n16:ArgumentDefinition ;
    rdfs:label "target"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n21:ToolClearArgument1 a n16:ArgumentDefinition ;
    rdfs:label "tool"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n21:ToolClearArgument2 a n16:ArgumentDefinition ;
    rdfs:label "target"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n21:ToolFaultArgument1 a n16:ArgumentDefinition ;
    rdfs:label "operator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n21:ToolFaultArgument2 a n16:ArgumentDefinition ;
    rdfs:label "tool"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n21:ToolFunctionalArgument1 a n16:ArgumentDefinition ;
    rdfs:label "tool"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n21:ToolHeldArgument1 a n16:ArgumentDefinition ;
    rdfs:label "operator"@en ;
    n16:argumentPosition "1"^^xsd:positiveInteger .

n21:ToolHeldArgument2 a n16:ArgumentDefinition ;
    rdfs:label "tool"@en ;
    n16:argumentPosition "2"^^xsd:positiveInteger .

n21:ToolUseDescription a owl:Class ;
    rdfs:label "Tool use description"@en ;
    rdfs:comment "Maintain tool availability/control and establish an explicitly observed intended effect."@en ;
    rdfs:subClassOf n21:ToolInteractionDescription .

n22:hasTransferContract a owl:ObjectProperty ;
    rdfs:label "Has transfer contract"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range n22:TransferContract ;
    rdfs:subPropertyOf ac:hasComponent .

n24:BiologicalObject a owl:Class ;
    rdfs:label "Biological object"@en ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:PhysicalBody .

n24:Design a owl:Class ;
    rdfs:label "Design"@en,
        "Design"@it ;
    rdfs:comment """A Description of the Situation, in terms of structure and function, held by an Entity for some reason.
A design is usually accompanied by the rationales behind the construction of the designed Entity (i.e. of the reasons why a design is claimed to be as such). For example, the actual design (a Situation) of a car or of a law is based on both the specification (a Description) of the structure, and the rationales used to construct cars or laws.
While designs typically describe entities to be constructed, they can also be used to describe 'refunctionalized' entities, or to hypothesize unknown functions. For example, a cradle can be refunctionalized as a flowerpot based on a certain home design.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:DesignedArtifact a owl:Class ;
    rdfs:label "Designed artifact"@en,
        "Artefatto progettato"@it ;
    rdfs:comment "A PhysicalArtifact that is also described by a Design. This excludes simple recycling or refunctionalization of natural objects. Most common sense 'artifacts' can be included in this class: cars, lamps, houses, chips, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:isDescribedBy ;
            owl:someValuesFrom n24:Design ],
        n24:PhysicalArtifact .

n24:EventType a owl:Class ;
    rdfs:label "Event type"@en,
        "Tipo di evento"@it ;
    rdfs:comment "A Concept that classifies an Event . An event type describes how an Event should be interpreted, executed, expected, seen, etc., according to the Description that the EventType isDefinedIn (or used in)" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Event ;
            owl:onProperty n24:classifies ],
        n24:Concept ;
    owl:disjointWith n24:Parameter,
        n24:Role .

n24:FormalEntity a owl:Class ;
    rdfs:label "Formal entity"@en,
        "Entità formale astratta"@it ;
    rdfs:comment """Entities that are formally defined and are considered independent from the social context in which they are used. They cannot be localized in space or time. Also called 'Platonic entities'.
Mathematical and logical entities are included in this class: sets, categories, tuples, costants, variables, etc.
Abstract formal entities are distinguished from information objects, which are supposed to be part of a social context, and are localized in space and time, therefore being (social) objects.
For example, the class 'Quark' is an abstract formal entity from the purely set-theoretical perspective, but it is an InformationObject from the viewpoint of ontology design, when e.g. implemented in a logical language like OWL.
Abstract formal entities are also distinguished from Concept(s), Collection(s), and Description(s), which are part of a social context, therefore being SocialObject(s) as well.
For example, the class 'Quark' is an abstract FormalEntity from the purely set-theoretical perspective, but it is a Concept within history of science and cultural dynamics.

These distinctions allow to represent two different notions of 'semantics': the first one is abstract and formal ('formal semantics'), and formallyInterprets symbols that are about entities whatsoever; for example, the term 'Quark' isAbout the Collection of all quarks, and that Collection isFormalGroundingFor the abstract class 'Quark' (in the extensional sense). 
The second notion is social, localized in space-time ('social semantics'), and can be used to interpret entities in the intensional sense. For example, the Collection of all quarks isCoveredBy the Concept 'Quark', which is also expressed by the term 'Quark'.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Abstract .

n24:FunctionalSubstance a owl:Class ;
    rdfs:label "Functional substance"@en ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Substance .

n24:Goal a owl:Class ;
    rdfs:label "Goal"@en,
        "Scopo"@it ;
    rdfs:comment "The Description of a Situation that is desired by an Agent, and usually associated to a Plan that describes how to actually achieve it" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:PhysicalArtifact a owl:Class ;
    rdfs:label "Physical artifact"@en,
        "Artefatto fisico"@it ;
    rdfs:comment """Any PhysicalObject that isDescribedBy a Plan .
This axiomatization is weak, but allows to talk of artifacts in a very general sense, i.e. including recycled objects, objects with an intentional functional change, natural objects that are given a certain function, even though they are not modified or structurally designed, etc. PhysicalArtifact(s) are not considered disjoint from PhysicalBody(s), in order to allow a dual classification when needed. E.g.,
FunctionalSubstance(s) are included here as well.
Immaterial (non-physical) artifacts (e.g. texts, ideas, cultural movements, corporations, communities, etc. can be modelled as social objects (see SocialObject), which are all 'artifactual' in the weak sense assumed here.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:isDescribedBy ;
            owl:someValuesFrom n24:Plan ],
        n24:PhysicalObject .

n24:PhysicalAttribute a owl:Class ;
    rdfs:label "Physical attribute"@en,
        "Caratteristica fisica"@it ;
    rdfs:comment "Physical value of a physical object, e.g. density, color, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:PhysicalObject ;
            owl:onProperty n24:isRegionFor ],
        n24:Region ;
    owl:disjointWith n24:SpaceRegion,
        n24:TimeInterval .

n24:Process a owl:Class ;
    rdfs:label "Process"@en,
        "Processo"@it ;
    rdfs:comment """This is a placeholder for events that are considered in their evolution, or anyway not strictly dependent on agents, tasks, and plans. 
See Event class for some thoughts on classifying events. See also 'Transition'.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Event .

n24:Substance a owl:Class ;
    rdfs:label "Substance"@en,
        "Sostanza"@it ;
    rdfs:comment """Any PhysicalBody that has not necessarily specified (designed) boundaries, e.g. a pile of trash, some sand, etc. 
In this sense, an artistic object made of trash or a dose of medicine in the form of a pill would be a FunctionalSubstance, and a DesignedArtifact, since its boundaries are specified by a Design; aleatoric objects that are outcomes of an artistic process might be still considered DesignedArtifact(s), and Substance(s).""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:PhysicalBody .

n24:Workflow a owl:Class ;
    rdfs:label "Workflow"@en,
        "Workflow"@it ;
    rdfs:comment "A Plan that defines Role(s), Task(s), and a specific structure for tasks to be executed, usually supporting the work of an Organization" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:definesRole ;
            owl:someValuesFrom n24:Role ],
        [ a owl:Restriction ;
            owl:onProperty n24:definesTask ;
            owl:someValuesFrom n24:Task ],
        n24:Plan .

n24:coparticipatesWith a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "co-participates with"@en,
        "copartecipa con"@it ;
    rdfs:comment "A relation between two objects participating in a same Event; e.g., 'Vitas and Jimmy are playing tennis'." ;
    rdfs:domain n24:Object ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Object ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:coparticipatesWith .

n24:executesTask a owl:ObjectProperty ;
    rdfs:label "executes task"@en,
        "esegue il task"@it ;
    rdfs:comment "A relation between an action and a task, e.g. 'putting some water in a pot and putting the pot on a fire until the water starts bubbling' executes the task 'boiling'." ;
    rdfs:domain n24:Action ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Task ;
    rdfs:subPropertyOf n24:isClassifiedBy ;
    owl:inverseOf n24:isExecutedIn .

n24:farFrom a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "far from"@en ;
    rdfs:comment "Generic distance relation between any Entity(s). E.g. Rome is far from Beijing, astronomy is far from necromancy." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:farFrom .

n24:follows a owl:ObjectProperty,
        owl:TransitiveProperty ;
    rdfs:label "follows"@en,
        "segue"@it ;
    rdfs:comment """A relation between entities, expressing a 'sequence' schema. 
E.g. 'year 2000 follows 1999', 'preparing coffee' follows 'deciding what coffee to use', 'II World War follows I World War', etc. 
It can be used between tasks, processes or time intervals, and subproperties would fit best in order to distinguish the different uses.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:precedes .

n24:hasCommonBoundary a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "has common boundary"@en ;
    rdfs:comment "A relation to encode either formal or informal characterizations of 'boundaries' common to two different entities: an Event that ends when another begins, two abstract regions that have a common topological boundary, two objects that are said to be 'in contact' from a commonsense perspective, etc." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:hasCommonBoundary .

n24:hasLocation a owl:ObjectProperty ;
    rdfs:label "has location"@en,
        "ha localizzazione"@it ;
    rdfs:comment """A generic, relative spatial location, holding between any entities. E.g. 'the cat is on the mat', 'Omar is in Samarcanda', 'the wound is close to the femural artery'.
For 'absolute' locations, see SpaceRegion""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isLocationOf .

n24:hasMember a owl:ObjectProperty ;
    rdfs:label "has member"@en,
        "ha membro"@it ;
    rdfs:comment "A relation between collections and entities, e.g. 'my collection of saxophones includes an old Adolphe Sax original alto' (i.e. my collection has member an Adolphe Sax alto)." ;
    rdfs:domain n24:Collection ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isMemberOf .

n24:hasRegionDataValue a owl:DatatypeProperty ;
    rdfs:label "has region data value"@en,
        "regione ha valore"@it ;
    rdfs:comment "A datatype property that encodes values for a Region, e.g. a float for the Region Height." ;
    rdfs:domain n24:Region ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subPropertyOf n24:hasDataValue .

n24:hasTimeInterval a owl:ObjectProperty ;
    rdfs:label "has time interval"@en,
        "ha intervallo temporale"@it ;
    rdfs:comment "The generic relation between events and time intervals."@en ;
    rdfs:domain n24:Event ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:TimeInterval ;
    rdfs:subPropertyOf n24:hasRegion ;
    owl:inverseOf n24:isTimeIntervalOf .

n24:includesTime a owl:ObjectProperty ;
    rdfs:label "includes time"@en,
        "include tempo"@it ;
    rdfs:comment "A relation between situations and time intervals, e.g. 'this morning I've prepared my coffee and had my fingers burnt' (i.e.: preparing my coffee was held this morning). A data value attached to the time interval typically complements this modelling pattern." ;
    rdfs:domain n24:Situation ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:TimeInterval ;
    rdfs:subPropertyOf n24:isSettingFor ;
    owl:inverseOf n24:isTimeIncludedIn .

n24:isActionIncludedIn a owl:ObjectProperty ;
    rdfs:label "is action included in"@en,
        "è un'azione nel contesto di"@it ;
    rdfs:domain n24:Action ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Situation ;
    rdfs:subPropertyOf n24:isEventIncludedIn .

n24:isAgentIncludedIn a owl:ObjectProperty ;
    rdfs:label "is agent included in"@en,
        "è un agente nel contesto di"@it ;
    rdfs:domain n24:Agent ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Situation ;
    rdfs:subPropertyOf n24:isObjectIncludedIn .

n24:isAgentInvolvedIn a owl:ObjectProperty ;
    rdfs:label "is agent involved in"@en,
        "è un agente coinvolto in"@it ;
    rdfs:comment "Agent participation." ;
    rdfs:domain n24:Agent ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Event ;
    rdfs:subPropertyOf n24:isParticipantIn .

n24:isCharacterizedBy a owl:ObjectProperty ;
    rdfs:label "is characterized by {@en-us}",
        "is characterized by"@en,
        "è caratterizzato da"@it ;
    rdfs:domain n24:Collection ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isComponentOf a owl:ObjectProperty ;
    rdfs:label "is component of"@en,
        "è componente di"@it ;
    rdfs:comment "The hasPart relation without transitivity, holding between an Object (the system) and another (the component), and assuming a Design that structures the Object." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:isPartOf .

n24:isConceptExpressedBy a owl:ObjectProperty ;
    rdfs:label "is concept expressed by"@en,
        "è un concetto espresso da"@it ;
    rdfs:comment "A relation between an InformationObject and a Concept , e.g. the term \"dog\" expresses the Concept \"dog\". For expressing a relational meaning, see the more general object property: expresses" ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:InformationObject ;
    rdfs:subPropertyOf n24:isExpressedBy .

n24:isConceptUsedIn a owl:ObjectProperty ;
    rdfs:label "is concept used in"@en,
        "è un concetto usato in"@it ;
    rdfs:comment "A more generic relation holding between a Description and a Concept. In order to be used, a Concept must be previously definedIn another Description" ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:usesConcept .

n24:isConceptualizedBy a owl:ObjectProperty ;
    rdfs:label "is conceptualized by"@en,
        "è concettualizzato da"@it ;
    rdfs:comment "A relation stating that an Agent is internally representing a Description . E.g., 'John believes in the conspiracy theory'; 'Niels Bohr created a solar-system metaphor for his atomic theory'; 'Jacques assumes all swans are white'; 'the task force shares the attack plan'." ;
    rdfs:domain n24:SocialObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Agent ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isConcretelyExpressedBy a owl:ObjectProperty ;
    rdfs:label "is concretely expressed by"@en,
        "è espresso concretamente da"@it ;
    rdfs:comment "A relation between an InformationRealization and a Description, e.g. 'the printout of the Italian Constitution concretelyExpresses the Italian Constitution'. It should be supplied also with a rule stating that the InformationRealization realizes an InformationObject that expresses the Description" ;
    rdfs:domain n24:SocialObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:InformationRealization ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isConstituentOf a owl:ObjectProperty ;
    rdfs:label "is constituent of"@en,
        "è costituente di"@it ;
    rdfs:comment """'Constituency' depends on some layering of  the world described by the ontology. For example, scientific granularities (e.g. body-organ-tissue-cell) or ontological 'strata' (e.g. social-mental-biological-physical) are  typical layerings. 
Intuitively, a constituent is a part belonging to a lower layer. Since layering is actually a partition of the world described by the ontology, constituents are not properly classified as parts, although this kinship can be intuitive for common sense.
A desirable advantage of this distinction is that we are able to talk e.g. of physical constituents of non-physical objects (e.g. systems), while this is not possible in terms of parts.
Example of are the persons constituting a social system, the molecules constituting a person, the atoms constituting a river, etc. 
In all these examples, we notice a typical discontinuity between the constituted and the constituent object: e.g. a social system is conceptualized at a different layer from the persons that constitute it, a person is conceptualized at a different layer from the molecules that constitute them, and a river is conceptualized at a different layer from the atoms that constitute it.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isConstraintFor a owl:ObjectProperty ;
    rdfs:label "is constraint for"@en,
        "è un vincolo per"@it ;
    rdfs:comment """A relation between parameters and entities. It allows to assert generic constraints (encoded as parameters), e.g. MinimumAgeForDriving isConstraintFor John (where John is a legal subject under the TrafficLaw).
The intended semantics (not expressible in OWL) is that a Parameter isConstraintFor and Entity if the Parameter isParameterFor a Concept that classifies that Entity; moreover, it entails that a Parameter parametrizes a Region that isRegionFor that Entity. The use in OWL is therefore a shortcut to annotate what Parameter constrains what Entity""" ;
    rdfs:domain n24:Parameter ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:classifies .

n24:isCoveredBy a owl:ObjectProperty ;
    rdfs:label "is covered by"@en,
        "è ricoperto da"@it ;
    rdfs:comment """A relation between concepts and collections, where a Concept is said to cover a Collection; it corresponds to a link between the (reified) intensional and extensional interpretations of a (reified) class.
E.g. the collection of vintage saxophones is covered by the Concept 'Saxophone' with the Parameter 'Vintage'.""" ;
    rdfs:domain n24:Collection ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isExpandedIn a owl:ObjectProperty ;
    rdfs:label "is expanded in"@en,
        "è espansa in"@it ;
    rdfs:comment """A partial order relation that holds between descriptions. It represents the proper part relation between a description and another description featuring the same properties as the former, with at least one additional one.
Descriptions can be expanded either by adding other descriptions as parts, or by refining concepts that are used by them. 
An 'intention' to expand must be present (unless purely formal theories are considered, but even in this case a criterion of relevance is usually active).""" ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:isRelatedToDescription .

n24:isMemberOf a owl:ObjectProperty ;
    rdfs:label "is member of"@en,
        "è membro di"@it ;
    rdfs:comment "A relation between collections and entities, e.g. 'the Night Watch by Rembrandt is in the Rijksmuseum collection'; 'Davide is member of the Pen Club', 'Igor is one the subjects chosen for the experiment'." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Collection ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isParameterFor a owl:ObjectProperty ;
    rdfs:label "is parameter for"@en,
        "è un parametro per"@it ;
    rdfs:comment "A Concept can have a Parameter that constrains the attributes that a classified Entity can have in a certain Situation, e.g. a 4WheelDriver Role definedIn the ItalianTrafficLaw has a MinimumAge parameter on the Amount 16." ;
    rdfs:domain n24:Parameter ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:isRelatedToConcept .

n24:isPostconditionOf a owl:ObjectProperty ;
    rdfs:label "is postcondition of"@en,
        "è postcondizione di"@it ;
    rdfs:comment """Direct succession applied to situations. 
E.g., 'Taking some rest is a postcondition of my search for a hotel'.""" ;
    rdfs:domain [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:subPropertyOf n24:directlyFollows .

n24:isPreconditionOf a owl:ObjectProperty ;
    rdfs:label "is precondition of"@en,
        "è precondizione di"@it ;
    rdfs:comment """Direct precedence applied to situations. 
E.g., 'claiming to find nuclear weapons in a foreign country is a precondition to declare war against it'.""" ;
    rdfs:domain [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range [ a owl:Class ;
            owl:unionOf ( n24:Event n24:Situation ) ] ;
    rdfs:subPropertyOf n24:directlyPrecedes .

n24:isQualityOf a owl:ObjectProperty ;
    rdfs:label "is quality of"@en,
        "è una qualità di"@it ;
    rdfs:comment "A relation between entities and qualities, e.g. 'Dmitri's skin is yellowish'." ;
    rdfs:domain n24:Quality ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isReferenceOf a owl:ObjectProperty ;
    rdfs:label "is reference of"@en,
        "è il riferimento di"@it ;
    rdfs:comment """A relation between information objects and any Entity (including information objects). It can be used to talk about e.g. entities are references of proper nouns: the proper noun 'Leonardo da Vinci' isAbout the Person Leonardo da Vinci; as well as to talk about sets of entities that can be described by a common noun: the common noun 'person' isAbout the set of all persons in a domain of discourse, which can be represented in DOLCE-Ultralite as an individual of the class: Collection .
The isReferenceOf relation is irreflexive, differently from its inverse isAbout.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:InformationObject ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isRoleDefinedIn a owl:ObjectProperty ;
    rdfs:label "is role defined in"@en,
        "è un ruolo definito in"@it ;
    rdfs:comment "A relation between a description and a role, e.g. the role 'Ingredient' is defined in the recipe for a cake." ;
    rdfs:domain n24:Role ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:isDefinedIn .

n24:isRoleOf a owl:ObjectProperty ;
    rdfs:label "is role of"@en,
        "è un ruolo di"@it ;
    rdfs:comment "A relation between an object and a role, e.g. 'student' is the role of 'John'." ;
    rdfs:domain n24:Role ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Object ;
    rdfs:subPropertyOf n24:classifies .

n24:isSuperordinatedTo a owl:ObjectProperty ;
    rdfs:label "is superordinated to"@en,
        "è superordinato a"@it ;
    rdfs:comment "Direct precedence applied to concepts. E.g. the role 'Executive' is superordinated to 'DepartmentManager'." ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:directlyPrecedes,
        n24:isRelatedToConcept .

n24:isTimeIncludedIn a owl:ObjectProperty ;
    rdfs:label "is time included in"@en,
        "è un tempo nel contesto di"@it ;
    rdfs:domain n24:TimeInterval ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Situation ;
    rdfs:subPropertyOf n24:hasSetting .

n24:isTimeIntervalOf a owl:ObjectProperty ;
    rdfs:label "is time interval of"@en,
        "intervallo temporale di"@it ;
    rdfs:comment "The generic relation between time intervals and events."@en ;
    rdfs:domain n24:TimeInterval ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Event ;
    rdfs:subPropertyOf n24:isRegionFor .

n24:isTimeOfObservationOf a owl:ObjectProperty ;
    rdfs:label "is time of observation of"@en,
        "è il tempo di osservazione di"@it ;
    rdfs:comment """A relation to represent a (past, present or future) TimeInterval at which an Entity is observable.
In order to encode a specific time, a data value should be related to the TimeInterval. 
An alternative way of representing time is the datatype property: hasIntervalDate""" ;
    rdfs:domain n24:TimeInterval ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:isRegionFor .

n24:nearTo a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "near to"@en ;
    rdfs:comment "Generic distance relation between any Entity(s). E.g. Rome is near to Florence, astronomy is near to physics." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:nearTo .

n24:realizesInformationAbout a owl:ObjectProperty ;
    rdfs:label "realizes information about"@en,
        "realizza informazione che si riferisce a a"@it ;
    rdfs:comment "The relation between entities and information realizations, e.g. between Italy and a paper copy of the text of the Italian Constitution." ;
    rdfs:domain n24:InformationRealization ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

n24:sameSettingAs a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "is in the same setting as"@en,
        "è nella stessa situazione di"@it ;
    rdfs:comment "A relation between two entities participating in a same Situation; e.g., 'Our company provides an antivenom service' (the situation is the service, the two entities are the company and the antivenom)."@it ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl" ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:sameSettingAs .

n24:specializes a owl:ObjectProperty,
        owl:TransitiveProperty ;
    rdfs:label "specializes"@en,
        "specializza"@it ;
    rdfs:comment """A partial order relation that holds between social objects. 
It mainly represents the subsumption relation between e.g. a Concept or Description and another Concept (resp. Description) that is broader in extensional interpretation, but narrower in intensional interpretation. For example, the role PhDStudent specializes the role Student.
Another possible use is between a Collection that isCoveredBy a Concept A, and another Collection that isCoveredBy a Concept B that on its turm specializes A. For example, the 70,000 series Selmer Mark VI saxophone Collection specializes the Selmer Mark VI saxophone Collection.""" ;
    rdfs:domain n24:SocialObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:SocialObject ;
    rdfs:subPropertyOf n24:associatedWith .

n24:unifies a owl:ObjectProperty ;
    rdfs:label "unifies"@en,
        "unifica"@it ;
    rdfs:comment "A Collection has a unification criterion, provided by a Description; for example, a community of practice can be unified by a shared theory or interest, e.g. the community that makes research on mirror neurons shares some core knowledge about mirror neurons, which can be represented as a Description MirrorNeuronTheory that unifies the community. There can be several unifying descriptions." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Collection ;
    rdfs:subPropertyOf n24:associatedWith .

ac:CausalSpecification a owl:Class ;
    rdfs:label "Causal Specification"@en ;
    rdfs:comment "Proposed explanatory or causal model, whose support is recorded in claims." ;
    rdfs:subClassOf ac:ActionComponent .

ac:ControlSpecification a owl:Class ;
    rdfs:label "Control Specification"@en ;
    rdfs:comment "Optional abstract control specification without mandatory robot joints or actuators." ;
    rdfs:subClassOf ac:ActionComponent .

ac:FailureCriterion a owl:Class ;
    rdfs:label "Failure Criterion"@en ;
    rdfs:comment "Criterion used to recognize failure." ;
    rdfs:subClassOf ac:ConditionSpecification .

ac:PhysicalModelSpecification a owl:Class ;
    rdfs:label "Physical Model Specification"@en ;
    rdfs:comment "Description of a physical model with assumptions and applicability; not physics validation." ;
    rdfs:subClassOf ac:ActionComponent .

ac:RecoverySpecification a owl:Class ;
    rdfs:label "Recovery Specification"@en ;
    rdfs:comment "Description of a response to a failure condition." ;
    rdfs:subClassOf ac:ActionComponent .

ac:SuccessCriterion a owl:Class ;
    rdfs:label "Success Criterion"@en ;
    rdfs:comment "Criterion used to judge success; not a report of success." ;
    rdfs:subClassOf ac:ConditionSpecification .

n1:CapabilityProfile a owl:Class ;
    rdfs:label "Capability Profile"@en ;
    rdfs:comment "Grouped capability requirements; no robot capability is asserted."@en ;
    rdfs:subClassOf ac:ActionComponent .

n2:CausalModel a owl:Class ;
    rdfs:label "Causal Model"@en ;
    rdfs:comment "Description of proposed causal relationships, not an identified causal system."@en ;
    rdfs:subClassOf ac:CausalSpecification .

n2:InterventionSpecification a owl:Class ;
    rdfs:label "Intervention Specification"@en ;
    rdfs:comment "Proposed intervention described by an intended outcome. Does not establish an interventional effect."@en ;
    rdfs:subClassOf ac:ActionComponent .

n3:AtomicCondition a owl:Class ;
    rdfs:label "Atomic Condition"@en ;
    rdfs:comment "Truth of a relation pattern in an explicit state scope."@en ;
    rdfs:subClassOf n3:BooleanExpression .

n3:ParameterReference a owl:Class ;
    rdfs:label "Parameter Reference"@en ;
    rdfs:comment "Reference to an unspecified or externally supplied parameter in a particular state."@en ;
    rdfs:subClassOf n3:ValueExpression .

n9:FrameProjectionArtifact a owl:Class ;
    rdfs:label "Frame Projection Artifact"@en ;
    rdfs:comment "Derived frame output, not a world event or executable designator."@en ;
    rdfs:subClassOf ac:ProjectionArtifact .

n9:SlotMapping a owl:Class ;
    rdfs:label "Slot Mapping"@en ;
    rdfs:comment "Finite forward predicate path from the source action description."@en ;
    rdfs:subClassOf n24:Description .

n10:ContactSchema a owl:Class ;
    rdfs:label "Contact Schema"@en ;
    rdfs:comment "Contact pattern without inferred force."@en ;
    rdfs:subClassOf n10:ImageSchemaDescription .

n10:ContainmentSchema a owl:Class ;
    rdfs:label "Containment Schema"@en ;
    rdfs:comment "Content/container pattern."@en ;
    rdfs:subClassOf n10:ImageSchemaDescription .

n10:SupportSchema a owl:Class ;
    rdfs:label "Support Schema"@en ;
    rdfs:comment "Support/supportee pattern without inferred physical stability."@en ;
    rdfs:subClassOf n10:ImageSchemaDescription .

n11:RecordSnapshot a owl:Class ;
    rdfs:label "Record Snapshot"@en ;
    rdfs:comment "Digest of exact record dataset bytes under a named serialization profile; storage enforcement is external."@en ;
    rdfs:subClassOf n24:InformationObject .

mt:PouringDescription a owl:Class ;
    rdfs:label "Pouring description"@en ;
    rdfs:comment "Bounded partial-pouring profile: preconditions use scoped TransferReady for the intended endpoints/material, or explicit pre-state source containment and destination accessibility. Success binds intended portion/destination and delivered-target/spill-tolerance comparisons. A weaker conjunctive goal is permitted when success establishes it. Failure identifies these endpoints through a current TransferFault, or uses explicit post-state spill exceedance. Container tilting or a completed motion path alone is not transfer success. Quantities may remain unspecified; no physical values or validated flow model are implied."@en,
        "Material-transfer specialization; physical flow laws and tool requirements are supplied separately."@en ;
    rdfs:subClassOf mt:MaterialTransferDescription .

n13:QuantityObservation a owl:Class ;
    rdfs:label "Quantity Observation"@en ;
    rdfs:comment "Observation of a parameter in a scope; no numeric value or measurement device is implied."@en ;
    rdfs:subClassOf n13:ObservationRequirement .

n13:RelationObservation a owl:Class ;
    rdfs:label "Relation Observation"@en ;
    rdfs:comment "Observation of a relation pattern in a scope; an abstract requirement, not an observed fact."@en ;
    rdfs:subClassOf n13:ObservationRequirement .

n15:NormalizationSpecification a owl:Class ;
    rdfs:label "Normalization Specification"@en ;
    rdfs:comment "Declared scale for rendering a metric dimensionless; unknown scales remain unresolved."@en ;
    rdfs:subClassOf ac:ActionComponent .

n17:TerminalPhase a owl:Class ;
    rdfs:label "Terminal Phase"@en ;
    rdfs:comment "A phase with no outgoing transitions within its phase structure."@en ;
    rdfs:subClassOf ac:PhaseSpecification .

n18:SimulationStudySpecification a owl:Class ;
    rdfs:label "Simulation Study Specification"@en ;
    rdfs:comment "Description of a proposed simulation study, separate from evidence of an actual run."@en ;
    rdfs:subClassOf ac:ActionComponent .

n19:ConfigurationTarget a owl:Class ;
    rdfs:label "Configuration target"@en ;
    rdfs:comment "Descriptive configuration entries, without actual robot joint references."@en ;
    rdfs:subClassOf ac:ActionComponent .

n19:PathSpecification a owl:Class ;
    rdfs:label "Path specification"@en ;
    rdfs:comment "Ordered waypoint specifications without a timing or interpolation default."@en ;
    rdfs:subClassOf ac:ActionComponent .

n20:MismatchSpecification a owl:Class ;
    rdfs:label "Mismatch Specification"@en ;
    rdfs:comment "Proposed criterion for discrepancy between expected and supplied evidence."@en ;
    rdfs:subClassOf ac:ActionComponent .

n24:Abstract a owl:Class ;
    rdfs:label "Abstract"@en,
        "Astratto"@it ;
    rdfs:comment "Any Entity that cannot be located in space-time. E.g. mathematical entities: formal semantics elements, regions within dimensional spaces, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Entity ;
    owl:disjointWith n24:Event,
        n24:Object,
        n24:Quality .

n24:CollectiveAgent a owl:Class ;
    rdfs:label "Collective agent"@en,
        "Agente collettivo"@it ;
    rdfs:comment """A SocialAgent that is actedBy agents that are (and act as) members of a Collective. A collective agent can have roles that are also roles of those agents.
For example, in sociology, a 'group action' is the situation in which a number of people (that result to be members of a collective) in a given area behave in a coordinated way in order to achieve a (often common) goal. The Agent in such a Situation is not single, but a CollectiveAgent (a Group). This can be generalized to the notion of social movement, which assumes a large Community or even the entire Society as agents.
The difference between a CollectiveAgent and an Organization is that a Description that introduces a CollectiveAgent is also one that unifies the corresponding Collective. In practice, this difference makes collective agents 'less stable' than organizations, because they have a dedicated, publicly recognizable Description that is conceived to introduce them.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:isIntroducedBy ;
            owl:someValuesFrom n24:Description ],
        [ a owl:Restriction ;
            owl:onProperty n24:actsThrough ;
            owl:someValuesFrom n24:Agent ],
        n24:SocialAgent .

n24:InformationEntity a owl:Class ;
    rdfs:comment "A piece of information, be it concretely realized or not." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Entity .

n24:Person a owl:Class ;
    rdfs:label "Persona {it}",
        "Person"@en ;
    rdfs:comment "Persons in commonsense intuition, which does not apparently distinguish between either natural or social persons." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Agent .

n24:expresses a owl:ObjectProperty ;
    rdfs:label "expresses"@en,
        "esprime"@it ;
    rdfs:comment """A relation between an InformationObject and a 'meaning', generalized here as a 'SocialObject'. For example: 'A Beehive is a structure in which bees are kept, typically in the form of a dome or box.' (Oxford dictionary)'; 'the term Beehive expresses the concept Beehive in my apiculture ontology'.
The intuition for 'meaning' is intended to be very broad. A separate, large comment is included for those who want to investigate more on what kind of meaning can be represented in what form.""",
        """This is a large comment field for those who want to investigate the different uses of the 'expresses' relation for modeling different approaches to meaning characterization and modeling.
For example, in all these cases, some aspect of meaning is involved:

- Beehive means "a structure in which bees are kept, typically in the form of a dome or box." (Oxford dictionary)
- 'Beehive' is a synonym in noun synset 09218159 "beehive|hive" (WordNet)
- 'the term Beehive can be interpreted as the fact of 'being a beehive', i.e. a relation that holds for concepts such as Bee, Honey, Hosting, etc.'
- 'the text of Italian apiculture regulation expresses a rule by which beehives should be kept at least one kilometer away from inhabited areas'
- 'the term Beehive expresses the concept Beehive'
- ''Beehive' for apiculturists does not express the same meaning as for, say, fishermen'
- 'Your meaning of 'Beautiful' does not seem to fit mine'
- ''Beehive' is formally interpreted as the set of all beehives'
- 'from the term 'Beehive', we can build a vector space of statistically significant cooccurring terms in the documents that contain it'
- the lexeme 'Belly' expresses the role 'Body_Part' in the frame 'ObservableBodyParts' (FrameNet)

As the examples suggest, the 'meaning of meaning' is dependent on the background approach/theory that one assumes. One can hardly make a summary of the too many approaches and theories of meaning, therefore this relation is maybe the most controversial and difficult to explain; normally, in such cases it would be better to give up formalizing. 
However, the usefulness of having a 'semantic abstraction' in modeling information objects is so high (e.g. for the semantic web, interoperability, reengineering, etc.), that we accept this challenging task, although without taking any particular position in the debate. 
We provide here some examples, which we want to generalize upon when using the 'expresses' relation to model semantic aspects of social reality.

In the most common approach, lexicographers that write dictionaries, glossaries, etc. assume that the meaning of a term is a paraphrase (or 'gloss', or 'definition'). 
Another approach is provided by concept schemes like thesauri and lexicons, which assume that the meaning of a term is a 'concept', encoded as a 'lemma', 'synset', or 'descriptor'.
Still another approach is that of psychologists and cognitive scientists, which often assume that the meaning of an information object is a concept encoded in the mind or cognitive system of an agent. 
A radically different approach is taken by social scientists and semioticians, who usually assume that meanings of an information object are spread across the communication practices in which members of a community use that object.
Another approach that tackles the distributed nature of meaning is assumed by geometrical models of semantics, which assume that the meaning of an InformationObject (e.g. a word) results from the set of informational contexts (e.g. within texts) in which that object is used similarly.
The logical approach to meaning is still different, since it assumes that the meaning of e.g. a term is equivalent to the set of individuals that the term can be applied to; for example, the meaning of 'Ali' is e.g. an individual person called Ali, the meaning of 'Airplane' is e.g. the set of airplanes, etc. 
Finally, an approach taken by structuralist linguistics and frame semantics is that a meaning is the relational context in which an information object can be applied; for example, a meaning of 'Airplane' is situated e.g. in the context ('frame') of passenger airline flights.

These different approaches are not necessarily conflicting, and they mostly talk about different aspects of so-called 'semantics'. They can be summarized and modelled within DOLCE-Ultralite as follows (notice that such list is far from exhaustive):

(1) Informal meaning (as for linguistic or commonsense semantics: a distinction is assumed between (informal) meaning and reference; see isAbout for an alternative pattern on reference)
	- Paraphrase meaning (as for lexicographic semantics). Here it is modelled as the expresses relation between instances of InformationObject and different instances of InformationObject that act as 'paraphrases'
	- Conceptual meaning (as for 'concept scheme' semantics). Here it is modelled as the expresses relation between instances of InformationObject and instances of Concept
	- Relational meaning (as for frame semantics). Here it is modelled as the expresses relation between instances of InformationObject and instances of Description
	- Cognitive meaning (as for 'psychological' semantics). Here it is modelled as the expresses relation between any instance of InformationObject and any different instance of InformationObject that isRealizedBy a mental, cognitive or neural state (depending on which theory of mind is assumed). Such states can be considered here as instances of Process (occurring in the mind, cognitive system, or neural system of an agent)
	- Cultural meaning (as for 'social science' semantics). Here it is modelled as the expresses relation between instances of InformationObject and instances of SocialObject (institutions, cultural paradigms, norms, social practices, etc.)
	- Distributional meaning (as for geometrical models of meaning). Here it is modelled as the expresses relation between any instance of InformationObject and any different instance of InformationObject that isFormallyRepresentedIn some (geometrical) Region (e.g. a vector space)

(2) Formal meaning (as for logic and formal semantics: no distinction is assumed between informal meaning and reference, therefore between 'expresses' and 'isAbout', which can be used interchangeably)
	- Object-level formal meaning (as in the traditional first-order logic semantics). Here it is modelled as the expresses relation between an instance of InformationObject and an instance of Collection that isGroundingFor (in most cases) a Set; isGroundingFor is defined in the ontology: http://www.ontologydesignpatterns.org/ont/dul/IOLite.owl
	- Modal formal meaning (as in possible-world semantics). Here it is modelled as the expresses relation between an instance of InformationObject and an instance of Collection that isGroundingFor a Set, and which isPartOf some different instance of Collection that isGroundingFor a PossibleWorld

This is only a first step to provide a framework, in which one can model different aspects of meaning. A more developed ontology should approach the problem of integrating the different uses of 'expresses', so that different theories, resources, methods can interoperate.""" ;
    rdfs:domain n24:InformationObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:SocialObject ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isExpressedBy .

n24:includesEvent a owl:ObjectProperty ;
    rdfs:label "includes event"@en,
        "include l'evento"@it ;
    rdfs:comment "A relation between situations and events, e.g. 'this morning I've prepared my coffee and had my fingers burnt' (i.e.: the preparation of my coffee this morning included a burning of my fingers)." ;
    rdfs:domain n24:Situation ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Event ;
    rdfs:subPropertyOf n24:isSettingFor ;
    owl:inverseOf n24:isEventIncludedIn .

n24:includesObject a owl:ObjectProperty ;
    rdfs:label "includes object"@en,
        "include l'oggetto"@it ;
    rdfs:comment "A relation between situations and objects, e.g. 'this morning I've prepared my coffee and had my fingers burnt' (i.e.: the preparation of my coffee this morning included me)." ;
    rdfs:domain n24:Situation ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Object ;
    rdfs:subPropertyOf n24:isSettingFor ;
    owl:inverseOf n24:isObjectIncludedIn .

n24:isAbout a owl:ObjectProperty ;
    rdfs:label "is about"@en,
        "si riferisce a"@it ;
    rdfs:comment """A relation between information objects and any Entity (including information objects). It can be used to talk about e.g. entities are references of proper nouns: the proper noun 'Leonardo da Vinci' isAbout the Person Leonardo da Vinci; as well as to talk about sets of entities that can be described by a common noun: the common noun 'person' isAbout the set of all persons in a domain of discourse, which can be represented in DOLCE-Ultralite as an individual of the class: Collection .
The isAbout relation is reflexive (not expressible in OWL1.0), because information objects are also about themselves.""" ;
    rdfs:domain n24:InformationObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isReferenceOf .

n24:isEventIncludedIn a owl:ObjectProperty ;
    rdfs:label "is event included in"@en,
        "è un evento nel contesto di"@it ;
    rdfs:domain n24:Event ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Situation ;
    rdfs:subPropertyOf n24:hasSetting .

n24:isExecutedIn a owl:ObjectProperty ;
    rdfs:label "is executed in"@en,
        "è eseguito mediante"@it ;
    rdfs:comment "A relation between an action and a task, e.g. 'putting some water in a pot and putting the pot on a fire until the water starts bubbling' executes the task 'boiling'." ;
    rdfs:domain n24:Task ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Action ;
    rdfs:subPropertyOf n24:classifies .

n24:isIntroducedBy a owl:ObjectProperty ;
    rdfs:label "is introduced by"@en,
        "è introdotto da"@it ;
    rdfs:comment "A relation between a Description and a SocialAgent, e.g. a Constitutional Charter introduces the SocialAgent 'PresidentOfRepublic'." ;
    rdfs:domain n24:SocialAgent ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isLocationOf a owl:ObjectProperty ;
    rdfs:label "is location of"@en,
        "è una localizzazione di"@it ;
    rdfs:comment """A generic, relative localization, holding between any entities. E.g. 'Rome is the seat of the Pope', 'the liver is the location of the tumor'.
For 'absolute' locations, see SpaceRegion""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isObjectIncludedIn a owl:ObjectProperty ;
    rdfs:label "is object included in"@en,
        "è un oggetto nel contesto di"@it ;
    rdfs:domain n24:Object ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Situation ;
    rdfs:subPropertyOf n24:hasSetting .

n24:isPartOf a owl:ObjectProperty,
        owl:TransitiveProperty ;
    rdfs:label "is part of"@en,
        "è parte di"@it ;
    rdfs:comment "A relation between any entities, e.g.'brain is a part of the human body'." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isTaskDefinedIn a owl:ObjectProperty ;
    rdfs:label "is task defined in"@en,
        "è un task definito in"@it ;
    rdfs:comment "A relation between a description and a task, e.g. the task 'boil' is defined in a recipe for a cake." ;
    rdfs:domain n24:Task ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:isDefinedIn .

n24:isTaskOf a owl:ObjectProperty ;
    rdfs:label "is task of"@en,
        "è un obiettivo per"@it ;
    rdfs:comment "A relation between roles and tasks, e.g. 'students have the duty of giving exams' (i.e. the Role 'student' hasTask the Task 'giving exams')." ;
    rdfs:domain n24:Task ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Role ;
    rdfs:subPropertyOf n24:isRelatedToConcept .

n24:overlaps a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "overlaps"@en,
        "sovrapposto a"@it ;
    rdfs:comment """A schematic relation between any entities, e.g. 'the chest region overlaps with the abdomen region', 'my spoken words overlap with hers', 'the time of my leave overlaps with the time of your arrival', 'fibromyalgia overlaps with other conditions'.
Subproperties and restrictions can be used to specialize overlaps for objects, events, time intervals, etc.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:overlaps .

n24:parametrizes a owl:ObjectProperty ;
    rdfs:label "parametrizes"@en,
        "parametrizza"@it ;
    rdfs:comment """The relation between a Parameter, e.g. 'MajorAgeLimit', and a Region, e.g. '18_year'.
For a more data-oriented relation, see hasDataValue""" ;
    rdfs:domain n24:Parameter ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Region ;
    rdfs:subPropertyOf n24:classifies .

n24:realizes a owl:ObjectProperty ;
    rdfs:label "realizes"@en,
        "realizza"@it ;
    rdfs:comment "A relation between an information realization and an information object, e.g. the paper copy of the Italian Constitution realizes the text of the Constitution." ;
    rdfs:domain n24:InformationRealization ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:InformationObject ;
    rdfs:subPropertyOf n24:associatedWith .

n24:usesConcept a owl:ObjectProperty ;
    rdfs:label "uses concept"@en,
        "usa il concetto"@it ;
    rdfs:comment "A generic relation holding between a Description and a Concept. In order to be used, a Concept must be previously definedIn another Description. This last condition cannot be encoded for object properties in OWL." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:associatedWith .

ac:RoleSpecification a owl:Class ;
    rdfs:label "Role Specification"@en ;
    rdfs:comment "Associates a conceptual participant role with an entity specification in this description." ;
    rdfs:subClassOf ac:ActionComponent .

n1:ExecutionStrategy a owl:Class ;
    rdfs:label "Execution Strategy"@en ;
    rdfs:comment "Partial-order strategy specification; no executable schedule."@en ;
    rdfs:subClassOf ac:ActionComponent .

n2:SuppressionSpecification a owl:Class ;
    rdfs:label "Suppression Specification"@en ;
    rdfs:comment "Hypothesis that an intervention suppresses a specified causal link."@en ;
    rdfs:subClassOf ac:ActionComponent .

n3:Expression a owl:Class ;
    rdfs:label "Expression"@en ;
    rdfs:comment "A symbolic expression. Expressions are not physical events or truth assertions."@en ;
    rdfs:subClassOf n24:Description .

n3:QuantityConstant a owl:Class ;
    rdfs:label "Quantity Constant"@en ;
    rdfs:comment "Explicit decimal quantity and unit, including dimensionless quantities with an explicit unit identifier."@en ;
    rdfs:subClassOf n3:ValueExpression .

n5:AdaptationSpecification a owl:Class ;
    rdfs:label "Adaptation Specification"@en ;
    rdfs:comment "Proposal for updating an existing model parameter under a validity condition."@en ;
    rdfs:subClassOf ac:ActionComponent .

n5:ControllerKind a owl:Class ;
    rdfs:label "Controller Kind"@en ;
    rdfs:comment "Conceptual controller family."@en ;
    rdfs:subClassOf n24:Concept .

n9:FrameProjectionSpecification a owl:Class ;
    rdfs:label "Frame Projection Specification"@en ;
    rdfs:comment "Contract mapping one immutable source record to a frame view."@en ;
    rdfs:subClassOf ac:ProjectionSpecification .

n9:PathStep a owl:Class ;
    rdfs:label "Path Step"@en ;
    rdfs:comment "One ordered predicate traversal; no executable attachment."@en ;
    rdfs:subClassOf n24:Description .

n10:PathElement a owl:Class ;
    rdfs:label "Path Element"@en ;
    rdfs:comment "Ordered symbolic location specification, not a measured pose."@en ;
    rdfs:subClassOf ac:ActionComponent .

n12:EquationTerm a owl:Class ;
    rdfs:label "Equation Term"@en ;
    rdfs:comment "One signed scalar term in a bounded additive equation."@en ;
    rdfs:subClassOf ac:ActionComponent .

n12:QuantityContract a owl:Class ;
    rdfs:label "Quantity Contract"@en ;
    rdfs:comment "Explicit dimension vector and exact unit identifier for a scalar parameter. SI basis order: length,mass,time,current,temperature,amount,luminous intensity."@en ;
    rdfs:subClassOf ac:ActionComponent .

n14:TargetRegionSpecification a owl:Class ;
    rdfs:label "Target region specification"@en ;
    rdfs:comment "Described target region, optionally specified by a robot-motion pose target."@en ;
    rdfs:subClassOf ac:EntitySpecification .

n15:OptimizationDirection a owl:Class ;
    rdfs:label "Optimization Direction"@en ;
    rdfs:comment "Minimization or maximization of a metric."@en ;
    rdfs:subClassOf n24:Concept .

n16:ArgumentBinding a owl:Class ;
    rdfs:label "Argument Binding"@en ;
    rdfs:comment "Associates one signature position with an entity specification in a relation pattern."@en ;
    rdfs:subClassOf n24:Description .

n18:ModelArtifact a owl:Class ;
    rdfs:label "Model Artifact"@en ;
    rdfs:comment "Reference to a solver, geometry asset or other external artifact; no download or execution is implied."@en ;
    rdfs:subClassOf n24:InformationObject .

n18:ModelValidationRecord a owl:Class ;
    rdfs:label "Model Validation Record"@en ;
    rdfs:comment "Record linking a model to an external validation artifact and its stated scope. Not automatic acceptance."@en ;
    rdfs:subClassOf ac:EvidenceRecord .

n19:CoordinateSpecification a owl:Class ;
    rdfs:label "Coordinate specification"@en ;
    rdfs:comment "Abstract coordinate identity and linear or angular kind; grounding is injected."@en ;
    rdfs:subClassOf ac:ActionComponent .

n19:FrameSpecification a owl:Class ;
    rdfs:label "Frame specification"@en ;
    rdfs:comment "Description of a reference frame; no world transform is asserted."@en ;
    rdfs:subClassOf ac:EntitySpecification .

n19:MotionBounds a owl:Class ;
    rdfs:label "Motion bounds"@en ;
    rdfs:comment "Descriptive speed and acceleration bounds, not proof of trajectory feasibility."@en ;
    rdfs:subClassOf ac:ConstraintSpecification .

n19:WaypointSpecification a owl:Class ;
    rdfs:label "Waypoint specification"@en ;
    rdfs:comment "An indexed pose target in a path."@en ;
    rdfs:subClassOf ac:ActionComponent .

n24:PhysicalAgent a owl:Class ;
    rdfs:label "Physical agent"@en,
        "Agente fisico"@it ;
    rdfs:comment """A PhysicalObject that is capable of self-representing (conceptualizing) a Description in order to plan an Action. 
A PhysicalAgent is a substrate for (actsFor) a Social Agent""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Agent,
        n24:PhysicalObject .

n24:PhysicalBody a owl:Class ;
    rdfs:label "Physical body"@en ;
    rdfs:comment "Physical bodies are PhysicalObject(s), for which we tend to neutralize any possible artifactual character. They can have several granularity levels: geological, chemical, physical, biological, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:PhysicalObject .

n24:Relation a owl:Class ;
    rdfs:label "Relation"@en,
        "Relazione"@it ;
    rdfs:comment """Relations are descriptions that can be considered as the counterpart of formal relations (that are included in the FormalEntity class).
For example, 'givingGrantToInstitution(x,y,z)' with three argument types: Provider(x),Grant(y),Recipient(z), can have a Relation counterpart: 'GivingGrantToInstitution', which defines three Concept instances: Provider,Grant,Recipient.
Since social objects are not formal entities, Relation includes here any 'relation-like' entity in common sense, including social relations.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Description .

n24:SpaceRegion a owl:Class ;
    rdfs:label "Space region"@en,
        "Regione di spazio"@it ;
    rdfs:comment "Any Region in a dimensional space that is used to localize an Entity ; i.e., it is not used to represent some characteristic (e.g. it excludes time intervals, colors, size values, judgment values, etc.). Differently from a Place , a space region has a specific dimensional space." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Region ;
    owl:disjointWith n24:TimeInterval .

n24:definesRole a owl:ObjectProperty ;
    rdfs:label "defines role"@en,
        "definisce il ruolo"@it ;
    rdfs:comment "A relation between a description and a role, e.g. the recipe for a cake defines the role 'ingredient'." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Role ;
    rdfs:subPropertyOf n24:defines ;
    owl:inverseOf n24:isRoleDefinedIn .

n24:definesTask a owl:ObjectProperty ;
    rdfs:label "defines task"@en,
        "definisce il task"@it ;
    rdfs:comment "A relation between a description and a task, e.g. the recipe for a cake defines the task 'boil'." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Task ;
    rdfs:subPropertyOf n24:defines ;
    owl:inverseOf n24:isTaskDefinedIn .

n24:directlyFollows a owl:ObjectProperty ;
    rdfs:label "directly follows"@en,
        "segue direttamente"@it ;
    rdfs:comment "The intransitive follows relation. For example, Wednesday directly precedes Thursday. Directness of precedence depends on the designer conceptualization." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:follows ;
    owl:inverseOf n24:directlyPrecedes .

n24:hasComponent a owl:ObjectProperty ;
    rdfs:label "has component"@en,
        "ha componente"@it ;
    rdfs:comment "The hasPart relation without transitivity, holding between an Object (the system) and another (the component), and assuming a Design that structures the Object." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:hasPart ;
    owl:inverseOf n24:isComponentOf .

n24:hasDataValue a owl:DatatypeProperty ;
    rdfs:label "has data value"@en,
        "ha valore"@it ;
    rdfs:comment """A datatype property that encodes values from a datatype for an Entity. 
There are several ways to encode values in DOLCE (Ultralite):

1) Directly assert an xsd:_ value to an Entity by using hasDataValue
2) Assert a Region for an Entity by using hasRegion, and then assert an xsd:_ value to that Region, by using hasRegionDataValue
3) Assert a Quality for an Entity by using hasQuality, then assert a Region for that Quality, and assert an xsd:_ value to that Region, by using hasRegionDataValue
4) When the value is required, but not directly observed, assert a Parameter for an xsd:_ value by using hasParameterDataValue, and then associate the Parameter to an Entity by using isConstraintFor
5) When the value is required, but not directly observed, you can also assert a Parameter for a Region by using parametrizes, and then assert an xsd:_ value to that Region, by using hasRegionDataValue

The five approaches obey different requirements. 
For example, a simple value can be easily asserted by using pattern (1), but if one needs to assert an interval between two values, a Region should be introduced to materialize that interval, as pattern (2) suggests. 
Furthermore, if one needs to distinguish the individual Quality of a value, e.g. the particular nature of the density of a substance, pattern (3) can be used. 
Patterns (4) and (5) should be used instead when a constraint or a selection is modeled, independently from the actual observation of values in the real world.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl .

n24:hasParticipant a owl:ObjectProperty ;
    rdfs:label "has participant"@en,
        "ha come partecipante"@it ;
    rdfs:comment "A relation between an object and a process, e.g. 'John took part in the discussion', 'a large mass of snow fell during the avalanche', or 'a cook, some sugar, flour, etc. are all present in the cooking of a cake'." ;
    rdfs:domain n24:Event ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Object ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isParticipantIn .

n24:hasRegion a owl:ObjectProperty ;
    rdfs:label "has region"@en,
        "ha attributo"@it ;
    rdfs:comment "A relation between entities and regions, e.g. 'the number of wheels of that truck is 12', 'the time of the experiment is August 9th, 2004', 'the whale has been localized at 34 degrees E, 20 degrees S'." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Region ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isRegionFor .

n24:hasSetting a owl:ObjectProperty ;
    rdfs:label "has setting"@en,
        "è nel contesto di"@it ;
    rdfs:comment "A relation between entities and situations, e.g. 'this morning I've prepared my coffee with a new fantastic Arabica', i.e.: (an amount of) a new fantastic Arabica hasSetting the preparation of my coffee this morning." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Situation ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isSettingFor .

n24:isExpressedBy a owl:ObjectProperty ;
    rdfs:label "is expressed by"@en,
        "è espresso da"@it ;
    rdfs:comment """A relation between a dul:SocialObject (the 'meaning') and a dul:InformationObject (the 'expression'). 
For example: 'A Beehive is a structure in which bees are kept, typically in the form of a dome or box.' (Oxford dictionary)'; 'the term Beehive expresses the concept Beehive in my apiculture ontology'.
The intuition for 'meaning' is intended to be very broad. A separate, large comment is included in the encoding of 'expresses', for those who want to investigate more on what kind of meaning can be represented in what form.""" ;
    rdfs:domain n24:SocialObject ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:InformationObject ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isParticipantIn a owl:ObjectProperty ;
    rdfs:label "is participant in"@en,
        "è un partecipante a"@it ;
    rdfs:comment "A relation between an object and a process, e.g. 'John took part in the discussion', 'a large mass of snow fell during the avalanche', or 'a cook, some sugar, flour, etc. are all present in the cooking of a cake'." ;
    rdfs:domain n24:Object ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Event ;
    rdfs:subPropertyOf n24:associatedWith .

n24:isRelatedToDescription a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "is related to description"@en,
        "è associata alla descrizione"@it ;
    rdfs:comment "Any relation between descriptions." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isRelatedToDescription .

n24:precedes a owl:ObjectProperty,
        owl:TransitiveProperty ;
    rdfs:label "precedes"@en,
        "precede"@it ;
    rdfs:comment """A relation between entities, expressing a 'sequence' schema. 
E.g. 'year 1999 precedes 2000', 'deciding what coffee to use' precedes 'preparing coffee', 'World War II follows World War I', 'in the Milan to Rome autoroute, Bologna precedes Florence', etc.
It can then be used between tasks, processes, time intervals, spatially locate objects, situations, etc. 
Subproperties can be defined in order to distinguish the different uses.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

ac:BindingRecord a owl:Class ;
    rdfs:label "Binding Record"@en ;
    rdfs:comment "Optional externally supplied binding of an entity specification to a world entity identifier." ;
    rdfs:subClassOf n24:InformationObject .

ac:ObjectiveSpecification a owl:Class ;
    rdfs:label "Objective Specification"@en ;
    rdfs:comment "Optimization preference; distinct from a hard constraint." ;
    rdfs:subClassOf ac:ActionComponent .

ac:PerspectiveDescription a owl:Class ;
    rdfs:label "Perspective Description"@en ;
    rdfs:comment "Extension point for frames, image schemas, force dynamics and other views." ;
    rdfs:subClassOf ac:ActionComponent .

ac:ProjectionArtifact a owl:Class ;
    rdfs:label "Projection Artifact"@en ;
    rdfs:comment "Derived output from a specified version of a knowledge record; not inherently executable." ;
    rdfs:subClassOf n24:InformationObject ;
    owl:disjointWith n24:Event,
        n24:PhysicalObject .

n1:AssessmentOutcome a owl:Class ;
    rdfs:label "Assessment Outcome"@en ;
    rdfs:comment "Met, unmet or undetermined under a stated context."@en ;
    rdfs:subClassOf n24:Concept .

n1:CapabilityAssessment a owl:Class ;
    rdfs:label "Capability Assessment"@en ;
    rdfs:comment "Evidence-backed contextual assessment of a requirement, not a provider or robot."@en ;
    rdfs:subClassOf n24:InformationObject .

n2:MechanismSpecification a owl:Class ;
    rdfs:label "Mechanism Specification"@en ;
    rdfs:comment "Explanatory mechanism description, with explicit assumptions and applicability."@en ;
    rdfs:subClassOf ac:ActionComponent .

n3:Comparison a owl:Class ;
    rdfs:label "Comparison"@en ;
    rdfs:comment "Numeric comparison of two value expressions. Incompatible or absent units yield unknown."@en ;
    rdfs:subClassOf n3:BooleanExpression .

n4:DiagnosticCausalAssociation a owl:Class ;
    rdfs:label "Diagnostic Causal Association"@en ;
    rdfs:comment "Proposed association of a diagnostic hypothesis with an explanatory causal link; never a proof of root cause."@en ;
    rdfs:subClassOf ac:ActionComponent .

n6:AffordanceSpecification a owl:Class ;
    rdfs:label "Affordance Specification"@en ;
    rdfs:comment "Conditional action possibility relative to capability requirements and context."@en ;
    rdfs:subClassOf ac:ActionComponent .

n6:MaterialSpecification a owl:Class ;
    rdfs:label "Material Specification"@en ;
    rdfs:comment "Intensional material/composition description; no actual material instance."@en ;
    rdfs:subClassOf ac:ActionComponent .

n7:FailureProfile a owl:Class ;
    rdfs:label "Failure Profile"@en ;
    rdfs:comment "Potential failure descriptions, diagnostic hypotheses and recovery options."@en ;
    rdfs:subClassOf ac:ActionComponent .

n8:ForceParticipant a owl:Class ;
    rdfs:label "Force Participant"@en ;
    rdfs:comment "Entity specification in a qualitative force role."@en ;
    rdfs:subClassOf ac:ActionComponent .

n8:Resultant a owl:Class ;
    rdfs:label "Resultant"@en ;
    rdfs:comment "Described resultant motion/rest/unknown; never inferred from a force magnitude."@en ;
    rdfs:subClassOf n24:Concept .

n8:Tendency a owl:Class ;
    rdfs:label "Tendency"@en ;
    rdfs:comment "Qualitative tendency toward motion or rest."@en ;
    rdfs:subClassOf n24:Concept .

n9:SlotOverride a owl:Class ;
    rdfs:label "Slot Override"@en ;
    rdfs:comment "Explicit replacement or suppression of an inherited default."@en ;
    rdfs:subClassOf n24:Description .

n10:ImageSchemaDescription a owl:Class ;
    rdfs:label "Image Schema Description"@en ;
    rdfs:comment "Relational schematic perspective; geometric truth is externally grounded."@en ;
    rdfs:subClassOf ac:PerspectiveDescription .

n10:SourcePathGoalSchema a owl:Class ;
    rdfs:label "Source Path Goal Schema"@en ;
    rdfs:comment "Theme moving from source through ordered path elements to goal."@en ;
    rdfs:subClassOf n10:ImageSchemaDescription .

n12:AdditiveEquationContract a owl:Class ;
    rdfs:label "Additive Equation Contract"@en ;
    rdfs:comment "A bounded scalar additive equation: result equals signed sum of terms. No general equation parser or physical validity inference."@en ;
    rdfs:subClassOf ac:ActionComponent .

n14:ManipulationMotionRequirement a owl:Class ;
    rdfs:label "Manipulation motion requirement"@en ;
    rdfs:comment "Abstract motion kind and controlled entity required by manipulation; no provider binding or motion execution."@en ;
    rdfs:subClassOf ac:ActionComponent .

n15:AggregationPolicy a owl:Class ;
    rdfs:label "Aggregation Policy"@en ;
    rdfs:comment "Interpretation of multiple objectives; no optimizer is implemented."@en ;
    rdfs:subClassOf n24:Concept .

n17:PhaseStructure a owl:Class ;
    rdfs:label "Phase Structure"@en ;
    rdfs:comment "A finite graph describing action phases and guarded transitions; not an execution episode or scheduler."@en ;
    rdfs:subClassOf ac:ActionComponent .

n19:PositionTarget a owl:Class ;
    rdfs:label "Position target"@en ;
    rdfs:comment "Three scalar position parameters in the containing pose reference frame."@en ;
    rdfs:subClassOf ac:ActionComponent .

n20:ControlPointSpecification a owl:Class ;
    rdfs:label "Control Point Specification"@en ;
    rdfs:comment "Point where expected and observed sensory conditions can be compared."@en ;
    rdfs:subClassOf ac:ActionComponent .

n20:EventBoundarySpecification a owl:Class ;
    rdfs:label "Event Boundary Specification"@en ;
    rdfs:comment "Described phase-boundary condition, not an event occurrence."@en ;
    rdfs:subClassOf ac:ActionComponent .

n20:SensoryExpectation a owl:Class ;
    rdfs:label "Sensory Expectation"@en ;
    rdfs:comment "Predicted sensory condition and modality; not an observation."@en ;
    rdfs:subClassOf ac:ActionComponent .

n24:actsThrough a owl:ObjectProperty ;
    rdfs:label "acts through"@en,
        "agisce mediante"@it ;
    rdfs:comment "The relation holding between a PhysicalAgent and a SocialAgent. In principle, a SocialAgent requires at least one PhysicalAgent in order to act, but this dependency can be 'delegated', e.g. a university can be acted for by a department, which is acted for by physical agents. AKA isActedBy" ;
    rdfs:domain n24:SocialAgent ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Agent ;
    rdfs:subPropertyOf n24:associatedWith .

n24:defines a owl:ObjectProperty ;
    rdfs:label "defines"@en,
        "definisce"@it ;
    rdfs:comment "A relation between a Description and a Concept, e.g. a Workflow for a governmental Organization defines the Role 'officer', or 'the Italian Traffic Law defines the role Vehicle'." ;
    rdfs:domain n24:Description ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:usesConcept ;
    owl:inverseOf n24:isDefinedIn .

n24:directlyPrecedes a owl:ObjectProperty ;
    rdfs:label "directly precedes"@en,
        "precede direttamente"@it ;
    rdfs:comment "The intransitive precedes relation. For example, Monday directly precedes Tuesday. Directness of precedence depends on the designer conceptualization." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:precedes .

n24:isDefinedIn a owl:ObjectProperty ;
    rdfs:label "is defined in"@en,
        "è definito in"@it ;
    rdfs:comment "A relation between a Description and a Concept, e.g. a Workflow for a governmental Organization defines the Role 'officer', or 'the Italian Traffic Law defines the role Vehicle'." ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:isConceptUsedIn .

n24:isDescribedBy a owl:ObjectProperty ;
    rdfs:label "is described by"@en,
        "è descritto da"@it ;
    rdfs:comment """The relation between an Entity and a Description: a Description gives a unity to a Collection of parts (the components), or constituents, by assigning a Role to each of them in the context of a whole Object (the system).
A same Entity can be given different descriptions, for example, an old cradle can be given a unifying Description based on the original aesthetic design, the functionality it was built for, or a new aesthetic functionality in which it can be used as a flower pot.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:associatedWith .

n24:satisfies a owl:ObjectProperty ;
    rdfs:label "satisfies"@en,
        "soddisfa"@it ;
    rdfs:comment "A relation between a Situation and a Description, e.g. the execution of a Plan satisfies that plan." ;
    rdfs:domain n24:Situation ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Description ;
    rdfs:subPropertyOf n24:associatedWith .

ac:AnnotationRecord a owl:Class ;
    rdfs:label "Annotation Record"@en ;
    rdfs:comment "Record of an annotator contribution; an information record, not the computation event." ;
    rdfs:subClassOf n24:InformationObject .

ac:ConstraintSpecification a owl:Class ;
    rdfs:label "Constraint Specification"@en ;
    rdfs:comment "Requirement restricting permissible actions, states or parameters." ;
    rdfs:subClassOf ac:ConditionSpecification .

ac:Statement a owl:Class ;
    rdfs:label "Statement"@en ;
    rdfs:comment "Reified subject/predicate/object statement. The predicate is encoded as an IRI literal; no RDF assertion is implied." ;
    rdfs:subClassOf n24:InformationObject .

n1:ProcessModuleRequirement a owl:Class ;
    rdfs:label "Process Module Requirement"@en ;
    rdfs:comment "Abstract process interface requirement, independent of a provider API."@en ;
    rdfs:subClassOf ac:ActionComponent .

n3:ValueExpression a owl:Class ;
    rdfs:label "Value Expression"@en ;
    rdfs:comment "Quantity expression with a unit identifier."@en ;
    rdfs:subClassOf n3:Expression .

n5:DynamicalModelSpecification a owl:Class ;
    rdfs:label "Dynamical Model Specification"@en ;
    rdfs:comment "Partial state/input/output description of a physical model; does not establish controllability or dynamics fidelity."@en ;
    rdfs:subClassOf n18:PhysicalModelSpecification .

n5:VariableAssignment a owl:Class ;
    rdfs:label "Variable Assignment"@en ;
    rdfs:comment "Position and dimension of a model variable in a declared vector role."@en ;
    rdfs:subClassOf ac:ActionComponent .

n5:VariableRole a owl:Class ;
    rdfs:label "Variable Role"@en ;
    rdfs:comment "State, input, output or disturbance role."@en ;
    rdfs:subClassOf n24:Concept .

n6:EntityDescription a owl:Class ;
    rdfs:label "Entity Description"@en ;
    rdfs:comment "Record-scoped descriptive facet referencing a shared entity specification."@en ;
    rdfs:subClassOf ac:ActionComponent .

n7:FailureKind a owl:Class ;
    rdfs:label "Failure Kind"@en ;
    rdfs:comment "Reusable category of failure."@en ;
    rdfs:subClassOf n24:Concept .

n7:FailureSpecification a owl:Class ;
    rdfs:label "Failure Specification"@en ;
    rdfs:comment "Description of a failure mode; not an observed failure event."@en ;
    rdfs:subClassOf ac:ActionComponent .

n8:StrengthRelation a owl:Class ;
    rdfs:label "Strength Relation"@en ;
    rdfs:comment "Qualitative relative strength, with unknown explicitly representable."@en ;
    rdfs:subClassOf n24:Concept .

n11:SourceSnapshot a owl:Class ;
    rdfs:label "Source Snapshot"@en ;
    rdfs:comment "Versioned source content locator and optional exact supporting excerpt."@en ;
    rdfs:subClassOf n24:InformationObject .

n13:MonitoringPoint a owl:Class ;
    rdfs:label "Monitoring Point"@en ;
    rdfs:comment "Conceptual point at which checking is requested."@en ;
    rdfs:subClassOf n24:Concept .

n13:ObservationRequirement a owl:Class ;
    rdfs:label "Observation Requirement"@en ;
    rdfs:comment "An abstract input needed to evaluate a monitored condition."@en ;
    rdfs:subClassOf ac:ActionComponent .

n15:ObjectiveSet a owl:Class ;
    rdfs:label "Objective Set"@en ;
    rdfs:comment "Objective terms and declared aggregation policy, separately from hard constraints."@en ;
    rdfs:subClassOf ac:ActionComponent .

n16:RelationPattern a owl:Class ;
    rdfs:label "Relation Pattern"@en ;
    rdfs:comment "Application of a relation symbol to conceptual entity specifications. Not an asserted world relation."@en ;
    rdfs:subClassOf ac:ActionComponent .

n18:ModelVariable a owl:Class ;
    rdfs:label "Model Variable"@en ;
    rdfs:comment "Variable bound to a reusable parameter specification."@en ;
    rdfs:subClassOf ac:ActionComponent .

n18:PhysicalPropertySpecification a owl:Class ;
    rdfs:label "Physical Property Specification"@en ;
    rdfs:comment "Description of a physical property of an entity specification; no measured object is implied."@en ;
    rdfs:subClassOf ac:ActionComponent .

n19:CoordinateTarget a owl:Class ;
    rdfs:label "Coordinate target"@en ;
    rdfs:comment "Target parameter and error/tolerance parameters for one coordinate."@en ;
    rdfs:subClassOf ac:ActionComponent .

n24:Action a owl:Class ;
    rdfs:label "Action"@en,
        "Azione"@it ;
    rdfs:comment "An Event with at least one Agent that isParticipantIn it, and that executes a Task that typically isDefinedIn a Plan, Workflow, Project, etc." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:minCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n24:executesTask ],
        [ a owl:Restriction ;
            owl:onProperty n24:hasParticipant ;
            owl:someValuesFrom n24:Agent ],
        n24:Event .

n24:Plan a owl:Class ;
    rdfs:label "Plan"@en,
        "Piano"@it ;
    rdfs:comment "A Description having an explicit Goal, to be achieved by executing the plan" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:hasComponent ;
            owl:someValuesFrom n24:Goal ],
        n24:Description .

n24:isRegionFor a owl:ObjectProperty ;
    rdfs:label "is region for"@en,
        "è una regione di"@it ;
    rdfs:comment "A relation between entities and regions, e.g. 'the color of my car is red'." ;
    rdfs:domain n24:Region ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

n3:OutcomeSpecification a owl:Class ;
    rdfs:label "Outcome Specification"@en ;
    rdfs:comment "Desired or expected transition between conditions. Does not assert that it occurred."@en ;
    rdfs:subClassOf ac:ActionComponent .

n3:StateScope a owl:Class ;
    rdfs:label "State Scope"@en ;
    rdfs:comment "Symbolic evaluation scope. It is not an observed world state."@en ;
    rdfs:subClassOf n24:Concept .

n6:GeometrySpecification a owl:Class ;
    rdfs:label "Geometry Specification"@en ;
    rdfs:comment "Geometry description with optional external asset reference."@en ;
    rdfs:subClassOf ac:ActionComponent .

n6:MaterialState a owl:Class ;
    rdfs:label "Material State"@en ;
    rdfs:comment "Described material state category."@en ;
    rdfs:subClassOf n24:Concept .

n8:ForceDynamicDescription a owl:Class ;
    rdfs:label "Force Dynamic Description"@en ;
    rdfs:comment "Qualitative opposing-force pattern; no measured force or physical prediction."@en ;
    rdfs:subClassOf ac:PerspectiveDescription .

n9:FrameDefinition a owl:Class ;
    rdfs:label "Frame Definition"@en ;
    rdfs:comment "Slot-based descriptive perspective, with bounded single inheritance."@en ;
    rdfs:subClassOf ac:PerspectiveDescription .

n9:FrameSlotValue a owl:Class ;
    rdfs:label "Frame Slot Value"@en ;
    rdfs:comment "Projected value with source-claim provenance, or an explicitly marked default."@en ;
    rdfs:subClassOf n24:InformationObject .

n15:ObjectiveTerm a owl:Class ;
    rdfs:label "Objective Term"@en ;
    rdfs:comment "Preference over a metric parameter with explicit optimization direction."@en ;
    rdfs:subClassOf ac:ObjectiveSpecification .

n19:OrientationTarget a owl:Class ;
    rdfs:label "Orientation target"@en ;
    rdfs:comment "Quaternion XYZW parameter descriptions; no actual orientation measurement."@en ;
    rdfs:subClassOf ac:ActionComponent .

n19:PoseTarget a owl:Class ;
    rdfs:label "Pose target"@en ;
    rdfs:comment "Desired position and optional orientation, relative to an explicit frame specification."@en ;
    rdfs:subClassOf ac:ActionComponent .

n20:SensoryModality a owl:Class ;
    rdfs:label "Sensory Modality"@en ;
    rdfs:comment "Abstract modality independent of sensor implementation."@en ;
    rdfs:subClassOf n24:Concept .

n24:InformationRealization a owl:Class ;
    rdfs:label "Information realization"@en,
        "Informazione concreta"@it ;
    rdfs:comment "A concrete realization of an InformationObject, e.g. the written document containing the text of a law." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Class ;
            owl:unionOf ( n24:Event n24:PhysicalObject ) ],
        [ a owl:Restriction ;
            owl:onProperty n24:realizes ;
            owl:someValuesFrom n24:InformationObject ],
        n24:InformationEntity .

n24:hasConstituent a owl:ObjectProperty ;
    rdfs:label "has constituent"@en,
        "ha costituente"@it ;
    rdfs:comment """'Constituency' depends on some layering of  the world described by the ontology. For example, scientific granularities (e.g. body-organ-tissue-cell) or ontological 'strata' (e.g. social-mental-biological-physical) are  typical layerings. 
Intuitively, a constituent is a part belonging to a lower layer. Since layering is actually a partition of the world described by the ontology, constituents are not properly classified as parts, although this kinship can be intuitive for common sense.
A desirable advantage of this distinction is that we are able to talk e.g. of physical constituents of non-physical objects (e.g. systems), while this is not possible in terms of parts.
Example of are the persons constituting a social system, the molecules constituting a person, the atoms constituting a river, etc. 
In all these examples, we notice a typical discontinuity between the constituted and the constituent object: e.g. a social system is conceptualized at a different layer from the persons that constitute it, a person is conceptualized at a different layer from the molecules that constitute them, and a river is conceptualized at a different layer from the atoms that constitute it.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isConstituentOf .

n24:isClassifiedBy a owl:ObjectProperty ;
    rdfs:label "is classified by"@en,
        "è classificato da"@it ;
    rdfs:comment "A relation between a Concept and an Entity, e.g. 'John is considered a typical rude man'; your last concert constitutes the achievement of a lifetime; '20-year-old means she's mature enough'." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:associatedWith .

ac:EvidenceKind a owl:Class ;
    rdfs:label "Evidence Kind"@en ;
    rdfs:comment "Kind of evidence, not a guarantee of correctness." ;
    rdfs:subClassOf n24:Concept .

ac:ParticipantRole a owl:Class ;
    rdfs:label "Participant Role"@en ;
    rdfs:comment "Role defined by an action description, such as source, destination or manipulated object." ;
    rdfs:subClassOf n24:Role .

ac:ProjectionSpecification a owl:Class ;
    rdfs:label "Projection Specification"@en ;
    rdfs:comment "Contract for a projection, including required information and missing-information policy." ;
    rdfs:subClassOf n24:Description .

ac:requiresCapability a owl:ObjectProperty ;
    rdfs:label "Requires Capability"@en ;
    rdfs:comment "Abstract capability required by a description. This relation states a requirement, not satisfaction, provider availability or execution readiness. Specialized properties retain the kind of requiring description."@en ;
    rdfs:domain n24:Description ;
    rdfs:range ac:CapabilityRequirement .

n3:ComparisonOperator a owl:Class ;
    rdfs:label "Comparison Operator"@en ;
    rdfs:comment "A supported decimal comparison operator."@en ;
    rdfs:subClassOf n24:Concept .

n7:RecoveryOption a owl:Class ;
    rdfs:label "Recovery Option"@en ;
    rdfs:comment "Guarded descriptive recovery option with an intended restored condition."@en ;
    rdfs:subClassOf ac:RecoverySpecification .

n11:KnowledgeContext a owl:Class ;
    rdfs:label "Knowledge Context"@en ;
    rdfs:comment "An explicit evaluation scope. Distinct contexts do not share truth or values automatically."@en ;
    rdfs:subClassOf n24:Description .

mt:MaterialTransferDescription a owl:Class ;
    rdfs:label "Material transfer description"@en ;
    rdfs:comment "Describes intended material transfer and quantity referents, independent of transfer technique."@en ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty mt:evaluationContext ],
        [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty mt:quantityBasis ],
        [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n22:hasTransferContract ],
        ac:ActionDescription .

n13:MonitoringRequirement a owl:Class ;
    rdfs:label "Monitoring Requirement"@en ;
    rdfs:comment "Specification of a condition check, observation dependencies and declared responses. Does not create sensors or run monitors."@en ;
    rdfs:subClassOf ac:ActionComponent .

n17:PhaseTransition a owl:Class ;
    rdfs:label "Phase Transition"@en ;
    rdfs:comment "Directed transition specification with an explicit guard. Does not assert a transition occurred."@en ;
    rdfs:subClassOf ac:ActionComponent .

n18:ModelEquation a owl:Class ;
    rdfs:label "Model Equation"@en ;
    rdfs:comment "Equation text with declared language and variable bindings; not evaluated by OWL."@en ;
    rdfs:subClassOf ac:ActionComponent .

n18:PropertyKind a owl:Class ;
    rdfs:label "Property Kind"@en ;
    rdfs:comment "Concept identifying a physical property kind."@en ;
    rdfs:subClassOf n24:Concept .

n24:Quality a owl:Class ;
    rdfs:label "Quality"@en,
        "Qualità"@it ;
    rdfs:comment """Any aspect of an Entity (but not a part of it), which cannot exist without that Entity. For example, the way the surface of a specific PhysicalObject looks like, or the specific light of a place at a certain time, are examples of Quality, while the encoding of a Quality into e.g. a PhysicalAttribute should be modeled as a Region. 
From the design viewpoint, the Quality-Region distinction is useful only when individual aspects of an Entity are considered in a domain of discourse. 
For example, in an automotive context, it would be irrelevant to consider the aspects of car windows for a specific car, unless the factory wants to check a specific window against design parameters (anomaly detection). 
On the other hand, in an antiques context, the individual aspects for a specific piece of furniture are a major focus of attention, and may constitute the actual added value, because the design parameters for old furniture are often not fixed, and may not be viewed as 'anomalies'.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Quality ;
            owl:onProperty n24:hasPart ],
        [ a owl:Restriction ;
            owl:onProperty n24:hasRegion ;
            owl:someValuesFrom n24:Region ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Quality ;
            owl:onProperty n24:hasConstituent ],
        n24:Entity .

n24:classifies a owl:ObjectProperty ;
    rdfs:label "classifies"@en,
        "classifica"@it ;
    rdfs:comment "A relation between a Concept and an Entity, e.g. the Role 'student' classifies a Person 'John'." ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isClassifiedBy .

n24:isRelatedToConcept a owl:ObjectProperty,
        owl:SymmetricProperty ;
    rdfs:label "is related to concept"@en,
        "è associato al concetto"@it ;
    rdfs:comment "Any relation between concepts, e.g. superordinated, conceptual parthood, having a parameter, having a task, superordination, etc." ;
    rdfs:domain n24:Concept ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Concept ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isRelatedToConcept .

xsd:dateTime a rdfs:Datatype .

ac:ApplicabilitySpecification a owl:Class ;
    rdfs:label "Applicability Specification"@en ;
    rdfs:comment "Conditions limiting the scope of a claim or model." ;
    rdfs:subClassOf ac:ConditionSpecification .

ac:EpistemicStatus a owl:Class ;
    rdfs:label "Epistemic Status"@en ;
    rdfs:comment "Workflow status of a claim, distinct from probability or physical truth." ;
    rdfs:subClassOf n24:Concept .

n1:StrategyStep a owl:Class ;
    rdfs:label "Strategy Step"@en ;
    rdfs:comment "Descriptive action step with abstract process requirements."@en ;
    rdfs:subClassOf ac:ActionComponent .

n5:ControllerSpecification a owl:Class ;
    rdfs:label "Controller Specification"@en ;
    rdfs:comment "Controller design description; no executable controller or stability guarantee."@en ;
    rdfs:subClassOf ac:ControlSpecification .

n7:DiagnosticHypothesis a owl:Class ;
    rdfs:label "Diagnostic Hypothesis"@en ;
    rdfs:comment "Candidate cause explaining a failure; evidence does not automatically establish root cause."@en ;
    rdfs:subClassOf ac:ActionComponent .

n24:isSettingFor a owl:ObjectProperty ;
    rdfs:label "is setting for"@en,
        "include"@it ;
    rdfs:comment "A relation between situations and entities, e.g. 'this morning I've prepared my coffee with a new fantastic Arabica', i.e.: the preparation of my coffee this morning is the setting for (an amount of) a new fantastic Arabica." ;
    rdfs:domain n24:Situation ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith .

n3:BooleanExpression a owl:Class ;
    rdfs:label "Boolean Expression"@en ;
    rdfs:comment "An expression evaluated as true, false or unknown under explicit external inputs."@en ;
    rdfs:subClassOf n3:Expression .

n9:SlotDefinition a owl:Class ;
    rdfs:label "Slot Definition"@en ;
    rdfs:comment "Named slot, cardinality, source mapping and optional unverified default."@en ;
    rdfs:subClassOf n24:Concept .

ac:CapabilityRequirement a owl:Class ;
    rdfs:label "Capability Requirement"@en ;
    rdfs:comment "Abstract required ability; no robot or implementation is required." ;
    rdfs:subClassOf ac:ActionComponent .

ac:KnowledgeClaim a owl:Class ;
    rdfs:label "Knowledge Claim"@en ;
    rdfs:comment "Evidence-bearing assertion about descriptive content; its statement is not automatically asserted as a world fact." ;
    rdfs:subClassOf n24:InformationObject .

ac:KnowledgeRecord a owl:Class ;
    rdfs:label "Knowledge Record"@en ;
    rdfs:comment "Versioned information artifact expressing a shared action description, with claims and annotation history." ;
    rdfs:subClassOf n24:InformationObject ;
    owl:disjointWith n24:Event,
        n24:PhysicalObject .

n2:CausalLink a owl:Class ;
    rdfs:label "Causal Link"@en ;
    rdfs:comment "Reified causal hypothesis connecting antecedent and consequent conditions through a mechanism."@en ;
    rdfs:subClassOf ac:ActionComponent .

n11:ClaimAssessment a owl:Class ;
    rdfs:label "Claim Assessment"@en ;
    rdfs:comment "Immutable-by-contract assessment of one claim in one record and context; does not replace the original claim status."@en ;
    rdfs:subClassOf n24:InformationObject .

n13:MonitoringResponse a owl:Class ;
    rdfs:label "Monitoring Response"@en ;
    rdfs:comment "Declared response category; not a command or executable recovery plan."@en ;
    rdfs:subClassOf n24:Concept .

n18:PhysicalModelSpecification a owl:Class ;
    rdfs:label "Physical Model Specification"@en ;
    rdfs:comment "Model description with variables, assumptions, applicability and external artifacts."@en ;
    rdfs:subClassOf ac:PhysicalModelSpecification .

n24:Parameter a owl:Class ;
    rdfs:label "Parameter"@en,
        "Parametro"@it ;
    rdfs:comment "A Concept that classifies a Region; the difference between a Region and a Parameter is that regions represent sets of observable values, e.g. the height  of a given building, while parameters represent constraints or selections on observable values, e.g. 'VeryHigh'. Therefore, parameters can also be used to constrain regions, e.g. VeryHigh on a subset of values of the Region Height applied to buildings, or to add an external selection criterion , such as measurement units, to regions, e.g. Meter on a subset of values from the Region Length applied to the Region Length applied to roads." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Parameter ;
            owl:onProperty n24:hasPart ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Region ;
            owl:onProperty n24:classifies ],
        n24:Concept ;
    owl:disjointWith n24:Role .

n24:Task a owl:Class ;
    rdfs:label "Task"@en,
        "Task"@it ;
    rdfs:comment """An EventType that classifies an Action to be executed. 
For example, reaching a destination is a task that can be executed by performing certain actions, e.g. driving a car, buying a train ticket, etc. 
The actions to execute a task can also be organized according to a Plan that is not the same as the one that defines the task (if any). 
For example, reaching a destination could be defined by a plan to get on holidays, while the plan to execute the task can consist of putting some travels into a sequence.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Role ;
            owl:onProperty n24:isTaskOf ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Action ;
            owl:onProperty n24:isExecutedIn ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Description ;
            owl:onProperty n24:isTaskDefinedIn ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Task ;
            owl:onProperty n24:hasPart ],
        n24:EventType .

ac:EvidenceRecord a owl:Class ;
    rdfs:label "Evidence Record"@en ;
    rdfs:comment "Record of a source or result supporting or challenging a claim." ;
    rdfs:subClassOf n24:InformationObject .

n19:RobotMotionDescription a owl:Class ;
    rdfs:label "Robot motion description"@en ;
    rdfs:comment "Intended robot-part motion; no occurring motion or robot model asserted."@en ;
    rdfs:subClassOf ac:ActionDescription .

n22:TransferContract a owl:Class ;
    rdfs:label "Transfer Contract"@en ;
    rdfs:comment "Binds transfer quantities to a particular source, destination and substance; intended transfer only."@en ;
    rdfs:subClassOf ac:ActionComponent .

n24:PhysicalObject a owl:Class ;
    rdfs:label "Physical object"@en,
        "Oggetto fisico"@it ;
    rdfs:comment "Any Object that has a proper space region. The prototypical physical object has also an associated mass, but the nature of its mass can greatly vary based on the epistemological status of the object (scientifically measured, subjectively possible, imaginary)." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:PhysicalObject ;
            owl:onProperty n24:hasPart ],
        n24:Object ;
    owl:disjointWith n24:SocialObject .

n14:ObjectManipulationDescription a owl:Class ;
    rdfs:label "Object manipulation description"@en ;
    rdfs:comment "Intended interaction with a described object; no occurring action, object or robot asserted."@en ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n14:evaluationContext ],
        [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n14:object ],
        [ a owl:Restriction ;
            owl:maxCardinality "1"^^xsd:nonNegativeInteger ;
            owl:onProperty n14:manipulator ],
        ac:ActionDescription .

n24:Collection a owl:Class ;
    rdfs:label "Collection"@en,
        "Collezione"@it ;
    rdfs:comment """Any container for entities that share one or more common properties. E.g. "stone objects", "the nurses", "the Louvre Aegyptian collection", all the elections for the Italian President of the Republic. 
A collection is not a logical class: a collection is a first-order entity, while a class is second-order.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Collection ;
            owl:onProperty n24:hasPart ],
        n24:SocialObject .

n24:SocialAgent a owl:Class ;
    rdfs:label "Social agent"@en,
        "Agente sociale"@it ;
    rdfs:comment "Any individual whose existence is granted simply by its social communicability and capability of action (through some PhysicalAgent)." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:actsThrough ;
            owl:someValuesFrom n24:PhysicalAgent ],
        n24:Agent,
        n24:SocialObject .

n24:hasPart a owl:ObjectProperty,
        owl:TransitiveProperty ;
    rdfs:label "has part"@en,
        "ha parte"@it ;
    rdfs:comment """A schematic relation between any entities, e.g. 'the human body has a brain as part', '20th century contains year 1923', 'World War II includes the Pearl Harbour event'.
Subproperties and restrictions can be used to specialize hasPart for objects, events, etc.""" ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    rdfs:subPropertyOf n24:associatedWith ;
    owl:inverseOf n24:isPartOf .

n21:ToolInteractionDescription a owl:Class ;
    rdfs:label "Tool interaction description"@en ;
    rdfs:comment "Intended functional interaction through a tool; no robot, tool body or physical effect is asserted."@en ;
    rdfs:subClassOf ac:ActionDescription .

n24:TimeInterval a owl:Class ;
    rdfs:label "Time interval"@en,
        "Intervallo di tempo"@it ;
    rdfs:comment "Any Region in a dimensional space that aims at representing time." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Region .

n24:Agent a owl:Class ;
    rdfs:label "Agent"@en,
        "Agente"@it ;
    rdfs:comment "Additional comment: a computational agent can be considered as a PhysicalAgent that realizes a certain class of algorithms (that can be considered as instances of InformationObject) that allow to obtain some behaviors that are considered typical of agents in general. For an ontology of computational objects based on DOLCE see e.g. http://www.loa-cnr.it/COS/COS.owl, and http://www.loa-cnr.it/KCO/KCO.owl.",
        "Any agentive Object , either physical (e.g. a whale, a robot, an oak), or social (e.g. a corporation, an institution, a community)." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:Object .

n24:Role a owl:Class ;
    rdfs:label "Role"@en,
        "Ruolo"@it ;
    rdfs:comment "A Concept that classifies an Object" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Object ;
            owl:onProperty n24:classifies ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Role ;
            owl:onProperty n24:hasPart ],
        n24:Concept .

n24:Object a owl:Class ;
    rdfs:label "Object"@en,
        "Oggetto"@it ;
    rdfs:comment "Any physical, social, or mental object, or a substance. Following DOLCE Full, objects are always participating in some event (at least their own life), and are spatially located." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Object ;
            owl:onProperty n24:hasConstituent ],
        [ a owl:Restriction ;
            owl:onProperty n24:hasLocation ;
            owl:someValuesFrom n24:Entity ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Object ;
            owl:onProperty n24:hasPart ],
        [ a owl:Restriction ;
            owl:onProperty n24:isParticipantIn ;
            owl:someValuesFrom n24:Event ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Role ;
            owl:onProperty n24:isClassifiedBy ],
        n24:Entity ;
    owl:disjointWith n24:Quality .

ac:PhaseSpecification a owl:Class ;
    rdfs:label "Phase Specification"@en ;
    rdfs:comment "Description of a phase; not an event occurrence." ;
    rdfs:subClassOf ac:ActionComponent .

n24:Region a owl:Class ;
    rdfs:label "Region"@en,
        "Regione"@it ;
    rdfs:comment """Any region in a dimensional space (a dimensional space is a maximal Region), which can be used as a value for a quality of an Entity . For example, TimeInterval, SpaceRegion, PhysicalAttribute, Amount, SocialAttribute are all subclasses of Region. 
Regions are not data values in the ordinary knowledge representation sense; in order to get patterns for modelling data, see the properties: representsDataValue and hasDataValue""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Region ;
            owl:onProperty n24:overlaps ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Region ;
            owl:onProperty n24:precedes ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Region ;
            owl:onProperty n24:hasPart ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Region ;
            owl:onProperty n24:hasConstituent ],
        n24:Abstract .

n24:SocialObject a owl:Class ;
    rdfs:label "Social object"@en,
        "Oggetto sociale"@it ;
    rdfs:comment """Any Object that exists only within some communication Event, in which at least one PhysicalObject participates in. 
In other words, all objects that have been or are created in the process of social communication: for the sake of communication (InformationObject), for incorporating new individuals (SocialAgent, Place), for contextualizing existing entities (Situation), for collecting existing entities (Collection), or for describing existing entities (Description, Concept).
Being dependent on communication, all social objects need to be expressed by some information object (information object are self-expressing).""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:isExpressedBy ;
            owl:someValuesFrom n24:InformationObject ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:SocialObject ;
            owl:onProperty n24:hasPart ],
        n24:Object .

ac:ActionKind a owl:Class ;
    rdfs:label "Action Kind"@en ;
    rdfs:comment "A task concept identifying the kind of action described. No actual action occurrence is required." ;
    rdfs:subClassOf n24:Task .

n24:InformationObject a owl:Class ;
    rdfs:label "Information object"@en,
        "Oggetto informativo"@it ;
    rdfs:comment "A piece of information, such as a musical composition, a text, a word, a picture, independently from how it is concretely realized." ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:InformationEntity,
        n24:SocialObject ;
    owl:disjointWith n24:Situation,
        n24:SocialAgent .

ac:hasComponent a owl:ObjectProperty ;
    rdfs:label "Has Component"@en ;
    rdfs:domain ac:ActionDescription ;
    rdfs:range ac:ActionComponent ;
    rdfs:subPropertyOf n24:hasComponent .

n16:RelationDefinition a owl:Class ;
    rdfs:label "Relation Definition"@en ;
    rdfs:comment "A predicate symbol with an ordered argument signature. Its existence asserts no physical relation."@en ;
    rdfs:subClassOf n24:Concept .

n24:Event a owl:Class ;
    rdfs:label "Event"@en,
        "Evento"@it ;
    rdfs:comment """Any physical, social, or mental process, event, or state.

More theoretically, events can be classified in different ways, possibly based on 'aspect' (e.g. stative, continuous, accomplishement, achievement, etc.), on 'agentivity' (e.g. intentional, natural, etc.), or on 'typical participants' (e.g. human, physical, abstract, food, etc.).
Here no special direction is taken, and the following explains why: events are related to observable situations, and they can have different views at a same time.
If a position has to be suggested here anyway, the participant-based classification of events seems the most stable and appropriate for many modelling problems.

(1) Alternative aspectual views

Consider a same event 'rock erosion in the Sinni valley': it can be conceptualized as an accomplishment (what has brought a certain state to occur), as an achievement (the state resulting from a previous accomplishment), as a punctual event (if we collapse the time interval of the erosion into a time point), or as a transition (something that has changed from a state to a different one). 
In the erosion case, we could therefore have good motivations to shift from one aspect to another: a) causation focus, b) effectual focus, c) historical condensation, d) transition (causality).

The different views refer to the same event, but are still different: how to live with this seeming paradox? 
A typical solution e.g. in linguistics (cf. Levin's aspectual classes) and in DOLCE Full (cf. WonderWeb D18 axiomatization) is to classify events based on aspectual differences. But this solution would create different identities for a same event, where the difference is only based on the modeller's attitude.
An alternative solution is applied here, and exploits the notion of (observable) Situation; a Situation is a view, consistent with a Description, which can be observed of a set of entities. It can also be seen as a 'relational context' created by an observer on the basis of a 'frame'. Therefore, a Situation allows to create a context where each particular view can have a proper identity, while the Event preserves its own identity. 
For example, ErosionAsAccomplishment is a Situation where rock erosion is observed as a process leading to a certain achievement: the conditions (roles, parameters) that suggest such view are stated in a Description, which acts as a 'theory of accomplishments'. Similarly, ErosionAsTransition is a Situation where rock erosion is observed as an event that has changed a state to another: the conditions for such interpretation are stated in a different Description, which acts as a 'theory of state transitions'.
Consider that in no case the actual event is changed or enriched in parts by the aspectual view.

(2) Alternative intentionality views

Similarly to aspectual views, several intentionality views can be provided for a same Event. For example, one can investigate if an avalanche has been caused by immediate natural forces, or if there is any hint of an intentional effort to activate those natural forces.
Also in this case, the Event as such has not different identities, while the causal analysis generates situations with different identities, according to what Description is taken for interpreting the Event. 
On the other hand, if the possible actions of an Agent causing the starting of an avalanche are taken as parts of the Event, then this makes its identity change, because we are adding a part to it. 
Therefore, if intentionality is a criterion to classify events or not, this depends on if an ontology designer wants to consider causality as a relevant dimension for events' identity.

(3) Alternative participant views

A slightly different case is when we consider the basic participants to an Event. In this case, the identity of the Event is affected by the participating objects, because it depends on them. 
For example, if snow, mountain slopes, wind, waves, etc. are considered as an avalanche basic participants, or if we also want to add water, human agents, etc., that makes the identity of an avalanche change.
Anyway, this approach to event classification is based on the designer's choices, and more accurately mirrors lexical or commonsense classifications (see. e.g. WordNet 'supersenses' for verb synsets).

Ultimately, this discussion has no end, because realists will keep defending the idea that events in reality are not changed by the way we describe them, while constructivists will keep defending the idea that, whatever 'true reality' is about, it can't be modelled without the theoretical burden of how we observe and describe it. 
Both positions are in principle valid, but, if taken too radically, they focus on issues that are only partly relevant to the aim of computational ontologies, which only attempt to assist domain experts in representing what they want to conceptualize a certain portion of reality according to their own ideas. 
For this reason, in this ontology both events and situations are allowed, together with descriptions, in order to encode the modelling needs, independently from the position (if any) chosen by the designer.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Event ;
            owl:onProperty n24:hasConstituent ],
        [ a owl:Restriction ;
            owl:onProperty n24:hasParticipant ;
            owl:someValuesFrom n24:Object ],
        [ a owl:Restriction ;
            owl:allValuesFrom n24:Event ;
            owl:onProperty n24:hasPart ],
        [ a owl:Restriction ;
            owl:onProperty n24:hasTimeInterval ;
            owl:someValuesFrom n24:TimeInterval ],
        n24:Entity ;
    owl:disjointWith n24:Object,
        n24:Quality .

n24:Situation a owl:Class ;
    rdfs:label "Situation"@en,
        "Situazione"@it ;
    rdfs:comment """A view, consistent with ('satisfying') a Description, on a set of entities. 
It can also be seen as a 'relational context' created by an observer on the basis of a 'frame' (i.e. a Description). 
For example, a PlanExecution is a context including some actions executed by agents according to certain parameters and expected tasks to be achieved from a Plan; a DiagnosedSituation is a context of observed entities that is interpreted on the basis of a Diagnosis, etc.
Situation is also able to represent reified n-ary relations, where isSettingFor is the top-level relation for all binary projections of the n-ary relation. If used in a transformation pattern for n-ary relations, the designer should take care of creating only one subclass of Situation for each n-ary relation, otherwise the 'identification constraint' (Calvanese et al., IJCAI 2001) could be violated.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:onProperty n24:satisfies ;
            owl:someValuesFrom n24:Description ],
        n24:SocialObject ;
    owl:disjointWith n24:SocialAgent .

ac:ConditionSpecification a owl:Class ;
    rdfs:label "Condition Specification"@en ;
    rdfs:comment "A condition described symbolically. Does not assert that the condition holds." ;
    rdfs:subClassOf ac:ActionComponent .

ac:EntitySpecification a owl:Class ;
    rdfs:label "Entity Specification"@en ;
    rdfs:comment "Intensional specification of an entity slot; not a physical object or world instance." ;
    rdfs:subClassOf ac:ActionComponent .

ac:ActionDescription a owl:Class ;
    rdfs:label "Action Description"@en ;
    rdfs:comment "Conceptual description of an action, not an occurring action. Partial descriptions are permitted." ;
    rdfs:subClassOf n24:Description ;
    owl:disjointWith n24:Event,
        n24:PhysicalObject .

n24:Concept a owl:Class ;
    rdfs:label "Concept"@en,
        "Concetto"@it ;
    rdfs:comment """A Concept is a SocialObject, and isDefinedIn some Description; once defined, a Concept can be used in other Description(s). If a Concept isDefinedIn exactly one Description, see the LocalConcept class.
The classifies relation relates Concept(s) to Entity(s) at some TimeInterval""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf [ a owl:Restriction ;
            owl:allValuesFrom n24:Concept ;
            owl:onProperty n24:hasPart ],
        [ a owl:Restriction ;
            owl:onProperty n24:isDefinedIn ;
            owl:someValuesFrom n24:Description ],
        n24:SocialObject ;
    owl:disjointWith n24:Description,
        n24:InformationObject,
        n24:Situation,
        n24:SocialAgent .

ac:ParameterSpecification a owl:Class ;
    rdfs:label "Parameter Specification"@en ;
    rdfs:comment "Description of a parameter, including an optional value, unit and bounds. Absence means unspecified." ;
    rdfs:subClassOf ac:ActionComponent .

n24:Description a owl:Class ;
    rdfs:label "Description"@en,
        "Descrizione"@it ;
    rdfs:comment """A Description is a SocialObject that represents a conceptualization. 
It can be thought also as a 'descriptive context' that uses or defines concepts in order to create a view on a 'relational context' (cf. Situation) out of a set of data or observations. 
For example, a Plan is a Description of some actions to be executed by agents in a certain way, with certain parameters; a Diagnosis is a Description that provides an interpretation for a set of observed entities, etc.""" ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:subClassOf n24:SocialObject ;
    owl:disjointWith n24:InformationObject,
        n24:Situation,
        n24:SocialAgent .

n16:ArgumentDefinition a owl:Class ;
    rdfs:label "Argument Definition"@en ;
    rdfs:comment "One named position in a relation signature. Positions start at one."@en ;
    rdfs:subClassOf n24:Concept .

n24:associatedWith a owl:ObjectProperty,
        owl:SymmetricProperty,
        owl:TransitiveProperty ;
    rdfs:label "associatedWith" ;
    rdfs:comment "A catch-all object property, useful for alignment and querying purposes." ;
    rdfs:domain n24:Entity ;
    rdfs:isDefinedBy n23:DUL.owl ;
    rdfs:range n24:Entity ;
    owl:inverseOf n24:associatedWith .

ac:ActionComponent a owl:Class ;
    rdfs:label "Action Component"@en ;
    rdfs:comment "Reusable descriptive component of action knowledge." ;
    rdfs:subClassOf n24:Description ;
    owl:disjointWith n24:Event,
        n24:PhysicalObject .

n24:Entity a owl:Class ;
    rdfs:label "Entity"@en,
        "Entità"@it ;
    rdfs:comment "Anything: real, possible, or imaginary, which some modeller wants to talk about for some purpose." ;
    rdfs:isDefinedBy n23:DUL.owl .

[] a owl:AllDifferent ;
    owl:distinctMembers ( ac:CuratedExample ac:LLMGeneration ac:DocumentSource ac:LogicalDerivation ac:SimulationResult ac:Observation ) .

[] a owl:AllDisjointClasses ;
    owl:members ( n3:AtomicCondition n3:Conjunction n3:Disjunction n3:Negation n3:Comparison ) .

[] a owl:AllDisjointClasses ;
    owl:members ( n17:OrdinaryTransition n17:FailureTransition n17:RecoveryTransition ) .

[] a owl:AllDisjointClasses ;
    owl:members ( n3:ParameterReference n3:QuantityConstant ) .

[] a owl:AllDisjointClasses ;
    owl:members ( n17:SuccessfulTerminalPhase n17:FailedTerminalPhase ) .

[] a owl:AllDifferent ;
    owl:distinctMembers ( ac:Proposed ac:Supported ac:Disputed ac:Rejected ac:Unknown ) .

[] a owl:AllDisjointClasses ;
    owl:members ( n3:BooleanExpression n3:ValueExpression ) .

[] a owl:AllDisjointClasses ;
    owl:members ( n13:RelationObservation n13:QuantityObservation ) .

```
