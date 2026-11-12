# Perception and attention — 0.16.0

The fourth reusable domain is `perception-attention.ttl`:

`http://aicor.knowledge/ontologies/perception-attention.owl`

It imports robot motion, sensorimotor events, phases/transitions, monitoring and knowledge context. DUL is the foundation through the core. It does not import CoraPlex, SDT, ROS, RoboKudo, a camera model, object manipulation or mobility. The combined action-knowledge assembly now includes four domains and the registry contains 33 ontology documents.

## Five profiles

| Profile | Intended goal | Distinction preserved |
|---|---|---|
| Directing attention | Observer directed toward the target | Aiming is not visibility or detection |
| Visibility assessment | Visibility becomes known within the query scope | Occluded is a valid known assessment; existence/detection is not implied |
| Detection | At least one matching candidate is reported | Multiple detections need not resolve identity |
| Identification | One candidate satisfies the target-match contract, with no unresolved alternatives | No first-candidate or implicit latest-result choice |
| Bounded search | A matching candidate is found using ordered views and an explicit budget | Exhaustion is limited to the declared views/budget, not global absence |

Each profile has an observer, target, region, reference frame, evaluation context, scoped observation query, abstract attention/observation capability requirements, formal conditions, phases and monitoring requirements. Targets and observers are entity specifications, not physical world instances. The visual modality is used in the five synthetic examples; the shared modality vocabulary also permits other modalities.

A query binds its target, region, frame and modality. These must agree with the action description. Observation readiness and capability availability are explicit preconditions, not inferred from a declared requirement. Orienting attention may require robot motion, sensor configuration or selection of an observation region; no actuation technique is universally asserted by this module. Search viewpoints likewise describe observation requirements, not automatically executed navigation.

## Results and candidate hypotheses

`PerceptionResultRecord` specializes `kc:ObservationEvidence`. It records the selected knowledge record, context, answered query, result frame, observation timestamp, source identifier and operation outcome. Candidate hypotheses are information objects with an entity specification and a target-match classification. They are not grounded physical bodies or accepted identities.

Operation outcome and action goal are separate. A completed query can report no matching candidate; a completed visibility assessment can report occlusion. An acquisition/analysis failure is not evidence that the requested object does not exist. Unknown matches prevent premature identification. One known match plus an unresolved alternative remains unresolved; multiple known matches are ambiguous for identification.

Match assessments must mean satisfaction of the selected target's full requested constraints. A class-label match alone must not be supplied as a completed identity match when the request requires individual identity or additional attributes. This reference inspector does not perform feature extraction, class recognition, identity tracking or target-constraint matching itself.

Candidate hypotheses belong to one result record. Entity specifications may be reused deliberately, but shared mutable candidate-match assessments are rejected. Duplicate candidate entity descriptions within one result require explicit consolidation outside this inspector.

## Estimated poses and belief proposals

Optional `PoseEstimate` and `OrientationEstimate` structures describe estimated positions and quaternion XYZW components. They are distinct from robot-motion **desired pose targets**. An estimate must use the result/query frame; no transformation is silently inferred. The reference inspector requires finite metre coordinates for a supplied position estimate and finite dimensionless quaternion components when orientation is supplied. Negative coordinates are permitted. Quaternion normalization, covariance and uncertainty validation remain external work.

A detection can satisfy the current candidate-reporting goal without a pose estimate. Actions needing an object pose must impose and check that additional output requirement through a later task-specific acceptance contract. No pose, orientation or confidence is invented when it is missing.

A `BeliefUpdateProposal` can link an exact core claim to its source result, record and context. Scope checks require the proposed claim to belong to the record and the result to use the same record/context. This only validates the proposal's structure. There is no acceptance, materialization or world-model write operation. Proposed claims do not become accepted beliefs merely because a detector or LLM produced them.

## Freshness and source interpretation

Run SHACL and domain-integrity validation before reference inspection. The inspector is a bounded consumer, not a complete RDF validator. `tools/perception_attention.py` requires explicit record selection, RDF result identifiers, evaluation time and maximum observation age. It checks exact record/context/query/frame agreement, timezone-bearing timestamps, stale/future evidence and result/candidate structure. Age equal to the supplied maximum is fresh; no machine clock or default freshness threshold is read.

Observational, simulation and explicitly synthetic example evidence are supported by the bounded inspector. Synthetic input is labelled in its output. LLM-generation and document-source evidence do not count as acquired perception results under this initial policy. A malformed result raises an error; stale, future or unsupported evidence leaves the goal unresolved. This is a bounded engineering policy, not a general trust or evidential-quality model.

Multiple selected fresh results for a non-search query require explicit selection policy. Results are not silently sorted to choose the latest. Visibility or attention can change between timestamps, so such multiplicity is reported as requiring a policy rather than being declared a logical contradiction.

Freshness and scope validation do not establish measurement correctness. Provenance acceptance, calibration, confidence, model validity and synchronization of sensor/pose timestamps remain external requirements. `acceptedBeliefs` is always zero; `worldUpdated` and `executed` are always False. Goal truth is conditional on supplied result records and never proves an actual robot action occurred.

## Bounded search

A SearchSpecification contains indexed observation-view occurrences and an explicit dimensionless positive-integer attempt budget. Views bind viewpoint descriptions, regions and frames. The current strict profile searches the query's declared region/frame, with contiguous unique indices and one owning search per indexed occurrence.

The fixture budget of two is a labelled hypothetical example value, not an ontology default. A real task must supply its own budget. Unknown budgets cannot be replaced with an unlimited retry policy.

The inspector accepts a contiguous prefix of view results within the budget. Duplicate results for one view require selection; foreign views and out-of-order evidence do not become accepted attempts. A matching candidate permits stopping early. Results supplied after an earlier stopping condition require policy review. Unknown matches do not become unsuccessful attempts or absence claims.

Exhausting all declared views is distinguished from reaching the attempt budget while views remain. Both outcomes are local to the supplied search scope. They do not establish that the object is absent from the world, or even that the visible space was exhaustively observed. Search-space coverage, movement, stopping/continuation policies and identity association across views remain separate work.

## Conditions, phases and monitoring

Profiles use Prepare, Observe, Complete and Abort phases with guards for preconditions, goal satisfaction and explicit perception faults. Search additionally has an explicit local-exhaustion branch. A false detection goal with no technical fault does not automatically become a sensor failure. The generic detection phase skeleton may remain unresolved until a task-specific no-match or timeout policy is supplied; it is not a terminating executable plan.

Transition monitors enumerate required relation observations and request evidence when values are unknown. The inspector returns structured result classifications; mapping those records into accepted monitor facts is a future integration step. It does not directly fire transitions, manufacture observations, move a robot or publish beliefs. The new shared `condition_atoms.py` helper only consolidates the existing bounded condition-signature parser; object-manipulation semantics are preserved.

## Profiles, claims and tests

`examples/perception-attention-profiles.ttl` contains five hypothetical specification records. All included claims remain Proposed, with explicitly synthetic evidence. Every reachable descriptive non-type/non-label link has an exact statement. No perception-result records, candidate detections, images or accepted world observations are inserted in the example profiles. Result/pose/proposal fixtures are created only inside the validation process.

Run with requirements installed from the package root:

```bash
python tools/validate_perception_attention.py
python tools/reason_perception_attention.py
python tools/validate_imports.py
python tools/validate_object_manipulation.py
python tools/validate_mobility_navigation.py
python tools/review_competency.py
```

Java is needed for classification and consistency. Reasoning reports separate standalone/combined generic checks and optional AICOR consistency. Earlier AICOR loading instability must not be treated as repaired solely because one current consistency run passes. Full AICOR class satisfiability remains unverified.

Scoped queries: `queries/perception-requests.rq`, `queries/perception-search-views.rq`, and `queries/perception-results.rq`. Bind an explicit record, for example:

`http://aicor.knowledge/ontologies/perception-examples.owl#searchRecord`

Querying the specification-only examples for actual results correctly returns no rows. SPARQL retrieval does not itself apply freshness, evidence or acceptance policies.

## Remaining work

Source-backed sensing/visibility/attention knowledge, quantitative uncertainty models, task-specific required output fields, multimodal fusion, tracking, scene-to-world bindings, geometric visibility models, image/model adapters, accepted monitor facts and persistent belief-publication policy remain unfinished. Later action recipes can compose looking/detection/search with mobility and manipulation without coupling this domain to either implementation. Working projections and CoraPlex/RoboKudo adapters remain deferred.

## Release verification

94 new checks, 15 scoped query cases and 53 import/catalog checks passed. The earlier 102 object-manipulation, 104 mobility and 93 competency assertions passed as regressions. Standalone and combined assembly classification passed with no unsatisfiable named classes; generic examples consistency passed. The latest optional AICOR consistency check passed; the earlier normalization error remains unexplained. Of 66 prior RDF files, 65 are byte-identical; the combined assembly alone gains the new domain import.
