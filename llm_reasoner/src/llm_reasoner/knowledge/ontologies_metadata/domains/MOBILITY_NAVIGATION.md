# Mobility and navigation — 0.15.0

> Updated in 0.22.0: see [PRIMARY_ACTION_GENERATION_GUIDE.md](PRIMARY_ACTION_GENERATION_GUIDE.md) and `reports/release-0.22.0.json`. The dated development history and counts below retain their original scope.

The third reusable domain is **mobility-navigation.ttl**:

`http://aicor.knowledge/ontologies/mobility-navigation.owl`

It imports robot motion, phases/transitions, monitoring requirements and provenance/context. DUL is loaded transitively through the action core. It does not import object manipulation, CoraPlex, SDT, ROS, a robot model, actual map or simulator. The combined `all-action-knowledge.ttl` assembly imports all three implemented domains. There are now 32 registered ontology documents.

## Five families

| Family | Intended success in the current profile |
|---|---|
| Navigation | Robot in the intended region, with position and heading errors within specified tolerances |
| Route navigation | Navigation success plus route compliance over a bound evaluation interval |
| Facing | Heading error within tolerance and maximum translation drift within tolerance over a bound interval |
| Inter-level navigation | Navigation success plus explicit presence on the intended destination level |
| Elevator navigation | Inter-level success plus explicit exit from the cabin, with guarded entry/ride/exit stages |

The concrete classes specialize a shared `MobilityDescription`. The reference checker accepts the declared inheritance chain but rejects mixtures of distinct concrete profiles unless an explicit composite contract is added; it does not silently ignore route requirements on an elevator task. Action kinds remain individuals of `ac:ActionKind`, consistent with the core's DUL Task pattern. Every profile refers to a robot entity specification, target region, reference-frame specification, footprint specification, evaluation context, abstract locomotion/observation capability requirements, preconditions, goals, success and fault criteria.

These are descriptions of possible actions and requirements. They do not instantiate a physical robot, room, floor or elevator. Region, level, cabin, door and facility specifications extend the core EntitySpecification and therefore remain descriptive knowledge.

## Targets, frames and metrics

Targets use robot-motion pose specifications. Navigation examples include unspecified position and quaternion XYZW parameters. The facing profile describes bearing toward a target region; its heading error must be supplied relative to that bearing, not relative to the target region's own orientation.

Position errors, heading errors, tolerances and required clearance are scalar parameter specifications. Their units are explicitly declared as metres or radians where applicable; no numerical values or default safety thresholds are asserted. The footprint is an abstract description requiring externally bound geometry, not a computed bounding shape. Preconditions explicitly qualify navigability and route traversability by robot and footprint.

Reference inspection consumes nonnegative error metrics and tolerances supplied by the caller. It does not compute distance, heading, drift, collision clearance, inverse kinematics or physical feasibility. Quaternion normalization and frame transforms are not implemented. Different frames are rejected by the bounded authoring checks rather than silently transformed.

## Routes and map provenance

`RouteSpecification` describes source/destination regions, a reference frame, ordered segment occurrences and a versioned map-source snapshot. Each segment has a contiguous positive index and explicit from/to region specifications. The integrity checker verifies segment ownership, ordering, adjacency, endpoints, destination binding and frame agreement. Repeated regions are allowed; disconnected segment chains are rejected. Ordering does not establish traversability.

The example route has two hypothetical segments. Its source snapshot uses an explicitly synthetic locator/version/selector, not a downloaded or validated map. Real map content and versions must be injected through context and their evidence accepted separately. The `RouteTraversableBy(route, robot, footprint)` predicate is externally assessed; this module supplies no planner, robot footprint calculator or map-validity inference.

Reaching the endpoint alone does not demonstrate that the requested route was followed. `RouteFollowed` denotes externally assessed compliance over the supplied evaluation interval. The sampling coverage and allowable deviation defining that assessment are external contracts, not inferred by the ontology.

## Facing and interval evidence

Facing constrains the maximum translation drift over the action interval. A robot that moves away and returns to its starting position must not be represented as having zero maximum drift merely because endpoint displacement is zero. The reference evaluator consumes the externally calculated maximum-drift metric; it does not calculate it from a trajectory.

Both route navigation and facing require caller-supplied interval metadata for positive aggregate completion evidence:

```python
interval = {
    "context": "<the selected profile's context IRI>",
    "start": "2026-11-11T10:00:00Z",
    "end": "2026-11-11T10:01:00Z",
}
```

Context must match, timestamps must carry timezones and start must precede end. Without interval metadata, aggregate route compliance or drift evidence is withheld and cannot establish success. This scopes the question; it does not independently verify monitoring coverage, freshness or evidence quality. Use the existing provenance/context layer for observation timestamps and assessments. The caller must isolate facts from different action contexts and ensure their time/frame interpretation agrees.

## Inter-level and elevator semantics

Source and destination levels are explicitly distinct in the inter-level profile. Completion checks the destination level in addition to pose/region arrival; similar XY coordinates on another floor cannot substitute for destination-level evidence.

The elevator example has Prepare, Wait, Enter, Ride, Exit, Complete and Abort phases. Semantic stage-role concepts identify the cabin stages independently of their concrete phase IRIs.

| Transition | Required current evidence |
|---|---|
| Wait → Enter | Source door open, cabin at source level, entry clear for this robot/footprint |
| Enter → Ride | Robot inside cabin and source door explicitly closed |
| Ride → Exit | Cabin at destination level, destination door open, exit clear |
| Exit → Complete | Final pose/region/level criteria and robot explicitly outside the cabin |

Missing DoorOpen evidence does not establish DoorClosed. Source/destination door descriptions may ultimately bind to the same cabin door; their distinct descriptive roles do not assert different physical doors. The cabin must belong to the requested facility. An abstract facility-service precondition qualifies the requested level pair.

The phase graph records intended conditions and explicit navigation-fault branches. It does not send elevator commands, validate door safety, establish termination, provide timeouts or select a transition. Even a reported true guard is not execution permission. Additional ride invariants, authorization, accessibility conditions, hardware interlocks and recovery policies must be supplied by a qualified facility/execution model before deployment. No hardware capability is claimed by these examples.

## Failure, monitoring and unknown information

Failure criteria use an explicit `NavigationFault` report, not the absence of success evidence. Transition monitors declare every required relation/quantity leaf and an unknown-evidence response. Missing observations remain Unknown; no default safe route, open door, available capability or successful arrival is invented.

`tools/mobility_navigation.py` separately reports precondition, success criterion and fault truth, interval binding and simultaneous success/fault reports. It always reports that execution/readiness were not assessed. A True success expression is a result about supplied inputs, not proof that a real navigation action occurred. Facts or metrics from an LLM or a synthetic example do not become accepted observations through this evaluator.

## Provenance and examples

`examples/mobility-navigation-profiles.ttl` contains five hypothetical records. Every reachable descriptive non-type/non-label edge has an exact statement and Proposed claim in its record, backed by evidence explicitly labelled synthetic. No numerical target/tolerance/clearance, actual map, sensor observation or accepted world-state assertion is generated.

Old domains, aspects and example graphs remain unchanged. The combined action-knowledge assembly gains one import; the independent generic aspect assembly remains unchanged. The aspect subfolder organization from 0.14.1 is retained.

## Competency questions and verification

The implementation tests distinguish endpoint arrival from route compliance; heading from bounded translation; arrival coordinates from destination-level presence; and cabin arrival from successful exit. Negative checks cover broken route connectivity/order, wrong frames, map-version omission, footprint/clearance binding, mismatched levels/context, wrong cabin/facility/door references, missing guard evidence, units, interval bounds and exact provenance.

Run from the package root with requirements installed:

```bash
python tools/validate_mobility_navigation.py
python tools/reason_mobility_navigation.py
python tools/validate_imports.py
python tools/validate_robot_motion.py
python tools/validate_object_manipulation.py
python tools/review_competency.py
```

Java is required for classification and consistency. Reasoning reports distinguish standalone/combined generic checks from the optional AICOR check. The current optional check passed in this run; the earlier SceneNode normalization failure has not yet been explained, so this does not establish that historical loading instability is fixed. Full AICOR satisfiability remains unverified. No prior reasoning result is represented as a fresh successful AICOR check.

Scoped SPARQL queries are `queries/mobility-targets.rq`, `queries/mobility-route.rq` and `queries/mobility-guards.rq`. Bind `?record` explicitly, for example:

`http://aicor.knowledge/ontologies/mobility-examples.owl#routeNavigationRecord`

## Remaining work

Populate source-backed route/locomotion knowledge; bind actual robots, maps and geometry; define traversability, route compliance, clearance and freshness acceptance policies; add detailed elevator invariants/recovery; and connect verified planning/localization/controllers through separate adapters. Carrying remains a later composition of mobility and object-holding requirements. Perception/attention and general articulated mechanisms remain separate planned domains. Working projections remain deferred.

## Release verification

104 mobility checks, 15 scoped query cases and 52 import/catalog checks passed. The earlier 58 robot-motion, 102 object-manipulation and 93 competency assertions passed as regressions. The standalone domain and combined assembly classify with no unsatisfiable named classes; generic combined examples consistency passed. Optional AICOR combined consistency passed in the latest runs, while the cause of the earlier normalization failure remains unexplained. Of 63 prior ontology/example files, 62 are byte-identical; the combined assembly alone changed to import this domain.
