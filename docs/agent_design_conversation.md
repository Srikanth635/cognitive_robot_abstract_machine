# Wiki-Driven Action Designator Agent — Design Conversation

A Q&A-format record of a brainstorming session covering:
1. Testing the existing LLM wiki as a knowledge source
2. Designing a serializable structural schema for `PickUpAction`
3. Framing the wiki as a TBox and scene graph as an ABox
4. Designing a Claude-Agent-SDK-based agent over the wiki
5. Tracing a natural-language instruction through the full grounding pipeline
6. Replacing EQL-based grounding with LLM spatial/geometric/physics reasoning
7. Scaling the design to absorb new data layers (FrameNet, Flanagan motor phases)

The wiki referenced throughout lives at `docs/llm-wiki/` and was generated on a different
branch (commit `0528d8cf3`).

---

## Q1 — Wiki test: what are the grasp constraints / params for the pick-up action?

**Constraint:** answer using wiki content only, no source code. Be explicit about the
navigation path used.

### Navigation path (per CLAUDE.md §12)

```
CLAUDE.md (protocol)
  → index-pycram.md          (found: pycram.robot_plans.actions.core.PickUpAction)
  → wiki/entities/pycram.robot_plans.actions.core.PickUpAction.md   [starting page]
      └─ uses: pycram.datastructures.grasp.GraspDescription          [followed]
  → wiki/entities/pycram.datastructures.grasp.GraspDescription.md
      └─ used_by: pycram.robot_plans.motions.gripper                  [followed]
  → wiki/entities/pycram.robot_plans.motions.gripper.md
```

### Answer

**`PickUpAction` caller-visible fields**

| Field | Type | Role |
|---|---|---|
| `object_designator` | `Body` | SDT `Body` to pick up |
| `arm` | `Arms` | Which arm to use |
| `grasp_description` | `GraspDescription` | All grasp geometry |

**`GraspDescription` — where all grasp params live**

| Attribute | Type | Meaning |
|---|---|---|
| `approach_direction` | `ApproachDirection` | FRONT / BACK / LEFT / RIGHT |
| `vertical_alignment` | `VerticalAlignment` | TOP / BOTTOM / NoAlignment |
| `manipulator` | `Manipulator` | SDT manipulator annotation — supplies `front_facing_orientation` |
| `rotate_gripper` | `bool` | 90° roll of gripper around its x-axis |
| `manipulation_offset` | `float` | Pre-grasp standoff and lift height in metres (default 0.05) |

The grasp orientation sent to the MSC is a product of four quaternion multiplications:

```
SIDE_ROTATIONS[approach_direction]
  × VERTICAL_ROTATIONS[vertical_alignment]
  × HORIZONTAL_ROTATIONS[rotate_gripper]
  × manipulator.front_facing_orientation
```

The `_pose_sequence` method generates three waypoints driving the TCP motions:

1. **Pre-pose** — standoff = `bounding_box_half_extent + manipulation_offset`
2. **Grasp pose** — object centre, oriented by `grasp_orientation()`
3. **Lift pose** — same x/y as grasp, translated +z in map frame by `manipulation_offset`

**Translation to giskardpy goals**

| Phase | Motion | Goal |
|---|---|---|
| Open | `MoveGripperMotion(OPEN, arm)` | `JointPositionList` to open joint states |
| Reach | `MoveToolCenterPointMotion` × 2 | `CartesianPose(root, tip, pre_pose)` then `CartesianPose(root, tip, grasp_pose)` |
| Close | `MoveGripperMotion(CLOSE, arm, allow_gripper_collision=True)` | `JointPositionList` to close joint states |
| Lift | `MoveToolCenterPointMotion(lift_pose, TRANSLATION)` | `CartesianPosition` (position only) |

Root link selection: `world.root` if `robot.full_body_controlled`, else `robot.root`.

**Pre-conditions (gate execution):**
- `GripperIsFree(arm)` EQL predicate
- `pose_sequence_reachability_validator` — full MSC test on a deepcopy of the world

**Open question in the wiki:** `PickUpAction.execute` calls
`grasp_description.grasp_pose_sequence(body)` while `ReachAction` calls
`grasp_description._pose_sequence(target_pose, body, reverse)` — different signatures;
public vs private distinction unresolved.

---

## Q2 — Can we have a comprehensive data-structure representation of the pick-up action with all its motions, params, giskardpy constraints, and related info?

### Clarifying questions asked and answered

| Question | Answer |
|---|---|
| Primary intended use? | **Serializable record** (JSON/YAML schema for replay / comparison) |
| Constraint depth? | **Full QP formulation** (H, g, lb/ub arrays) |

### Proposed schema (YAML, type-annotated, no concrete values)

```yaml
# ─── action ────────────────────────────────────────────────────────────────
action:
  id: str                            # "pycram.robot_plans.actions.core.PickUpAction"
  package: str                       # "pycram"
  designator_base: str               # "ActionDescription"

  fields:
    object_designator:
      type: "sdt.world_description.world_entity.Body"
    arm:
      type: "Arms"
    grasp_description:
      type: "GraspDescription"

  preconditions:
    - predicate: "GripperIsFree"
      args: ["arm"]
    - predicate: "pose_sequence_reachability_validator"
      args: ["grasp_description.pose_sequence(object_designator)", "world_deepcopy"]

  postconditions:
    combinator: OR
    - predicate: "GripperIsNotFree"
      args: ["arm"]
    - predicate: "is_body_in_gripper"
      args: ["object_designator", "arm"]
      threshold: float               # > 0.9

# ─── grasp_description ─────────────────────────────────────────────────────
grasp_description:
  class: "pycram.datastructures.grasp.GraspDescription"
  fields:
    approach_direction: { type: "ApproachDirection" }   # FRONT|BACK|LEFT|RIGHT
    vertical_alignment: { type: "VerticalAlignment" }   # TOP|BOTTOM|NoAlignment
    manipulator:        { type: "Manipulator" }
    rotate_gripper:     { type: bool }
    manipulation_offset:{ type: float, unit: metres, default: 0.05 }

  derived:
    grasp_orientation:
      type: "Quaternion"
      formula: >
        SIDE_ROTATIONS[approach_direction]
        × VERTICAL_ROTATIONS[vertical_alignment]
        × HORIZONTAL_ROTATIONS[rotate_gripper]
        × manipulator.front_facing_orientation
    pose_sequence:
      pre_pose:   { frame: "object frame", offset: "manipulation_axis × (bbox_half + manipulation_offset)" }
      grasp_pose: { frame: "object frame", orientation: "grasp_orientation()" }
      lift_pose:  { frame: "map frame",    delta_z: "manipulation_offset" }

# ─── execution phases ──────────────────────────────────────────────────────
phases:
  - phase_id: 1
    combinator: sequential
    steps:
      - { kind: motion, motion_class: MoveGripperMotion,            goal: JointPositionList }
      - { kind: motion, motion_class: MoveToolCenterPointMotion,    goal: CartesianPose      }   # pre_pose
      - { kind: motion, motion_class: MoveToolCenterPointMotion,    goal: CartesianPose      }   # grasp_pose
      - { kind: motion, motion_class: MoveGripperMotion,            goal: JointPositionList }
      - { kind: world_mutation, method: "World.move_branch_with_fixed_connection", atomicity: true }

  - phase_id: 2
    combinator: single
    steps:
      - { kind: motion, motion_class: MoveToolCenterPointMotion,    goal: CartesianPosition  }   # lift_pose (TRANSLATION)

# ─── QP configuration & problem form ───────────────────────────────────────
qp_controller_config:
  target_frequency:   { type: float, default: 50 }
  prediction_horizon: { type: int,   default: 7  }
  max_derivative:     { type: "Derivatives", default: jerk }
  dof_weights:        { type: "Dict[str,float]", default: 0.01 }
  retries_with_relaxed_constraints: { type: int, default: 5 }
  qp_solver_class:    { default: QPSolverPIQP }

qp_problem_form:    # per tick, after apply_filters
  objective:
    H: "np.ndarray — diag(quadratic_weights)"
    g: "np.ndarray — linear_weights"
  constraints:
    box:        { lower: np.ndarray, upper: np.ndarray }
    equality:   { E: sp.csc_matrix,  b_E:  np.ndarray }
    inequality: { A: sp.csc_matrix,  lb_A: np.ndarray, ub_A: np.ndarray }
  filter_rule: "Slack vars with zero quadratic_weight pruned; DOF vars never pruned"
```

Each constraint within a goal's `ConstraintCollection` is one of:
`EqualityConstraint`, `InequalityConstraint`, `DerivativeEqualityConstraint`, or
`DerivativeInequalityConstraint`, with lifecycle-gated `quadratic_weight =
base_weight × if_eq(lifecycle_var, RUNNING, 1, 0)`.

---

## Q3 — Can the wiki act as a TBox, with concrete instances generated as an ABox from a scene graph + instruction?

### The mapping

| Description-Logic element | Already present in the codebase |
|---|---|
| Concept definitions | Wiki entity pages (`fields` tables) |
| Role hierarchy | `uses` / `used_by` frontmatter graph |
| Subsumption | `designator_base` chain |
| Axioms / restrictions | `preconditions` / `postconditions` |
| Derived properties | `grasp_description.derived` (formulas) |
| ABox individuals | SDT `World` — every `Body`, `Connection`, `SemanticAnnotation` |
| Role assertions | SDT kinematic edges; semantic predicates |
| Reasoning over ABox | `WorldReasoner` + `@symbolic_function` predicates |
| TBox→ABox grounding mechanism | `krrood.entity_query_language` (EQL) |

### Full conceptual pipeline

```
Instruction "pick up the red cup"
         ↓
  [concept selection]   wiki TBox → PickUpAction
         ↓
  [partial grounding]   object_designator ← ABox query: Body with semantic.colour=red
         ↓
  [EQL grounding]       arm, grasp_description ← precondition satisfaction
                        GraspDescription.calculate_grasp_descriptions(...) → ranked list
         ↓
  [axiom checking]      pose_sequence_reachability_validator prunes invalid candidates
         ↓
  [fully grounded ABox instance]
        PickUpAction(object_designator=cup_42, arm=Arms.RIGHT,
                     grasp_description=GraspDescription(FRONT, NoAlignment, ...))
         ↓
  [runtime expansion]   TBox phase schema → QP constraints
```

### Gaps that prevent automated grounding today

1. Concept definitions are markdown tables, not machine-readable schemas.
2. Preconditions are prose, not formal EQL expressions stored in the TBox.
3. Derived-property formulas live as prose, not callable rules.
4. The two-phase execution schema has no formal representation outside `execute()`.

The wiki is the closest thing to a formal TBox; the design choice is whether to make
its pages machine-parseable (structured frontmatter + formal predicates) or to generate
a separate ontology artifact from them.

---

## Q4 — Assuming a Claude-Agent-SDK agent over this wiki, what skills / subagents / tools / MCPs would let it answer different query levels with structured responses?

### Query taxonomy

| Tier | Query type | Example |
|---|---|---|
| L0 | Single concept / entity lookup | "What is a Designator?" |
| L1 | Structural — fields, params | "What params does PickUpAction take?" |
| L2 | Relational — graph navigation | "What depends on `sdt.world.World`?" |
| L3 | Compositional — multi-page synthesis | "Trace pick-up from instruction to QP" |
| L4 | Comparison / diff | "ActionNode vs MotionNode" |
| L5 | Grounding — TBox → ABox | "Given scene + instruction, produce concrete action" |
| L6 | Verification — wiki ↔ source | "Is `_pose_sequence` still as documented?" |
| L7 | Gap analysis | "What's still stub or contradictory?" |
| L8 | Maintenance | "Update the wiki for this PR" |
| L9 | Fall-through to source | (when wiki silent) |

### Tools (atomic primitives, no LLM reasoning inside)

| Tool | Purpose |
|---|---|
| `wiki_read_page(id)` | Resolve ID → file path per §4, return frontmatter + body |
| `wiki_resolve_link(id)` | Return one of 4 candidate paths or "missing" |
| `wiki_neighbors(id, direction, depth)` | Graph-slice of `uses` / `used_by` |
| `wiki_subgraph(root_id, depth, kind_filter)` | BFS dump as JSON |
| `wiki_path_between(a, b)` | Shortest dependency path |
| `wiki_frontmatter_query(predicate)` | Structured filter over YAML |
| `wiki_provenance_check(id)` | Verify cited commit SHAs against `git rev-list` |
| `wiki_open_questions_scan()` | Aggregate all Open-Questions sections vault-wide |
| `wiki_stub_create(id, used_by, tags)` | Emit minimal §13 stub + update index |
| `wiki_log_append(entry)` | Append-only log write |

### Skills (markdown recipes; orchestrate tools)

| Skill | Codifies |
|---|---|
| `wiki-navigate` | §12 lookup recipe — default L0–L4 entry point |
| `wiki-ingest` | §6 workflow — drives L8 |
| `wiki-lint` | §9 checks — drives L7 |
| `wiki-stub-policy` | §13 invariants |
| `tbox-instantiate` | TBox concept → field enumeration → ABox query → preconditions → grounded instance |
| `schema-extract` | Walk entity + `uses` closure → full structural schema |
| `entity-compare` | Side-by-side structured diff |
| `wiki-vs-source` | Cited source @ pinned commit + drift report |
| `pre-pr-wiki-update` | Triggers ingest whenever code change touches cited paths |

### Subagents (isolated context, specialized roles)

| Subagent | Role |
|---|---|
| **wiki-explorer** | Read-only multi-page navigation; returns relevant pages + excerpts |
| **wiki-ingester** | PR diff / new module → full changeset (pages, links, stubs, log entry) |
| **wiki-linter** | §9 battery vault-wide, parallel per-rule |
| **schema-extractor** | Entity ID → deep structural schema (action → motions → constraints → QP) |
| **grounding-agent** | Instruction + scene graph → ABox instance + validation |
| **code-corroborator** | Wiki claim + cited commit → source read + drift report |
| **gap-analyzer** | `log.md` + Open Questions → ranked by in-degree |
| **bridge-tracer** | Cross-package coupling-point analysis |

### MCPs (external systems — ABox & ground truth)

| MCP | Tools exposed |
|---|---|
| **scene-graph** | `body_list`, `connection_tree`, `semantic_annotations_for`, `pose_of`, `gripper_state(arm)`, `eql_evaluate(expr)` |
| **source-at-sha** | `read_at_commit(path, sha, lines)`, `git_blame`, `commit_diff` |
| **pr-events** | `list_prs`, `pr_diff`, `pr_files_changed` |
| **eql-evaluator** | `evaluate(expr, scene)` |
| **sim-runner** | `run_plan(grounded_action, scene)`, `last_execution_data` |
| **graph-viz** | `mermaid_for_subgraph`, `excalidraw_for_bridge` |
| **wiki-search-index** | Vector / BM25 over body text |

### Bootstrap priority

1. `wiki_read_page` + `wiki_resolve_link` + `wiki_neighbors` tools — every other component
   sits on these.
2. `wiki-navigate` skill — encodes §12 once.
3. `wiki-explorer` subagent — context isolation pays off the moment a query needs >3 reads.
4. `source-at-sha` MCP — closes the L6 verification loop; without it the wiki's claims
   silently drift.

### Design principle

Markdown is the source of truth. Tools parse YAML frontmatter + markdown body
on-the-fly into structured projections. **No machine-readable sidecar files** — they
will drift.

---

## Q5 — Walk through "pick up the cup from the table": full pipeline, sample outputs at each stage.

(Assumptions: scene-graph MCP live with a tabletop scene; Tiago robot,
`full_body_controlled=True`; wiki fully ingested.)

### Stage 1 — Intent parsing → concept selection
**Component:** `wiki-navigate` skill

```json
{
  "intent": "manipulation.pick_up",
  "tbox_concept": "pycram.robot_plans.actions.core.PickUpAction",
  "object_hint": "cup",
  "location_hint": "on table",
  "free_variables": ["object_designator", "arm", "grasp_description"]
}
```

### Stage 2 — Scene graph ABox query
**Component:** `scene-graph MCP`

```json
{
  "candidates": [{
    "id": "mug_42",
    "semantic_annotations": ["Container", "IsPerceivable", "HasRootBody"],
    "pose": {"position": [1.2, 0.4, 0.82], "orientation": [0,0,0,1]},
    "bounding_box": {"x": 0.08, "y": 0.08, "z": 0.12},
    "parent_connection": "rests_on → kitchen_table_7"
  }],
  "resolved_object_designator": "mug_42"
}
```

### Stage 3 — TBox loading
**Component:** `wiki-explorer` subagent, `schema-extract` skill — produces structured
TBox schema (fields, preconditions, postconditions, grasp_description fields).

### Stage 4 — Variable grounding (EQL version — see Q6 for LLM-reasoning version)
**Component:** `grounding-agent` + `eql-evaluator MCP`

```python
variable(Arms, domain=[Arms.LEFT, Arms.RIGHT])
  .where(GripperIsFree(arm))
  .where(reachable(arm, mug_42))
```

Result: `arm = Arms.RIGHT` (smaller angular offset).

### Stage 5 — GraspDescription grounding
**Component:** `grounding-agent` + `sim-runner MCP`

```python
candidates = GraspDescription.calculate_grasp_descriptions(
    manipulator=tiago.right_arm.manipulator,
    object_pose=mug_42.pose
)
# ranked list; sim-runner validates rank-1 candidate via pose_sequence_reachability_validator
```

Result: `GraspDescription(FRONT, NoAlignment, right_manipulator, False, 0.05)`.

### Stage 6 — Final grounded ABox instance

```python
PickUpAction(
    object_designator = Body(id="mug_42"),
    arm = Arms.RIGHT,
    grasp_description = GraspDescription(
        approach_direction  = ApproachDirection.FRONT,
        vertical_alignment  = VerticalAlignment.NoAlignment,
        manipulator         = tiago.right_arm.manipulator,
        rotate_gripper      = False,
        manipulation_offset = 0.05,
    )
)
```

### Stage 7 — Execution expansion (TBox phase schema drives execution)

6 motion steps → giskardpy goals → QP constraints → joint velocity commands.

### Failure handling

| Failure | Handling |
|---|---|
| No "cup" matched | scene-graph empty → agent asks user to clarify |
| Both grippers occupied | EQL no satisfying binding → PlanFailure pre-execution |
| All grasp candidates fail reachability | sim-runner FAIL → gap-analyzer flags as open question |
| Wiki silent on mid-pipeline concept | L9 fall-through + log `query` entry in `log.md` |
| Scene-graph snapshot stale | Provenance timestamp check → user warning |

---

## Q6 — Replace EQL-based grounding with LLM spatial / geometric / physics reasoning.

**Constraint:** the LLM itself chooses values for `arm`, approach direction, vertical
alignment, etc. by reasoning over scene geometry — not via EQL filtering.

### What changes

| Stage | Previous (EQL) | Revised (LLM) |
|---|---|---|
| Arm selection | EQL: `GripperIsFree ∩ reachable` | LLM spatial reasoning |
| Grasp params | `calculate_grasp_descriptions` → EQL filter | LLM geometric + physics reasoning |
| eql-evaluator MCP | Drives grounding | Removed from grounding path |
| sim-runner MCP | Optional ranking helper | Promoted to **hard validation gate** |

### New stage — Spatial Context Preparation

A `spatial-context` tool transforms raw scene-graph data into LLM-readable form:

```
ROBOT FRAME CONTEXT:
- Target mug_42 is 1.2m in front (+x), 0.4m to the LEFT (+y), at 0.82m height
- Right arm shoulder at y=-0.2 → 0.6m lateral from right shoulder (across body)
- Left  arm shoulder at y=+0.2 → 0.2m lateral from left shoulder (natural reach)
- Reach magnitudes:  right 1.36m (over limit), left 1.24m (within reach)

APPROACH GEOMETRY:
- FRONT (-x): 1.2m clear corridor
- LEFT  (-y): 0.3m before kitchen wall — tight
- BACK / RIGHT: occluded by robot body

OBJECT GEOMETRY:
- Cylindrical vessel, 8cm × 12cm, ~0.3kg
- No fragility signals; stable base
```

### LLM reasoning chain (Stage 4 revised)

```
FIELD: arm
  Domain: [Arms.LEFT, Arms.RIGHT]
  Analysis:
    - mug 0.4m left of centerline; left shoulder 0.2m further left = natural alignment
    - right shoulder requires 0.6m lateral extension (crosses body midline)
    - both reachable after mobile-base repositioning
  → arm = Arms.LEFT

FIELD: approach_direction
  - FRONT: 1.2m clear corridor
  - LEFT: only 0.3m wall clearance — insufficient retreat path
  - BACK: would require passing object
  → approach_direction = FRONT

FIELD: vertical_alignment
  - Cylinder, no asymmetry → side grasp maximises contact patch
  - TOP risks rim slip
  → vertical_alignment = NoAlignment

FIELD: rotate_gripper
  - Symmetric cylinder, no preferred orientation
  → rotate_gripper = False

FIELD: manipulation_offset
  - 0.3 kg, 8 cm diameter; default standoff sufficient
  → manipulation_offset = 0.05
```

### Structured output

```json
{
  "arm": {
    "value": "Arms.LEFT",
    "reasoning": "0.2m lateral deviation vs 0.6m for right; mobile base compensates"
  },
  "grasp_description": {
    "approach_direction":  {"value": "FRONT",        "reasoning": "1.2m clear corridor"},
    "vertical_alignment":  {"value": "NoAlignment",  "reasoning": "Cylinder → side grasp"},
    "rotate_gripper":      {"value": false,          "reasoning": "Symmetric"},
    "manipulation_offset": {"value": 0.05,           "reasoning": "Default; light load"}
  },
  "confidence": "high",
  "flagged_risks": ["left arm near max reach — base reposition required"]
}
```

### Validation gate (sim-runner)

`sim-runner MCP` now runs as a hard binary check after the LLM has selected.
On failure, the failure reason is returned to the grounding-agent which **re-reasons**
— it does not enumerate the next EQL binding.

```
Sim-runner: FAIL — "left arm cannot reach grasp_pose even after base repositioning"
LLM: "Reach estimate was off. Right arm — despite larger lateral deviation — gives a
       straighter elbow configuration. Retry with Arms.RIGHT."
```

### Architectural consequence

Reasoning and validation are now **separated**:

- **LLM reasons qualitatively** — spatial layout, geometric affordances, physics
  intuition, trade-offs between valid options. (LLMs are good at this; EQL isn't.)
- **sim-runner validates quantitatively** — IK feasibility, collision, exact reach.
  (Simulators are good at this; LLMs aren't.)

EQL stays in the system but its job shifts to **post-execution assertion checking**
(did `GripperIsNotFree`, `is_body_in_gripper > 0.9` hold after execution?) rather than
pre-execution grounding.

The `spatial-context` tool is now the highest-leverage new component: how you frame
the scene geometry for the LLM determines reasoning quality more than anything else.

---

## Q7 — How does the design scale over new data like FrameNet representations and Flanagan motion phases? Where do such things fit, and how do they complement the design to produce enriched action designators alongside executable pycram ones?

### Architectural placement

FrameNet and Flanagan **bracket the existing design** from opposite ends:

```
┌─────────────────────────────────────────────────────┐
│  NL: "pick up the cup gently"                        │
│  FrameNet — semantic frame / Frame Elements          │  ← ABOVE
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  TBox (wiki) + LLM grounding                         │
│  PickUpAction(object_designator, arm, grasp_descr.)  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Flanagan — motor phases & profiles                  │  ← BELOW
│  Pre-shape → Transport → Decel → Contact → Load …    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  giskardpy QP / joint trajectories                   │
└─────────────────────────────────────────────────────┘
```

### FrameNet — contribution and placement

**What it contributes:** semantic Frame Elements bridge NL to TBox + carry parameter
modifiers pycram has no field for.

| FrameNet FE | Maps to pycram | Additional semantics |
|---|---|---|
| `Theme` | `object_designator` | — |
| `Body_part` / `Instrument` | `arm` | — |
| `Source` | navigation precondition | spatial constraint, not in pycram |
| `Goal` | implicit postcondition | clarifies intent for failure recovery |
| `Manner` ("gently") | none directly | → modifies `manipulation_offset`, QP `dof_weights`, velocity limits |
| `Degree` ("fully", "quickly") | none directly | → `target_frequency`, `prediction_horizon` |

**Where it lives:**
- New wiki page: `concepts/concept.framenet-annotation.md`
- New section on action entity pages: `framenet_binding` (frame, FE map, parameter modifiers)
- New skill: `framenet-ground`
- New MCP: `framenet-data` (your generated dataset)

**How it enriches grounding:**
The instruction-parsing stage gains a FrameNet parse before TBox lookup; FE values flow
into both pycram field assignments and downstream parameter modifiers ("gently" →
manipulation_offset ×1.6, dof_weights ×0.5).

### Flanagan motor phases — contribution and placement

**What they contribute:** motor-control decomposition richer than pycram's binary
OPEN/CLOSE and waypoint-only motions.

| Flanagan phase | Current pycram | What Flanagan adds |
|---|---|---|
| Pre-shape | `MoveGripperMotion(OPEN)` (binary) | Aperture tuned to object bbox |
| Transport accel | `MoveToolCenterPointMotion(pre_pose)` | Trapezoidal velocity profile |
| Transport decel | (same step, no profile) | Velocity ramp-down near contact |
| Contact | end of grasp_pose motion | Contact detection event |
| Load | `MoveGripperMotion(CLOSE)` (binary) | Force threshold loop |
| Lift | `MoveToolCenterPointMotion(lift_pose)` | dof_weight scaling by payload |

Critically: transport and pre-shape are **parallel** in real grasping but **sequential**
in current pycram.

**Where they live:**
- New wiki page: `concepts/concept.flanagan-phases.md`
- New section on action entity pages: `flanagan_binding` (phase map → QP modifications)
- New skill: `flanagan-enrich`
- New MCP: `flanagan-data`

**How they enrich execution:**
Each Flanagan phase emits structured QP modifications — aperture formulas, velocity
profile envelopes, force thresholds, payload-scaled dof_weights — layered on top of the
base pycram MSC.

### The enriched ABox instance

```python
EnrichedPickUpAction(

    pycram = PickUpAction(
        object_designator = Body(id="mug_42"),
        arm               = Arms.LEFT,
        grasp_description = GraspDescription(
            approach_direction  = ApproachDirection.FRONT,
            vertical_alignment  = VerticalAlignment.NoAlignment,
            rotate_gripper      = False,
            manipulation_offset = 0.08,     # overridden by Manner="gently"
        )
    ),

    framenet = FrameAnnotation(
        frame = "Cause_motion",
        lexical_unit = "pick up.v",
        elements = {
            "Agent":     "tiago_robot",
            "Theme":     "mug_42",
            "Source":    "kitchen_table_7",
            "Manner":    "gently",
            "Body_part": "Arms.LEFT",
        },
        parameter_modifiers = {
            "manipulation_offset": 0.08,    # 0.05 × 1.6 for "gently"
            "dof_weights_scale":   0.5,
        }
    ),

    flanagan = MotorSchema(
        phases = [
            FlanaganPhase(name="pre_shape",  parallel="transport",
                          aperture=0.106, qp_mod={"JointPositionList.goal_state": "computed_aperture"}),
            FlanaganPhase(name="transport",  profile="trapezoidal", peak_at=0.6,
                          qp_mod={"velocity_upper_limit": "high", "dof_weights": 0.005}),
            FlanaganPhase(name="decel",      trigger="dist_to_grasp < 0.05",
                          qp_mod={"velocity_upper_limit": "low"}),
            FlanaganPhase(name="load",       trigger="contact_detected",
                          qp_mod={"compliance_weight": 0.3}),
            FlanaganPhase(name="lift",
                          qp_mod={"dof_weights_scale": 1.0 + 0.3 / robot_payload}),
        ]
    )
)
```

### Interaction flow

```
Instruction + Manner FEs
        │
        ├──[FrameNet]──→ parameter_modifiers → override pycram defaults
        │                (manipulation_offset, dof_weights, velocity limits)
        │
        ▼
LLM grounding (spatial + geometric reasoning)
        │
        ├──[Flanagan]──→ phase decomposition → QP modifications per phase
        │                (aperture, velocity profiles, force thresholds)
        │
        ▼
Enriched ABox instance
        │
        ├── pycram runtime uses pycram layer directly (backward compatible)
        ├── QP controller reads Flanagan modifications on top
        └── Explanation queries answered from FrameNet layer
```

### Spectrum of execution fidelity

| Configuration | Behaviour |
|---|---|
| pycram only | Current behaviour, fully executable |
| + FrameNet | Better NL grounding, Manner handling |
| + Flanagan | Smoother velocity profiles, tuned apertures |
| All three | Richest action designator |

### Components added

| Component | Kind |
|---|---|
| `concept.framenet-annotation` | Wiki page (TBox) |
| `concept.flanagan-phases` | Wiki page (TBox) |
| `framenet-ground` | Skill |
| `flanagan-enrich` | Skill |
| `framenet-parser` | Tool |
| `flanagan-schema-lookup(action_id)` | Tool |
| `framenet-data MCP` | MCP |
| `flanagan-data MCP` | MCP |
| `enriched-instance-assembler` | Subagent |

### Scaling pattern

Each new data type follows the same four-step recipe:

1. **Declare in the wiki as a new TBox concept** — gives it an ID, schema, and
   `uses` / `used_by` links to existing pages.
2. **Expose it via an MCP** — data stays in its own store; the MCP makes it queryable.
3. **Add a skill** that reads the TBox + queries the MCP and emits structured
   modifications.
4. **The enriched ABox instance** gains a new layer without changing existing layers.

A third dataset — affordance annotations, force/torque demonstrations, manipulation
ontologies — would follow the same pattern. The architecture doesn't change; it grows
new leaves.

---

## Summary of the overall design

- **The wiki is a TBox** — concept definitions, role hierarchy, axioms, derived rules.
- **The SDT scene graph is the ABox** — concrete individuals and role assertions.
- **EQL is the formal grounding & assertion checker** — used post-execution, not for
  reasoning over qualitative trade-offs.
- **The LLM is the qualitative reasoner** — spatial, geometric, physics intuition.
- **The simulator is the quantitative validator** — hard feasibility checks.
- **FrameNet sits above** — semantic bridge from NL to TBox concept + parameter modifiers.
- **Flanagan sits below** — motor-control enrichment of execution.
- **Tools, skills, subagents, MCPs** are layered: tools are atomic, skills orchestrate,
  subagents isolate context, MCPs reach external systems.
- **Markdown remains the source of truth** — structured projections are derived on the fly.

The architecture is additive: pycram-only execution still works; each enrichment layer
augments without replacing.
