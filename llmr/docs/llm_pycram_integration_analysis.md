# LLM → PyCRAM Integration Analysis

## 1. Pipeline Overview

```
User Instruction (natural language)
        │
        ▼
┌─────────────────────────────┐
│  ReflectiveParser           │  intent_entity.py
│  IntentType classification  │  → classifies to one of 10 intent types
└────────────┬────────────────┘
             │  IntentType enum value  (e.g. PickingUp, Opening)
             ▼
┌─────────────────────────────┐
│  Action Decomposition Graph │  enhanced_ad_graph.py
│  3-step LLM chain:          │
│  1. Field Extraction        │  → raw JSON attributes per action core
│  2. Semantic Enrichment     │  → JSON with _props (size, color, etc.)
│  3. CRAM Plan Generation    │  → LISP S-expression string
└────────────┬────────────────┘
             │  CRAM plan string
             │  e.g. (an action (type PickingUp)
             │         (object (:tag milk (an object (type Substance ...))))
             │         (source (a location (on (:tag countertop ...)))))
             ▼
┌─────────────────────────────┐
│  CRAMToPyCRAMSerializer     │  cram_to_pycram.py
│  - Parse S-expression       │
│  - Extract roles            │
│    (object, source, goal,   │
│     utensil, content …)     │
│  - Normalize action_type    │
│  - Lookup in _ACTION_MAP    │
│  - Build PartialDesignator  │
└────────────┬────────────────┘
             │  PyCRAM PartialDesignator
             ▼
┌─────────────────────────────┐
│  SimulationBridge           │  simulation_bridge.py
│  - Resolve CRAM tags        │
│    → live world Body objects│
│  - Auto-inject NavigateAction│
│    for placement actions    │
│  - Build grasp descriptions │
│  - Wrap in SequentialPlan   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  PyCRAM Execution           │  pycram/robot_plans/actions/
│  Core + Composite actions   │
│  → Robot simulation         │
└─────────────────────────────┘
```

---

## 2. Action Type Inventory

### 2.1 What the LLM Can Classify (IntentType Enum — 10 types)

| Enum Name | Value | Feeds Into Action Core |
|-----------|-------|------------------------|
| `PICK`  | `PickingUp`  | PickingUp |
| `PLACE` | `Placing`    | Placing   |
| `POUR`  | `Pouring`    | Pouring   |
| `CUT`   | `Cutting`    | Cutting   |
| `OPEN`  | `Opening`    | Opening   |
| `PULL`  | `Pulling`    | Pulling   |
| `STIR`  | `Stirring`   | Stirring  |
| `MIX`   | `Mixing`     | Mixing    |
| `HEAT`  | `Heating`    | *(no action core defined — see §3)* |
| `COOL`  | `Cooling`    | Cooling   |

### 2.2 What the CRAM Action Core Library Supports (43 action cores in JSON)

The JSON resource (`cram_action_cores.json`) and the Pydantic models (`cram_gen_models.py`) define 43 action cores. These are what the LLM is prompted to fill templates for:

| Action Core | Has PyCRAM Mapping? | PyCRAM Class |
|-------------|---------------------|--------------|
| PickingUp | ✅ | PickUpAction |
| Placing | ✅ | PlaceAction |
| Opening | ✅ | OpenAction |
| Shutting | ✅ | CloseAction |
| Pouring | ✅ | PouringAction |
| Cutting | ✅ | CuttingAction |
| Mixing | ✅ | MixingAction |
| Stirring | ✅ (→ MixingAction) | MixingAction |
| Lifting | ✅ (→ PickUpAction) | PickUpAction |
| Taking | ✅ (→ PickUpAction) | PickUpAction |
| Removing | ✅ (→ PickUpAction) | PickUpAction |
| OperatingATap | ✅ (→ PouringAction) | PouringAction |
| Cooling | ❌ | — |
| Adding | ❌ | — |
| Arranging | ❌ | — |
| Baking | ❌ | — |
| Cooking | ❌ | — |
| Evaluating | ❌ | — |
| Filling | ❌ | — |
| Flavouring | ❌ | — |
| Flipping | ❌ | — |
| Neutralizing | ❌ | — |
| Peeling | ❌ | — |
| Pipetting | ❌ | — |
| Preheating | ❌ | — |
| Pressing | ❌ | — |
| Pulling | ❌ | — |
| Rolling | ❌ | — |
| Serving | ❌ | — |
| Shaking | ❌ | — |
| Spooning | ❌ | — |
| Spreading | ❌ | — |
| Sprinkling | ❌ | — |
| Starting | ❌ | — |
| Stopping | ❌ | — |
| Storing | ❌ | — |
| Turning | ❌ | — |
| TurningOnElectricalDevice | ❌ | — |
| Unscrewing | ❌ | — |
| UsingMeasuringCup | ❌ | — |
| UsingSpiceJar | ❌ | — |
| Waiting | ❌ | — |
| Holding | ❌ | — |

**Summary: 11 of 43 action cores have a PyCRAM mapping. 32 have none.**

### 2.3 What PyCRAM Can Execute (24 action classes)

| PyCRAM Class | Module | Category |
|---|---|---|
| PickUpAction | core/pick_up | Core |
| GraspingAction | core/pick_up | Core |
| ReachAction | core/pick_up | Core |
| PlaceAction | core/placing | Core |
| OpenAction | core/container | Core |
| CloseAction | core/container | Core |
| NavigateAction | core/navigation | Core |
| LookAtAction | core/navigation | Core |
| DetectAction | core/misc | Core |
| MoveTorsoAction | core/robot_body | Core |
| SetGripperAction | core/robot_body | Core |
| ParkArmsAction | core/robot_body | Core |
| CarryAction | core/robot_body | Core |
| FollowTCPPathAction | core/robot_body | Core |
| CuttingAction | composite/tool_based | Composite |
| MixingAction | composite/tool_based | Composite |
| PouringAction | composite/tool_based | Composite |
| TransportAction | composite/transporting | Composite |
| PickAndPlaceAction | composite/transporting | Composite |
| MoveAndPlaceAction | composite/transporting | Composite |
| MoveAndPickUpAction | composite/transporting | Composite |
| EfficientTransportAction | composite/transporting | Composite |
| SearchAction | composite/searching | Composite |
| FaceAtAction | composite/facing | Composite |

---

## 3. Identified Mismatches and Issues

### 3.1 Heating — Broken End-to-End

- `IntentType.HEAT = "Heating"` exists in the intent classifier
- **No `Heating` action core** exists in `cram_action_cores.json`
- **No `Heating` Pydantic class** exists in `cram_gen_models.py`
- **No `Heating` mapping** in `_ACTION_MAP`
- **No `HeatingAction`** in PyCRAM

**Effect**: If the user says something like "heat the water", the intent parser classifies it as `Heating`, but the graph crashes at `_json_data[action_core]["action_roles"]` because "Heating" is not a key in the JSON.

**Fix needed**: Either remove `HEAT` from `IntentType`, or add the full chain: action core JSON entry + Pydantic class + PyCRAM mapping.

---

### 3.2 Pulling — Incomplete Chain

- `IntentType.PULL = "Pulling"` exists
- `Pulling` action core exists in the JSON and as a Pydantic class
- **No `Pulling` → PyCRAM mapping** in `_ACTION_MAP`
- No `PullingAction` in PyCRAM

**Effect**: The LLM can generate a valid `Pulling` CRAM plan, but `to_partial_designator` will fail or fall through to a no-op when executing.

**Fix needed**: Map `"pulling"` to the closest PyCRAM action (e.g. `PickUpAction` or `OpenAction` depending on context), or add a composite `PullingAction` to PyCRAM.

---

### 3.3 Opening Action — Body Resolution Fails for Generic LLM Output

- The LLM frequently generates `(type cabinet_drawer)` with no `:tag`
- The body resolver cannot match `cabinet_drawer` to any world body (names like `apartment/cabinet1_drawer_middle` don't substring-match)
- Even if a drawer body were resolved, `OpenAction` requires a **handle** body specifically
- This causes `AttributeError: 'NoneType' object has no attribute '_world'` deep inside pycram

**Fix needed**: In `SimulationBridge`, for Opening actions where body resolution returns `None` or returns a non-handle body, fall back to finding the nearest handle body to the robot (which is safe because the user pre-navigates the robot).

---

### 3.4 Transport Action — Unbounded CostmapLocation Loop

- `TransportAction.execute()` internally searches for a place-navigation pose using `CostmapLocation` with the pickup grasp description as a constraint
- The pickup grasp direction (e.g. side approach) is often incompatible with placing on a table (needs overhead approach)
- IK fails for all 600 candidate positions, so the loop runs indefinitely
- No timeout mechanism inside `TransportAction`

**Fix needed**: In `SimulationBridge.execute()`, detect Transport actions and decompose them into PickUp + Place via `execute_batch()`, which has the timeout-protected `_resolve_placement_nav_pose()` path.

---

### 3.5 LLM CRAM Output Format Inconsistencies

The LLM sometimes generates structurally invalid CRAM strings:

| Problem | Example Output | Expected |
|---------|---------------|----------|
| `(perform ...)` wrapper | `(perform (an action (type open-object) ...))` | `(an action (type Opening) ...)` |
| Wrong action type name | `(type open-object)` | `(type Opening)` |
| Missing `:tag` | `(an object (type cabinet_drawer))` | `(:tag handle_cab10_t (an object ...))` |
| Unmapped action type | `(type Heating)` | No PyCRAM mapping exists |

**Root cause**: The `cram_plan_prompt` is permissive and the LLM has variation in how it formats the output. The parser handles some of these (`open-object` normalizes to `openobject` which maps to `OpenAction`), but body resolution always fails without `:tag`.

**Fix needed**: Either tighten the CRAM generation prompt to always include `:tag`, or make the body resolver smarter for specific action types.

---

### 3.6 32 Action Cores With No PyCRAM Path

The pipeline can generate valid CRAM plans for 32 action cores (Peeling, Flavouring, Baking, etc.) that have no PyCRAM mapping. When a user asks "peel the potato", the system produces a CRAM plan but then silently fails or errors at the serializer step.

These fall into two groups:
- **Physically meaningful but not yet implemented in PyCRAM**: Peeling, Pressing, Filling, Flipping, Spreading, Rolling — would need new composite PyCRAM actions
- **Higher-level / non-robotic**: Evaluating, Waiting, Starting, Stopping, Cooking, Baking — these are cognitive/process concepts, not direct robot motion primitives

---

## 4. Action Coverage Matrix

```
IntentType (10) ─── can classify ──► Action Core (43) ─── CRAM template ──► CRAM string
      │                                                                            │
      │                                                                            ▼
      │                                                              _ACTION_MAP lookup (11 mapped)
      │                                                                            │
      │                                               ┌─────────────┬─────────────┘
      │                                               │ Mapped (11) │ Unmapped (32) → ❌ fails
      │                                               ▼
      │                                        PyCRAM class (24 total)
      │                                               │
      │                                               ▼
      └──── HEAT (no action core) ──────────► ❌ graph crash
            PULL (no PyCRAM mapping) ────────► ❌ serializer error
```

---

## 5. What Works End-to-End Today

| User Intent | Action Core | PyCRAM Execution | Notes |
|-------------|-------------|-----------------|-------|
| Pick up object | PickingUp | ✅ PickUpAction | Requires object `:tag` |
| Place object | Placing | ✅ PlaceAction | Needs nav pre-step |
| Pour liquid | Pouring | ✅ PouringAction | Requires source + dest `:tag` |
| Cut object | Cutting | ✅ CuttingAction | Requires tool `:tag` |
| Open drawer/door | Opening | ⚠️ OpenAction | Fails if no `:tag`; needs handle body |
| Stir contents | Stirring | ⚠️ MixingAction | Mapped to Mix, acceptable approximation |
| Mix contents | Mixing | ✅ MixingAction | |
| Transport object | Transporting | ⚠️ TransportAction | Hangs if place IK fails |
| Close container | Shutting | ✅ CloseAction | |
| Pull object | Pulling | ❌ None | No PyCRAM mapping |
| Heat something | Heating | ❌ None | No action core, crashes graph |
| Cool something | Cooling | ❌ None | No PyCRAM mapping |
| Peel / press / flip etc. | Various | ❌ None | 32 unmapped action cores |

---

## 6. Recommended Changes (Priority Order)

### High Priority

1. **Remove `HEAT` from `IntentType`** or add the full chain (action core JSON + Pydantic class + PyCRAM mapping). Currently causes a hard crash in the action decomposition graph.

2. **Fix `OpenAction` body resolution in `SimulationBridge`**: Add nearest-handle fallback when the body resolver returns `None` for Opening actions.

3. **Fix `TransportAction` hang in `SimulationBridge`**: Decompose Transport into PickUp + Place via `execute_batch()` to use the timeout-protected navigation path.

### Medium Priority

4. **Add `:tag` enforcement to the CRAM generation prompt**: Require the LLM to always produce `:tag NAME` in object expressions. This is the single largest cause of body resolution failures.

5. **Map `Pulling` to a PyCRAM action**: Best approximation is `OpenAction` (pulling a drawer open) or `PickUpAction` (pulling an object toward the robot). Context-dependent.

6. **Map `Cooling` to a PyCRAM action**: Could map to `Waiting` (robot waits while object cools) or simply be treated as a no-op with a log message.

### Low Priority

7. **Expand `IntentType`** to cover the most important currently-unmapped action cores that PyCRAM supports: `Transporting`, `Shutting`, `Navigating` (the intent classifier only has 10 types but PyCRAM can handle far more).

8. **Add composite PyCRAM actions** for physically meaningful operations: Peeling, Pressing, Filling, Flipping — these would require new PyCRAM robot plan implementations.

9. **Add unambiguous error messages** in `SimulationBridge.to_partial_designator()` when `object_designator` is `None` — instead of a cryptic `AttributeError` from deep inside pycram, raise a clear `ValueError` naming the unresolved entity.

---

## 7. Key File Locations

| Component | File |
|-----------|------|
| Intent classification | `llmr/src/llmr/workflows/pydantics/intent_entity_models.py` |
| Action core Pydantic models | `llmr/src/llmr/workflows/pydantics/cram_gen_models.py` |
| CRAM templates (JSON) | `llmr/src/llmr/workflows/resources/cram_action_cores.json` |
| LLM graph & prompts | `llmr/src/llmr/workflows/graphs/enhanced_ad_graph.py` |
| CRAM prompt templates | `llmr/src/llmr/workflows/prompts/cram_gen_prompts.py` |
| CRAM → PyCRAM serializer | `llmr/src/llmr/serializers/cram_to_pycram.py` |
| Body resolver | `llmr/src/llmr/serializers/body_resolver.py` |
| Simulation bridge | `llmr/src/llmr/serializers/simulation_bridge.py` |
| PyCRAM core actions | `pycram/src/pycram/robot_plans/actions/core/` |
| PyCRAM composite actions | `pycram/src/pycram/robot_plans/actions/composite/` |
