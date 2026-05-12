"""Flanagan motion-phase reasoner for the llmr pipeline.
    from llmr_updated_arch.generation.flanagan import FlanaganReasoner

    backend = LLMBackend(
        llm=llm,
        instruction="pick up the milk from the table",
        symbol_type=WorldBody,
        reasoners=[FlanaganReasoner(llm=llm)],
    )
    result = next(iter(backend.evaluate(match)))
    rep = backend.semantics.motion_phases
    for phase in rep.phases:
        print(phase.phase, phase.target_object)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Union

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Dict, List, Literal, Optional

from llmr_updated_arch.integrations.krrood.introspect import FieldKind
from llmr_updated_arch.integrations.krrood.match_reader import render_resolved_slots
from llmr_updated_arch.generation import Reasoner
from llmr_updated_arch.schemas import MotionPhase, FlanaganMotionPlan

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llmr_updated_arch.integrations.krrood.match_reader import MatchSnapshot as MatchData
    from llmr_updated_arch.schemas import ActionAnnotationBundle as ActionSemantics

logger = logging.getLogger(__name__)

# ── Phase alias table for in-process normalisation ────────────────────────────
# Maps common LLM synonyms (lowercase_snake) → canonical CamelCase name.
# Used by _canonicalize_phase_name(); never enforced on the LLM.

_PHASE_ALIASES: Dict[str, str] = {
    "approach": "Approach", "reach": "Approach", "move_to": "Approach",
    "grasp": "Grasp", "grab": "Grasp", "grip": "Grasp", "pick": "Grasp", "pick_up": "Grasp",
    "lift": "Lift", "raise": "Lift", "elevate": "Lift",
    "transport": "Transport", "carry": "Transport", "move": "Transport", "transfer": "Transport",
    "place": "Place", "set_down": "Place", "put_down": "Place", "put": "Place", "lower": "Place",
    "align": "Align", "position": "Align", "orient": "Align",
    "cut": "Cut", "slice": "Slice",
    "tilt": "Tilt", "pour": "Pour",
    "insert": "Insert", "push_in": "Insert",
    "withdraw": "Withdraw", "pull_out": "Withdraw", "retract": "Withdraw",
    "release": "Release", "let_go": "Release", "drop": "Release",
    "reorient": "Reorient", "rotate": "Reorient", "flip": "Reorient",
    "stabilize": "Stabilize", "steady": "Stabilize", "hold": "Stabilize",
    "inspect": "Inspect", "check": "Inspect", "verify": "Inspect",
}


def _canonicalize_phase_name(raw: str) -> str:
    """Map a raw LLM-generated phase name to the canonical vocabulary.

    Lowercases and snake-cases *raw* then looks it up in ``_PHASE_ALIASES``.
    If already a canonical name (case-insensitive), returns the canonical form.
    Falls back to ``raw.title()`` for truly novel phases so they still look tidy.
    """
    key = raw.lower().replace(" ", "_").replace("-", "_")
    if key in _PHASE_ALIASES:
        return _PHASE_ALIASES[key]
    canonical_lower = {v.lower(): v for v in _PHASE_ALIASES.values()}
    if key in canonical_lower:
        return canonical_lower[key]
    return raw.title() if raw else raw


# ── Pipeline-internal Pydantic models (not part of public schemas) ────────────


class _PhaseStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: str
    target_object: str
    description: Optional[str] = None


class _ObjectAwarePhasePlanner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phases: List[_PhaseStep]


# ── Building-block schemas (shared by _PhaseAnnotation) ──────────────────────


class _ConditionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    value: Union[str, bool, int, float, None] = None


class _ForceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = ""
    expected_range_N: Optional[List[float]] = None
    expected_range_Nm: Optional[List[float]] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_malformed(cls, data: Any) -> Any:
        """Normalise varied LLM outputs: plain str, wrong-key dict, or correct dict."""
        def _extract_range(text: str) -> Optional[List[float]]:
            nums = re.findall(r"[\d.]+", str(text))
            return [float(nums[0]), float(nums[1])] if len(nums) >= 2 else None

        if isinstance(data, str):
            return {"type": data, "expected_range_N": _extract_range(data)}
        if isinstance(data, dict) and "type" not in data:
            key = next(iter(data), "unknown")
            val = data[key]
            return {"type": key, "expected_range_N": _extract_range(val) if isinstance(val, str) else None}
        return data


class _ForceDynamics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contact: bool = False
    motion_type: str = ""
    force_exerted: str = ""
    force_profile: Optional[_ForceProfile] = None


class _SensorSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal_name: str
    value: Union[str, bool, int, float, None] = None


# ── Holistic annotation schema (Call 2 in the 2-call pipeline) ───────────────
# Replaces the 6 separate map schemas above once Phase 4 rewires the pipeline.
# All six annotation types live inline per phase so the LLM reasons holistically
# and can cross-reference (e.g. force profile informs recovery strategy).


class _PhaseAnnotation(BaseModel):
    """All annotations for one motion phase — produced in a single LLM call."""

    model_config = ConfigDict(extra="forbid")

    phase_key: str
    """Matches the ``{Phase}_{object}`` key from Call 1 (e.g. ``"Grasp_milk"``)."""

    preconditions: List[_ConditionEntry] = Field(default_factory=list)
    """Symbolic conditions that must hold before this phase starts."""

    goal_state: List[_ConditionEntry] = Field(default_factory=list)
    """Symbolic conditions that must hold after this phase completes."""

    force_dynamics: Optional[_ForceDynamics] = None
    """Contact type, motion type and force profile; None for non-contact phases."""

    sensory_feedback: List[_SensorSignal] = Field(default_factory=list)
    """Expected sensor signals (force, vision, proprioception) during execution."""

    possible_failures: List[str] = Field(default_factory=list)
    """Failure modes that could prevent this phase from completing."""

    recovery_strategies: List[str] = Field(default_factory=list)
    """Corrective actions the robot can attempt for each failure mode."""

    max_duration_sec: float = 5.0
    """Upper-bound execution time in seconds."""

    urgency: Literal["low", "medium", "high"] = "medium"
    """Timing pressure for this phase."""


class _AnnotatedPlan(BaseModel):
    """Holistic annotation of all phases — output of the single annotation call."""

    model_config = ConfigDict(extra="forbid")
    phases: List[_PhaseAnnotation]


# ── Prompt templates ──────────────────────────────────────────────────────────

_TASK_DECOMPOSER_PROMPT = ChatPromptTemplate.from_template("""
You are an embodied robot that must physically perform manipulation tasks.

**IMPORTANT: You may be executing a sequence of actions. Consider what has already been done.**

Given context:
1. Current instruction: What you need to do now
2. Previous actions: What you've already completed (if any)

**Use previous actions to understand current state:**
- If previous action was "pick up the cup", the cup is now in your gripper
- If previous action was "open the drawer", the drawer is now open
- If previous action was "cut the apple", the apple is now in pieces
- Skip redundant phases based on what's already done

Before selecting motion phases, reason through:
1. What objects are involved and what are their properties?
2. What is the CURRENT state based on previous actions?
3. Are there containers (drawer, box, jar, shelf)? Do they need opening/closing?
4. What are the ordered sub-goals for THIS instruction?
5. What SPECIFIC object or part is being manipulated at each step?

Use short, descriptive verb phrases for phase names (e.g. "Approach", "Grasp", "Lift").
Standard phases include: Approach, Grasp, Lift, Transport, Place, Align, Cut, Slice, Tilt, Pour,
Insert, Withdraw, Release, Reorient, Stabilize, Inspect.
If none of these fits a step, use a clear descriptive name — do not force a poor match.

**Typical Patterns (adapt freely based on the actual task):**
- Transfer/move an object: Approach → Grasp → Lift → Transport → Place → Release
- Open drawer/door: Approach → Grasp (handle) → Withdraw (pull open)
- Pick from container: Approach → Grasp → Lift → Withdraw
- Place into container: Approach → Align → Insert → Release
- Pouring: Grasp → Align → Tilt → Pour → Reorient
- Cutting: Grasp (object) → Stabilize → Grasp (knife) → Align → Cut / Slice

**Context-Aware Examples:**

Example 1 - No previous context:
<previous_actions>: []
<instruction>: cut the apple into slices
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "apple", "description": "move to apple on cutting board"}},
    {{"phase": "Stabilize", "target_object": "apple", "description": "hold apple steady with one hand"}},
    {{"phase": "Grasp", "target_object": "knife", "description": "grip the knife with the other hand"}},
    {{"phase": "Align", "target_object": "knife", "description": "position knife above apple"}},
    {{"phase": "Cut", "target_object": "apple", "description": "slice downward through apple"}},
    {{"phase": "Release", "target_object": "knife", "description": "set knife down safely"}}
  ]
}}

Example 2 - With previous context (cup already in hand):
<previous_actions>: ["pick up the cup from the table"]
<instruction>: place it in the sink
<o>: {{
  "phases": [
    {{"phase": "Transport", "target_object": "cup", "description": "carry cup to sink"}},
    {{"phase": "Align", "target_object": "cup", "description": "position cup over sink"}},
    {{"phase": "Place", "target_object": "cup", "description": "set cup in sink"}},
    {{"phase": "Release", "target_object": "cup", "description": "release grip on cup"}}
  ]
}}
Note: No Approach/Grasp/Lift needed - cup already in gripper!

Example 3 - With previous context (drawer already open):
<previous_actions>: ["open the drawer"]
<instruction>: take out the spoon
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "spoon", "description": "move to spoon in open drawer"}},
    {{"phase": "Grasp", "target_object": "spoon", "description": "grip the spoon"}},
    {{"phase": "Lift", "target_object": "spoon", "description": "lift spoon from drawer"}},
    {{"phase": "Withdraw", "target_object": "arm", "description": "extract arm from drawer"}}
  ]
}}
Note: No drawer opening needed - already open!

**Standard Examples (no context):**

<previous_actions>: []
<instruction>: place the bowl on the counter
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "counter", "description": "move toward counter"}},
    {{"phase": "Align", "target_object": "bowl", "description": "position bowl directly above counter surface"}},
    {{"phase": "Place", "target_object": "bowl", "description": "lower bowl onto counter"}},
    {{"phase": "Release", "target_object": "bowl", "description": "release grip on bowl"}}
  ]
}}

<previous_actions>: []
<instruction>: pour water from bottle into glass
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "bottle", "description": "move to bottle"}},
    {{"phase": "Grasp", "target_object": "bottle", "description": "grip bottle"}},
    {{"phase": "Lift", "target_object": "bottle", "description": "lift bottle"}},
    {{"phase": "Transport", "target_object": "bottle", "description": "move bottle to glass"}},
    {{"phase": "Align", "target_object": "bottle", "description": "position bottle over glass"}},
    {{"phase": "Tilt", "target_object": "bottle", "description": "tilt bottle to pour angle"}},
    {{"phase": "Pour", "target_object": "water", "description": "pour water into glass"}},
    {{"phase": "Reorient", "target_object": "bottle", "description": "return bottle upright"}},
    {{"phase": "Place", "target_object": "bottle", "description": "set bottle down"}},
    {{"phase": "Release", "target_object": "bottle", "description": "release grip"}}
  ]
}}

---

Now analyze this instruction with its context and provide object-aware motion phases as a JSON object:

REMEMBER:
1. Prefer standard phase names where they fit; use a descriptive name when they don't.
2. Consider previous actions to avoid redundant phases.
3. If object is already grasped, skip Approach/Grasp/Lift.
4. If container is already open, skip opening phases.

<previous_actions>: {previous_actions}
<instruction>: {instruction}
""")





# ── Holistic annotation prompt (Call 2) ───────────────────────────────────────

_ANNOTATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert robotics motion planner. Annotate ALL phases of a manipulation task \
in a SINGLE pass, reasoning holistically across the full sequence.

You will receive:
- instruction: the NL task the robot must execute
- phases: ordered list of motion phases (each has a phase_key, phase name, target object, description)
- world_context: a serialized snapshot of the current scene (objects, types, positions, relations)
- resolved_slots: the specific arm, object instance, and grasp configuration already resolved

For EACH phase produce:
  preconditions      — symbolic key-value conditions that must hold before the phase starts
  goal_state         — symbolic key-value conditions that must hold after the phase completes
  force_dynamics     — contact type, motion type, force estimate and force profile (null for non-contact phases)
  sensory_feedback   — expected sensor signals during execution (force, vision, proprioception)
  possible_failures  — failure modes that could prevent this phase completing
  recovery_strategies — corrective actions for each failure mode
  max_duration_sec   — upper-bound execution time in seconds (float)
  urgency            — "low", "medium", or "high"

HOW TO USE WORLD CONTEXT:
- Use object positions to determine spatial preconditions (reachability, clearance)
- Use object type / annotations to ground force estimates (heavy/light, rigid/soft, large/small)
- Use scene layout to identify occlusion or obstacle failure modes
- If world_context is "(no world context available)", reason from object names and instruction alone

HOW TO USE RESOLVED PARAMETERS:
- Specialise sensor names to the resolved arm (e.g. "LEFT_force_sensor_N" when arm=LEFT)
- Use the grounded object's specific name in goal states (e.g. "milk_bottle in LEFT gripper")
- Use grasp type to select the appropriate force mode and approach direction
- If resolved_slots is "(no resolved parameters)", reason from instruction alone

CRITICAL RULES:
- Use EXACTLY the phase_key strings from the phases list — do not invent or modify them
- For non-contact phases (Approach, Transport, Inspect): set force_dynamics to null
- Precondition and goal_state values can be: boolean, string, or number
- Populate all phases — do not skip any

OUTPUT FORMAT:
{{
  "phases": [
    {{
      "phase_key": "<exact value from phases list>",
      "preconditions": [{{"key": "gripper_open", "value": true}}],
      "goal_state": [{{"key": "object_grasped", "value": true}}],
      "force_dynamics": {{
        "contact": true,
        "motion_type": "pinch",
        "force_exerted": "5-8N",
        "force_profile": {{"type": "compression", "expected_range_N": [5, 8]}}
      }},
      "sensory_feedback": [{{"signal_name": "force_sensor_N", "value": 6.0}}],
      "possible_failures": ["gripper misaligned"],
      "recovery_strategies": ["reopen and reposition gripper"],
      "max_duration_sec": 2.5,
      "urgency": "medium"
    }}
  ]
}}

---

EXAMPLE:

Instruction: place the book on the top shelf
World context:
  - book: PrintedMaterial at (0.30m, 0.10m, 0.95m), light, rigid — already in robot gripper
  - shelf_1: Furniture at (1.50m, 0.20m, 1.40m), top shelf surface clear
Resolved parameters:
  Action: PlaceAction
  - object_designator: book (Body, position 0.30m, 0.10m, 0.95m)
  - arm: RIGHT
  - target_location: shelf_1 (SupportSurface, position 1.50m, 0.20m, 1.42m)

Phases:
[
  {{"phase_key": "Approach_shelf_1", "phase": "Approach", "target_object": "shelf_1", "description": "move arm toward shelf"}},
  {{"phase_key": "Align_book",       "phase": "Align",    "target_object": "book",    "description": "position book directly above shelf surface"}},
  {{"phase_key": "Place_book",       "phase": "Place",    "target_object": "book",    "description": "lower book onto shelf"}}
]

Output:
{{
  "phases": [
    {{
      "phase_key": "Approach_shelf_1",
      "preconditions": [
        {{"key": "book_in_RIGHT_gripper", "value": true}},
        {{"key": "shelf_visible", "value": true}},
        {{"key": "path_to_shelf_clear", "value": true}}
      ],
      "goal_state": [
        {{"key": "arm_near_shelf", "value": true}},
        {{"key": "distance_to_shelf_m", "value": 0.2}}
      ],
      "force_dynamics": null,
      "sensory_feedback": [
        {{"signal_name": "camera_shelf_detected", "value": true}},
        {{"signal_name": "RIGHT_arm_joint_position", "value": "pre_place"}}
      ],
      "possible_failures": ["shelf not reachable", "path blocked by obstacle"],
      "recovery_strategies": ["replan arm trajectory", "request operator assistance"],
      "max_duration_sec": 3.0,
      "urgency": "low"
    }},
    {{
      "phase_key": "Align_book",
      "preconditions": [
        {{"key": "arm_near_shelf", "value": true}},
        {{"key": "book_in_RIGHT_gripper", "value": true}}
      ],
      "goal_state": [
        {{"key": "book_above_shelf_surface", "value": true}},
        {{"key": "book_orientation_level", "value": true}}
      ],
      "force_dynamics": null,
      "sensory_feedback": [
        {{"signal_name": "camera_book_over_shelf", "value": true}},
        {{"signal_name": "RIGHT_wrist_angle_deg", "value": 0.0}}
      ],
      "possible_failures": ["misalignment with shelf edge", "insufficient clearance above shelf"],
      "recovery_strategies": ["re-adjust wrist rotation", "raise arm and reposition"],
      "max_duration_sec": 2.0,
      "urgency": "medium"
    }},
    {{
      "phase_key": "Place_book",
      "preconditions": [
        {{"key": "book_above_shelf_surface", "value": true}},
        {{"key": "book_orientation_level", "value": true}}
      ],
      "goal_state": [
        {{"key": "book_on_shelf", "value": true}},
        {{"key": "book_stable", "value": true}}
      ],
      "force_dynamics": {{
        "contact": true,
        "motion_type": "downward_placement",
        "force_exerted": "1-3N downward",
        "force_profile": {{"type": "compression", "expected_range_N": [1, 3]}}
      }},
      "sensory_feedback": [
        {{"signal_name": "RIGHT_force_sensor_N", "value": 2.0}},
        {{"signal_name": "contact_with_shelf_surface", "value": true}},
        {{"signal_name": "RIGHT_gripper_load_N", "value": 0.5}}
      ],
      "possible_failures": ["book slides off shelf edge", "shelf height estimate wrong"],
      "recovery_strategies": ["reposition and retry placement", "adjust target height by -0.02m"],
      "max_duration_sec": 2.5,
      "urgency": "medium"
    }}
  ]
}}

---

Now annotate the following task:

Instruction: {instruction}

World context:
{world_context}

Resolved action parameters:
{resolved_slots}

Phases:
{phases}

Annotate EACH phase. Use the EXACT phase_key values from the phases list above.
""")


# ── Reasoner ──────────────────────────────────────────────────────────────────


class FlanaganReasoner(Reasoner):
    """Populate `semantics.motion_phases` with a Flanagan motion-phase plan.

    The pipeline uses two LLM calls:
    1. decompose the instruction into ordered motion phases
    2. annotate those phases using the world context and resolved action slots

    If `semantics.instruction` is empty, the reasoner skips silently.

    :param llm: LangChain-compatible chat model.
    :param previous_actions: Optional list of already-completed NL instructions
        for sequential-task context.
    """

    REASONER_NAME = "flanagan_reasoner"
    PROMPT_VERSION = "flanagan_v1"

    def __init__(
        self,
        llm: "BaseChatModel",
        previous_actions: Optional[List[str]] = None,
    ) -> None:
        self._previous_actions: List[str] = previous_actions or []
        self._decomposer_chain = (
            _TASK_DECOMPOSER_PROMPT
            | llm.with_structured_output(_ObjectAwarePhasePlanner, method="function_calling")
        )
        self._annotation_chain = (
            _ANNOTATION_PROMPT
            | llm.with_structured_output(_AnnotatedPlan, method="function_calling")
        )

    def annotate(
        self,
        semantics: "ActionSemantics",
        match_data: "MatchData",
        world_context: str,
    ) -> None:
        instruction = semantics.instruction
        if not instruction:
            logger.debug("FlanaganReasoner: no instruction on semantics — skipping.")
            return

        # ── Call 1: free-form phase decomposition ─────────────────────────────
        phase_plan: _ObjectAwarePhasePlanner = self._decomposer_chain.invoke(
            {"instruction": instruction, "previous_actions": self._previous_actions}
        )

        # Normalise phase names in-process and attach phase_key for prompt + lookup.
        phases_with_objects = [
            {
                "phase": _canonicalize_phase_name(phase_step.phase),
                "target_object": phase_step.target_object,
                "description": phase_step.description,
                "phase_key": (
                    f"{_canonicalize_phase_name(phase_step.phase)}_{phase_step.target_object}"
                    if phase_step.target_object
                    else _canonicalize_phase_name(phase_step.phase)
                ),
            }
            for phase_step in phase_plan.phases
        ]

        # ── Call 2: holistic annotation with world + match context ────────────
        resolved_slots_text = render_resolved_slots(match_data)
        annotation_by_key: Dict[str, _PhaseAnnotation] = {}
        try:
            annotation_result: _AnnotatedPlan = self._annotation_chain.invoke({
                "instruction": instruction,
                "phases": phases_with_objects,
                "world_context": world_context or "(no world context available)",
                "resolved_slots": resolved_slots_text,
            })
            annotation_by_key = {a.phase_key: a for a in annotation_result.phases}
        except Exception as exc:
            logger.warning(
                "FlanaganReasoner: annotation step failed (%s) — phases will have empty annotations.", exc
            )

        # ── Compose final representation ──────────────────────────────────────
        composed_phases: List[MotionPhase] = []
        for entry in phases_with_objects:
            phase = entry["phase"]
            obj = entry.get("target_object", "")
            ann = annotation_by_key.get(entry["phase_key"])

            composed_phases.append(
                MotionPhase(
                    phase=phase,
                    target_object=obj,
                    description=entry.get("description"),
                    symbol=(
                        f"->[ robot {phase.lower()}s {obj}]"
                        if obj else f"->[ robot performs {phase.lower()}]"
                    ),
                    preconditions={c.key: c.value for c in ann.preconditions} if ann else {},
                    goal_state={c.key: c.value for c in ann.goal_state} if ann else {},
                    force_dynamics=(
                        ann.force_dynamics.model_dump() if ann and ann.force_dynamics else {}
                    ),
                    sensory_feedback=(
                        {s.signal_name: s.value for s in ann.sensory_feedback} if ann else {}
                    ),
                    failure_and_recovery=(
                        {
                            "possible_failures": ann.possible_failures,
                            "recovery_strategies": ann.recovery_strategies,
                        }
                        if ann else {}
                    ),
                    temporal_constraints=(
                        {"max_duration_sec": ann.max_duration_sec, "urgency": ann.urgency}
                        if ann else {}
                    ),
                )
            )

        semantics.motion_phases = FlanaganMotionPlan(
            instruction=instruction,
            phases=composed_phases,
        )
