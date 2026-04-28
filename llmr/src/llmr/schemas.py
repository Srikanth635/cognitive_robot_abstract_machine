"""Pydantic schemas for LLM structured output.

Three categories:

  - :class:`EntityDescription` — the LLM's pre-grounding description of a
    world entity; the grounder turns it into a :class:`Symbol` instance.
  - :class:`SlotValue`, :class:`SlotFillingOutput`,
    :class:`ActionClassificationResult` — slot-filling and action-classification
    outputs consumed by :class:`LLMBackend` and the NL factory entry points.
  - :class:`ActionAnnotationBundle` (+ :class:`FlanaganMotionPlan`,
    :class:`FrameNetAnnotation`) — open-schema sidecar that accumulates
    LLM-inferred annotations around one action.  Not read by the grounder or
    PyCRAM execution — populated by pluggable :class:`~llmr.reasoning.Reasoner`
    implementations after slot filling completes.
"""

from __future__ import annotations

from typing_extensions import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ── Entity description (pre-grounding) ────────────────────────────────────────


class EntityDescription(BaseModel):
    """
    Semantic description of an entity BEFORE it is resolved to a world object.

    The LLM populates this from the NL instruction. LLMBackend uses it to
    reason about which world body the user is referring to — considering
    name, semantic type, spatial context, and discriminating attributes.

    This is deliberately a description (not a grounded ID) so the LLM can
    apply contextual reasoning rather than pure name matching.
    """

    name: str
    """
    The noun phrase as it appears in the instruction.
    E.g. "milk bottle", "red cup on the left", "the heavy box".
    """

    semantic_type: Optional[str] = None
    """
    Ontological type hint from the instruction or world annotations.
    E.g. "FoodItem", "Container", "SupportSurface".
    Used to narrow candidate bodies by annotation type.
    """

    spatial_context: Optional[str] = None
    """
    Spatial relationship string from the instruction.
    E.g. "on the kitchen counter", "next to the sink", "in the fridge".
    Used for proximity-based disambiguation when multiple candidates exist.
    """

    attributes: Optional[Dict[str, str]] = None
    """
    Discriminating key/value attributes from the instruction.
    E.g. {"color": "red", "size": "large", "material": "glass"}.
    """


# ── Slot-filling output ───────────────────────────────────────────────────────


class SlotValue(BaseModel):
    """A single resolved slot produced by the LLM reasoning step."""

    field_name: str
    """
    Name of the Match field being resolved.
    For complex sub-fields use dotted notation: 'grasp_description.grasp_type'.
    Must match an attribute name (or sub-attribute) on the action class.
    """

    value: Optional[str] = None
    """
    Resolved concrete value as a string.
    - ENUM / parameter slots: the enum member name (e.g. 'LEFT', 'FRONT').
    - ENTITY slots: the world body display name — kept for fallback grounding
      if entity_description is absent.
    - COMPLEX sub-field slots (dotted names): the resolved sub-field value.
    Null when entity_description fully captures the resolution.
    """

    entity_description: Optional[EntityDescription] = None
    """
    For ENTITY and POSE slots: the LLM's semantic description of the entity.
    The grounder uses name + semantic_type + spatial_context + attributes to
    find the matching Symbol instance in SymbolGraph.
    Required for ENTITY/POSE slots; null for parameter/enum/primitive slots.
    """

    reasoning: str = ""
    """Per-slot explanation of why this value was chosen."""


class SlotFillingOutput(BaseModel):
    """
    Structured output from the LLM slot-filling step inside LLMBackend._evaluate().

    Generic across all action types — no per-action subclassing needed.
    One SlotValue per free slot (top-level and complex sub-fields combined).
    """

    action_type: str
    """The action class name being resolved (echoed back for traceability)."""

    slots: List[SlotValue]
    """
    One entry per free slot in the Match expression.
    Nested complex Match leaves are represented as dotted entries, e.g.:
      SlotValue(field_name='grasp_description.grasp_type', value='TOP')
      SlotValue(field_name='grasp_description.approach_direction', value='FRONT')
    Entity sub-fields inside complex fields may also appear as dotted entries
    with entity_description populated.
    """

    overall_reasoning: str = ""
    """High-level explanation of the resolution strategy."""


# ── Action classification ─────────────────────────────────────────────────────


class ActionClassificationResult(BaseModel):
    """Output of the action classification step used by `plan_from_instruction()`."""

    action_type: str
    """
    Exact Python class name of the chosen action.
    E.g. 'PickUpAction', 'NavigateAction', 'PlaceAction'.
    Must match a key in the action registry.
    """

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    """LLM self-reported confidence. Informational only."""

    reasoning: str = ""
    """Why this action type was chosen."""


# ── Semantic sidecar (future reasoners) ───────────────────────────────────────


# ── Flanagan motion-phase schemas ─────────────────────────────────────────────


class MotionPhase(BaseModel):
    """One composed phase entry produced by
    :class:`~llmr.reasoning.flanagan_reasoner.FlanaganReasoner`.

    Each field mirrors the corresponding sub-pipeline output keyed by
    ``{phase}_{target_object}`` (e.g. ``"Grasp_milk"``).
    """

    phase: str
    """Canonical phase name (e.g. ``"Grasp"``, ``"Lift"``, ``"Transport"``)."""

    target_object: str
    """Object or part being manipulated in this phase."""

    description: Optional[str] = None
    """Brief human-readable description of what happens."""

    symbol: str = ""
    """Symbolic action notation, e.g. ``"->[ robot grasps milk]"``."""

    goal_state: Dict[str, Any] = Field(default_factory=dict)
    """Symbolic conditions that must hold after the phase completes."""

    preconditions: Dict[str, Any] = Field(default_factory=dict)
    """Symbolic conditions required before the phase can start."""

    force_dynamics: Dict[str, Any] = Field(default_factory=dict)
    """Contact type, motion type, force exerted, and force profile."""

    sensory_feedback: Dict[str, Any] = Field(default_factory=dict)
    """Expected sensor signals (force, vision, proprioception) during execution."""

    failure_and_recovery: Dict[str, Any] = Field(default_factory=dict)
    """Possible failure modes and corresponding recovery strategies."""

    temporal_constraints: Dict[str, Any] = Field(default_factory=dict)
    """Timing bounds: ``max_duration_sec`` and ``urgency`` level."""


class FlanaganMotionPlan(BaseModel):
    """Complete Flanagan motion-phase plan for one action instruction.

    Produced by :class:`~llmr.reasoning.flanagan_reasoner.FlanaganReasoner`
    and stored on :attr:`ActionAnnotationBundle.motion_phases`.
    """

    instruction: str
    """The original NL instruction this plan was generated for."""

    phases: List[MotionPhase]
    """Ordered list of motion phases with full per-phase annotations."""


# ── FrameNet schemas ──────────────────────────────────────────────────────────


class FrameNetCoreElements(BaseModel):
    """Core Frame Elements — conceptually necessary participants in a FrameNet frame."""

    model_config = {"extra": "forbid"}

    agent: Optional[str] = None
    """Volitional entity performing the action (typically the robot)."""

    theme: Optional[str] = None
    """Entity undergoing motion or location change (motion verbs)."""

    patient: Optional[str] = None
    """Entity undergoing physical modification or direct effect (change-of-state verbs)."""

    instrument: Optional[str] = None
    """Tool or means used to perform the action."""

    source: Optional[str] = None
    """Origin location or initial state."""

    goal: Optional[str] = None
    """Destination location or target state."""

    result: Optional[str] = None
    """Resulting state or configuration after the action."""

    other_core_elements: Optional[str] = Field(
        default=None,
        description="Additional core elements as comma-separated key:value pairs.",
    )


class FrameNetPeripheralElements(BaseModel):
    """Peripheral Frame Elements — optional circumstantial modifiers."""

    model_config = {"extra": "forbid"}

    location: Optional[str] = None
    manner: Optional[str] = None
    direction: Optional[str] = None
    time: Optional[str] = None
    purpose: Optional[str] = None
    quantity: Optional[str] = None
    portion: Optional[str] = None
    speed: Optional[str] = None
    path: Optional[str] = None

    other_peripheral_elements: Optional[str] = Field(
        default=None,
        description="Additional peripheral elements as comma-separated key:value pairs.",
    )


class FrameNetAnnotation(BaseModel):
    """Complete FrameNet-style semantic representation of one action instruction.

    Produced by :class:`~llmr.reasoning.framenet_reasoner.FrameNetReasoner` and
    stored on :attr:`ActionAnnotationBundle.frames`.
    """

    model_config = {"extra": "forbid", "populate_by_name": True}

    framenet: str = Field(
        description="Snake_case semantic label for the action type "
        "(e.g., picking_up_object, cutting_food)."
    )
    frame: str = Field(
        description="Official FrameNet frame name in CamelCase (e.g., Getting, Placing, Cutting)."
    )
    lexical_unit: str = Field(
        alias="lexical-unit",
        description="Lexical unit that evokes the frame: lemma.pos (e.g., pick_up.v).",
    )
    core: FrameNetCoreElements = Field(description="Core frame elements.")
    peripheral: FrameNetPeripheralElements = Field(description="Peripheral frame elements.")

    @field_validator("lexical_unit")
    @classmethod
    def _validate_lexical_unit(cls, v: str) -> str:
        if "." not in v:
            raise ValueError("Lexical unit must follow format: lemma.pos (e.g., pick_up.v)")
        return v


# ── Semantic sidecar ──────────────────────────────────────────────────────────


class ActionAnnotationBundle(BaseModel):
    """Sidecar bundle of LLM-inferred annotations around one action.

    The grounder and PyCRAM execution never read it.  Populated opportunistically
    by :class:`~llmr.reasoning.Reasoner` implementations after slot filling
    completes; preserved on :attr:`LLMBackend.semantics` for downstream consumers
    (explainers, monitors, replay tools, planners).
    """

    action_type: str
    """Echo of the action class name for traceability across reasoner outputs."""

    instruction: Optional[str] = None
    """Original NL instruction — copied from :attr:`LLMBackend.instruction` so
    reasoners that need it (e.g. :class:`~llmr.reasoning.framenet_reasoner.FrameNetReasoner`)
    can read it without receiving the backend as a dependency."""

    # ── Core reasoners ────────────────────────────────────────────────────────
    classification: Optional[ActionClassificationResult] = None
    """Output of the action-classification step (:func:`~llmr.reasoning.slot_filler.classify_action`)."""

    slot_filling: Optional[SlotFillingOutput] = None
    """Output of the slot-filling step (:func:`~llmr.reasoning.slot_filler.run_slot_filler`)."""

    # ── Pluggable reasoner outputs ────────────────────────────────────────────
    motion_phases: Optional[FlanaganMotionPlan] = None
    """Flanagan motion-phase plan — populated by
    :class:`~llmr.reasoning.flanagan_reasoner.FlanaganReasoner`."""

    frames: Optional[FrameNetAnnotation] = None
    """FrameNet semantic representation — populated by
    :class:`~llmr.reasoning.framenet_reasoner.FrameNetReasoner`."""

    preconditions: Optional[List[str]] = None
    postconditions: Optional[List[str]] = None
    affordances: Optional[List[str]] = None

    # Open bag for experimental reasoners before they earn a typed slot
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

