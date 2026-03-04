"""State type definitions for the llmr workflow graphs."""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Dict, List
from ..pydantics.intent_entity_models import InstructionList

from langgraph.graph import MessagesState

from ..pydantics.flanagan_models import FlanaganState

if TYPE_CHECKING:
    from ..pydantics.intent_entity_models import InstructionList


class ActionDecompState(MessagesState):
    """State for the Action Decomposition (AD) pipeline."""

    instruction: str
    action_type: str
    action_core: List[str]
    action_core_attributes: List[str]
    enriched_action_core_attributes: List[str]
    cram_plan_response: List[str]
    context: str
    intents: "InstructionList"
    user_id: str
    thread_id: str


class MainPipelineState(MessagesState):
    """Combined state for the top-level pipeline (Flanagan + FrameNet + AD)."""

    instruction: str
    action_type: str
    action_core: List[str]
    action_core_attributes: List[str]
    enriched_action_core_attributes: List[str]
    cram_plan_response: List[str]
    intents: "InstructionList"
    premotion_phase: str
    phaser: str
    flanagan: str
    framenet_model: str
    context: str


class ModelReasoningState(MessagesState):
    """Internal state shared between model-reasoning agent nodes (FrameNet, Flanagan, PyCRAM)."""

    instruction: str
    action_type: str
    action_core: str
    action_core_attributes: str
    enriched_action_core_attributes: str
    cram_plan_response: str
    intents: dict
    premotion_phase: str
    phaser: str
    flanagan: str
    framenet_model: str
    context: str
    pycram_action_names: str
    pycram_action_models: str


class PyCramGroundingState(MessagesState):
    """State for the PyCRAM grounding pipeline."""

    atomics: str
    cram_plans: str
    belief_state_context: str
    context: str
    grounded_cram_plans: List[str]
    action_names: List[str]
    designator_models: str


__all__ = [
    "FlanaganState",
    "ActionDecompState",
    "MainPipelineState",
    "ModelReasoningState",
    "PyCramGroundingState",
]
