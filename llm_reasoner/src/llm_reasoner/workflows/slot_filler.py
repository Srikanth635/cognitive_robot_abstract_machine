"""
Slot filler — the core LLM reasoning calls that power LLMBackend._evaluate()
and the nl_plan() factory's action classification step.

Two public functions:

  classify_action()   — NL instruction → action class (for nl_plan factory)
  run_slot_filler()   — Match context → resolved slot values (for LLMBackend)

Design difference from llmr/workflows/nodes/slot_filler.py:
  - No LangGraph StateGraph — direct LLM calls via with_structured_output().
    LangGraph added complexity without benefit for single-node graphs.
  - No hardcoded action-type-specific schemas (PickUpSlotSchema, PlaceSlotSchema).
    Uses generic ActionReasoningOutput so new action types work automatically.
  - LLM is always injected — no lazy global singletons.
  - Both functions are pure: same inputs → same behaviour, no side effects.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import pkgutil
import typing
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

from llm_reasoner.workflows.schemas.action_slot import (
    ActionClassification,
    ActionReasoningOutput,
)

# --------------------------------------------------------------------------- #
# System prompts                                                               #
# --------------------------------------------------------------------------- #

_SLOT_FILLER_SYSTEM_PROMPT = """\
You are a robot action parameter resolver with strong spatial and physical reasoning.

You receive:
  1. A natural-language instruction from a human operator.
  2. The target robot action type and all its free (unfilled) parameter slots.
  3. Already-fixed slot values (do not change these).
  4. The full current world state: objects, positions, semantic annotations.

Your task: for every FREE slot, reason carefully and produce the most physically
sensible concrete value.

Guidelines per slot type:
  ENTITY slots (object_designator, target, surface, etc.):
    - Identify which world object the instruction refers to.
    - Use name matching, semantic type, spatial context, and attributes.
    - If multiple candidates exist, pick the most contextually salient one
      (closest to robot, most recently mentioned, best matching description).
    - Return the exact world body NAME as the value (it will be looked up).

  ENUM / PARAMETER slots (arm, grasp_type, approach_direction, etc.):
    - Infer from the instruction, object position relative to robot, and physics.
    - "pick up the heavy box" → power grasp, approach from open/accessible side.
    - "hand me the knife" → approach from handle end (safety).
    - If the instruction specifies a value explicitly, always honour it.

Always provide per-slot reasoning. Return structured JSON.
"""

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a robot action classifier.

Given a natural-language instruction, identify which robot action class it
corresponds to from the list of available action classes below.

Available action classes:
{action_classes}

Return the EXACT Python class name (e.g. "PickUpAction" not "pick up action").
Return structured JSON.
"""


# --------------------------------------------------------------------------- #
# Public: action classification (used by nl_plan factory)                     #
# --------------------------------------------------------------------------- #

def classify_action(
    instruction: str,
    llm: "BaseChatModel",
    world: Any,
    action_registry: Optional[Dict[str, type]] = None,
) -> Optional[type]:
    """
    Use the LLM to classify which PyCRAM action class corresponds to the
    NL instruction.

    The action_registry maps class name → class. If not provided, it is
    auto-discovered from the pycram package via _discover_action_classes().

    :param instruction:      The NL instruction.
    :param llm:              LangChain-compatible chat model.
    :param world:            WorldLike (unused here, reserved for future
                             world-conditioned classification).
    :param action_registry:  Optional pre-built registry. Auto-discovered if None.
    :returns: The action class, or None if classification fails.
    """
    if action_registry is None:
        action_registry = _discover_action_classes()

    if not action_registry:
        return None

    class_list = "\n".join(f"  - {name}" for name in sorted(action_registry))
    system_prompt = _CLASSIFIER_SYSTEM_PROMPT.format(action_classes=class_list)

    structured_llm = llm.with_structured_output(ActionClassification)
    try:
        result: ActionClassification = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ])
        return action_registry.get(result.action_type)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Public: slot filling (used by LLMBackend._evaluate)                         #
# --------------------------------------------------------------------------- #

def run_slot_filler(
    instruction: str,
    action_type: str,
    free_slots: List[Tuple[str, Any]],
    fixed_slots: Dict[str, Any],
    world_context: str,
    llm: "BaseChatModel",
) -> Optional[Dict[str, Any]]:
    """
    Core LLM reasoning call: given the full action context and world state,
    resolve concrete values for all free Match slots.

    :param instruction:   The NL instruction.
    :param action_type:   The action class name (e.g. "PickUpAction").
    :param free_slots:    List of (field_name, field_type) for unresolved slots.
    :param fixed_slots:   Already-resolved {field_name: value} (do not change).
    :param world_context: Serialized world state string from world_serializer.
    :param llm:           LangChain-compatible chat model.
    :returns: Dict mapping field_name → resolved value, or None on failure.
    """
    user_message = _build_slot_filler_message(
        instruction=instruction,
        action_type=action_type,
        free_slots=free_slots,
        fixed_slots=fixed_slots,
        world_context=world_context,
    )

    structured_llm = llm.with_structured_output(ActionReasoningOutput)
    try:
        output: ActionReasoningOutput = structured_llm.invoke([
            {"role": "system", "content": _SLOT_FILLER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])
        return {slot.field_name: slot.value for slot in output.slots}
    except Exception:
        logger.exception("run_slot_filler: LLM invocation failed")
        return None


# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #

def _build_slot_filler_message(
    instruction: str,
    action_type: str,
    free_slots: List[Tuple[str, Any]],
    fixed_slots: Dict[str, Any],
    world_context: str,
) -> str:
    lines = [
        f"Instruction: {instruction!r}",
        f"Action type: {action_type}",
        "",
        "Free slots to fill:",
    ]
    for field_name, field_type in free_slots:
        unwrapped = _unwrap_type(field_type)
        type_hint = getattr(unwrapped, "__name__", str(unwrapped)) if unwrapped is not None else "unknown"

        if unwrapped is not None and isinstance(unwrapped, type) and issubclass(unwrapped, Enum):
            members = " | ".join(unwrapped.__members__)
            lines.append(f"  - {field_name}  (type: {type_hint}, allowed values: {members})")

        elif unwrapped is not None and isinstance(unwrapped, type) and dataclasses.is_dataclass(unwrapped):
            lines.append(f"  - {field_name}  (type: {type_hint} — fill its sub-fields using '{field_name}.<sub_field>' naming):")
            for sub_name, sub_type in _complex_subfields(unwrapped):
                sub_hint = getattr(sub_type, "__name__", str(sub_type)) if sub_type is not None else "unknown"
                if sub_type is not None and isinstance(sub_type, type) and issubclass(sub_type, Enum):
                    members = " | ".join(sub_type.__members__)
                    lines.append(f"      - {field_name}.{sub_name}  (type: {sub_hint}, allowed values: {members})")
                else:
                    lines.append(f"      - {field_name}.{sub_name}  (type: {sub_hint})")

        else:
            lines.append(f"  - {field_name}  (type: {type_hint})")

    if fixed_slots:
        lines.append("\nAlready-fixed slots (honour these, do not change):")
        for field_name, value in fixed_slots.items():
            lines.append(f"  - {field_name} = {value!r}")

    lines.append(f"\n{world_context}")
    return "\n".join(lines)


def _unwrap_type(t: Any) -> Any:
    """Strip Optional[X] / Union[X, None] → X. Returns t unchanged for anything else."""
    if typing.get_origin(t) is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


def _is_context_type(t: Any) -> bool:
    """Return True for robot-component types that must be injected from context, not LLM-resolved."""
    try:
        from semantic_digital_twin.robots.abstract_robot import Manipulator, Camera
        return isinstance(t, type) and issubclass(t, (Manipulator, Camera))
    except Exception:
        return False


def _complex_subfields(cls: type) -> List[Tuple[str, Any]]:
    """Return (name, resolved_type) for each LLM-fillable sub-field of a dataclass.

    Skips CONTEXT-typed fields (Manipulator, Camera) — those are injected from
    the world, not resolved by the LLM.
    """
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = getattr(cls, "__annotations__", {})

    result = []
    for f in dataclasses.fields(cls):
        sub_type = _unwrap_type(hints.get(f.name, f.type))
        if _is_context_type(sub_type):
            continue
        result.append((f.name, sub_type))
    return result


def _discover_action_classes() -> Dict[str, type]:
    """
    Auto-discover concrete PyCRAM action classes by scanning the
    pycram.robot_plans.actions package.

    Returns a dict mapping class name → class.
    Falls back to an empty dict if pycram is not importable.

    Mirrors the role of llmr/pycram_bridge/action_registry.py but without
    maintaining a manual registry — discovery is automatic.
    """
    try:
        import pycram.robot_plans.actions as actions_pkg
    except ImportError:
        return {}

    result: Dict[str, type] = {}
    for _, module_name, _ in pkgutil.walk_packages(
        actions_pkg.__path__, prefix=actions_pkg.__name__ + "."
    ):
        try:
            mod = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    name.endswith("Action")
                    and not inspect.isabstract(obj)
                    and obj.__module__.startswith("pycram")
                ):
                    result[name] = obj
        except Exception:
            continue

    return result
