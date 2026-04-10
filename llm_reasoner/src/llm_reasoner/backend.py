"""
LLMBackend — a GenerativeBackend that uses an LLM as a reasoning engine.

This is the core of llm_reasoner. It follows exactly the same pattern as
ProbabilisticBackend in krrood/entity_query_language/backends.py but instead
of sampling from a probabilistic model, it asks an LLM to reason over the full
world state and produce concrete values for all free Match slots.

Variable assignment pattern (mirrors parameterizer.py and backends.py):
    mapped_var = expression._get_mapped_variable_by_name(field_name)
    mapped_var._value_ = resolved_value
    expression._update_kwargs_from_literal_values()
    yield expression.construct_instance()
"""
from __future__ import annotations

import dataclasses
import logging
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

T = TypeVar("T")


@dataclass
class LLMBackend(GenerativeBackend):
    """
    A GenerativeBackend that uses an LLM as a reasoning engine to fill
    underspecified Match expressions from natural-language instructions.

    Unlike ProbabilisticBackend (which samples from a probabilistic model),
    LLMBackend leverages the LLM's world knowledge and common-sense reasoning
    to infer concrete values for all free slots — including:

    - Entity grounding: "the milk on the table" → specific Body object in world
    - Parameter inference: arm, grasp type, approach direction from context
    - Constraint reasoning: physical feasibility, spatial relationships
    - Ambiguity resolution: when multiple candidates exist, pick the most salient

    The EQL / SymbolGraph layer is used only for final validation (does the
    resolved entity actually exist in the world?), not for the reasoning itself.

    Usage::

        context.query_backend = LLMBackend(
            instruction="pick up the milk from the table",
            llm=my_llm,
            world=context.world,
        )
        action = underspecified(PickUpAction)(object_designator=..., arm=..., grasp_description=...)
        plan = execute_single(action, context)
        plan.perform()
    """

    instruction: str
    """The natural-language instruction describing what the robot should do."""

    llm: "BaseChatModel"
    """
    A LangChain-compatible chat model used for reasoning.
    Inject via make_llm() or pass any BaseChatModel directly.
    No global singletons — the caller controls the model.
    """

    world: Any
    """
    The world object (SDT WorldLike) used to:
    - Serialize world state for LLM context
    - Validate that LLM-resolved entity names exist in the world
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        """
        Core evaluation: ask the LLM to fill all free slots in the Match,
        then assign the resolved values and yield a constructed instance.

        Mirrors the pattern from ProbabilisticBackend._evaluate() and
        EntityQueryLanguageBackend._generate_raw_results(), but instead of
        sampling from a model or enumerating a variable domain, we call the LLM.
        """
        from llm_reasoner.world_serializer import serialize_world
        from llm_reasoner.workflows.slot_filler import run_slot_filler

        # ------------------------------------------------------------------ #
        # Step 1: Identify free slots (Ellipsis) and already-fixed slots      #
        # This mirrors assignments_for_conditioning in parameterizer.py       #
        # ------------------------------------------------------------------ #
        free_slots: List[Tuple[str, Any]] = []   # [(field_name, field_type), ...]
        fixed_slots: Dict[str, Any] = {}
        field_types: Dict[str, Any] = {}         # field_name → resolved type

        for attr_match in expression.matches_with_variables:
            field_name = attr_match.name_from_variable_access_path
            value = attr_match.assigned_variable._value_
            field_type = attr_match.assigned_variable._type_
            field_types[field_name] = field_type

            if isinstance(value, type(Ellipsis)):
                free_slots.append((field_name, field_type))
            else:
                fixed_slots[field_name] = value

        if not free_slots:
            # Nothing to fill — construct directly from fixed values
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()
            return

        # ------------------------------------------------------------------ #
        # Step 2: Serialize world state for LLM context                       #
        # ------------------------------------------------------------------ #
        world_context = serialize_world(self.world)

        # ------------------------------------------------------------------ #
        # Step 3: LLM reasoning — fill all free slots                         #
        # The LLM receives: instruction, action type, free slot names+types,  #
        # fixed slot values, and full world state. It reasons and returns      #
        # concrete values for every free slot.                                 #
        # ------------------------------------------------------------------ #
        resolved = run_slot_filler(
            instruction=self.instruction,
            action_type=expression.type.__name__,
            free_slots=free_slots,
            fixed_slots=fixed_slots,
            world_context=world_context,
            llm=self.llm,
        )

        if resolved is None:
            # LLM failed to produce a valid output — yield nothing
            return

        # ------------------------------------------------------------------ #
        # Step 4: Reconstruct complex objects and coerce enums                #
        # For COMPLEX free slots (dataclasses like GraspDescription), the LLM #
        # returns flat "field.sub_field" keys. Reassemble them into an        #
        # instance and inject CONTEXT sub-fields (Manipulator) from the world.#
        # For simple ENUM slots, coerce the string to the enum member.        #
        # ------------------------------------------------------------------ #
        from llm_reasoner.workflows.slot_filler import (
            _unwrap_type, _is_context_type, _complex_subfields,
        )

        for field_name, field_type in free_slots:
            unwrapped = _unwrap_type(field_type)
            if not (isinstance(unwrapped, type) and dataclasses.is_dataclass(unwrapped)):
                continue  # handled below for simple types
            resolved[field_name] = _reconstruct_complex(
                field_name, unwrapped, resolved, self.world
            )

        # ------------------------------------------------------------------ #
        # Step 5: Write values into the Match variable graph                  #
        # ------------------------------------------------------------------ #
        for field_name, value in resolved.items():
            if "." in field_name:
                continue  # consumed sub-field key, skip
            mapped_var = expression._get_mapped_variable_by_name(field_name)
            if mapped_var is None:
                continue
            field_type = field_types.get(field_name)
            if (
                isinstance(value, str)
                and isinstance(field_type, type)
                and issubclass(field_type, Enum)
            ):
                value = _coerce_enum(value, field_type)
            mapped_var._value_ = value

        expression._update_kwargs_from_literal_values()
        yield expression.construct_instance()


def _reconstruct_complex(
    field_name: str,
    cls: type,
    resolved: Dict[str, Any],
    world: Any,
) -> Any:
    """Build a dataclass instance (e.g. GraspDescription) from flat 'field.sub' keys.

    For each sub-field of *cls*:
    - CONTEXT types (Manipulator, Camera): injected from *world* using the already-
      resolved arm value so the right manipulator is chosen.
    - ENUM sub-fields: coerced from string → enum member.
    - PRIMITIVE sub-fields with defaults: uses the resolved value or falls back to
      the dataclass field default.
    """
    from llm_reasoner.workflows.slot_filler import _unwrap_type, _is_context_type

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = getattr(cls, "__annotations__", {})

    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        sub_type = _unwrap_type(hints.get(f.name, f.type))
        sub_key = f"{field_name}.{f.name}"

        if _is_context_type(sub_type):
            # Inject the matching Manipulator from the world.
            # Look for any resolved arm value to pick the right one.
            arm_value = next(
                (v for k, v in resolved.items() if "arm" in k.lower() and not isinstance(v, str)),
                None,
            )
            manip = _get_manipulator(world, arm_value)
            if manip is not None:
                kwargs[f.name] = manip
            # If no manipulator found, omit and let the dataclass default handle it.

        elif sub_key in resolved:
            val = resolved[sub_key]
            if isinstance(val, str) and isinstance(sub_type, type) and issubclass(sub_type, Enum):
                val = _coerce_enum(val, sub_type)
            elif isinstance(val, str) and sub_type is bool:
                val = val.lower() in ("true", "1", "yes")
            kwargs[f.name] = val

        # Sub-fields not in resolved and not CONTEXT fall through — the dataclass
        # will use its own default if one exists.

    try:
        return cls(**kwargs)
    except Exception as exc:
        logger.warning("_reconstruct_complex: could not build %s(%s): %s", cls.__name__, kwargs, exc)
        return None


def _get_manipulator(world: Any, arm: Any) -> Any:
    """Return the Manipulator for *arm* from the SDT world, or None."""
    try:
        from semantic_digital_twin.robots.abstract_robot import AbstractRobot
        robots = world.get_semantic_annotations_by_type(AbstractRobot)
        if not robots:
            return None
        robot = robots[0]
        if hasattr(robot, "get_manipulator_for_arm") and arm is not None:
            return robot.get_manipulator_for_arm(arm)
        if hasattr(robot, "manipulators"):
            manips = robot.manipulators
            if isinstance(manips, dict) and arm is not None:
                return manips.get(arm) or next(iter(manips.values()), None)
            if isinstance(manips, list) and manips:
                idx = getattr(arm, "value", 0)
                return manips[idx] if idx < len(manips) else manips[0]
    except Exception as exc:
        logger.debug("_get_manipulator: %s", exc)
    return None


def _coerce_enum(value: str, enum_type: type) -> Any:
    """Convert a string returned by the LLM to the matching enum member.

    Tries exact match first, then case-insensitive match.
    Falls back to the first member and logs a warning if nothing matches.
    Mirrors the coercion logic in krrood/pycram_bridge/auto_handler._resolve_enum().
    """
    try:
        return enum_type[value]
    except KeyError:
        pass
    value_upper = value.upper()
    for member in enum_type:
        if member.name.upper() == value_upper:
            return member
    first = next(iter(enum_type))
    logger.warning(
        "_coerce_enum: '%s' is not a valid member of %s %s — falling back to %s",
        value, enum_type.__name__, list(enum_type.__members__), first.name,
    )
    return first
