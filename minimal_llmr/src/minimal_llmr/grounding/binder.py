"""ParameterBinder — dispatch grounding by inspecting WrappedField directly.

bind_one() is the single dispatch point:
  referent (Symbol / Pose / Type[X]) → ReferentResolver → candidates[0]
  discrete (Enum / primitive)        → coerce_enum() or coerce_primitive()
  nested   (dataclass)               → skip (sub-parameters handle it)
"""

from __future__ import annotations

import logging
import typing
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from minimal_llmr.bridge.template import is_referent, is_nested
from minimal_llmr.core.schemas import ParameterInterpretation, ParameterInterpretations, ReferentDescription

if TYPE_CHECKING:
    from minimal_llmr.bridge.template import ActionParameter
    from minimal_llmr.grounding.referent import ReferentResolver

logger = logging.getLogger(__name__)

_UNRESOLVED = object()


def bind_one(
    param: "ActionParameter",
    interpretations: ParameterInterpretations,
    resolver: "ReferentResolver",
    unresolved: Any = _UNRESOLVED,
) -> Any:
    """Resolve one ActionParameter to a concrete Python value.

    Returns *unresolved* when the parameter cannot be bound.
    """
    interp_by_name = {i.param_name: i for i in interpretations.interpretations}
    interp: Optional[ParameterInterpretation] = interp_by_name.get(param.prompt_name)

    if is_nested(param.wf, param.field_type):
        return unresolved

    if is_referent(param.wf, param.field_type):
        return _resolve_referent(param, interp, resolver, unresolved)

    # Discrete: enum or primitive
    if interp is not None and interp.value:
        if isinstance(param.field_type, type) and issubclass(param.field_type, Enum):
            return coerce_enum(interp.value, param.field_type)
        return coerce_primitive(interp.value, param.field_type)

    return unresolved


# ── Referent resolution ────────────────────────────────────────────────────────


def _resolve_referent(
    param: "ActionParameter",
    interp: Optional[ParameterInterpretation],
    resolver: "ReferentResolver",
    unresolved: Any,
) -> Any:
    from minimal_llmr.bridge.world import resolve_symbol_class
    from minimal_llmr.bridge.template import _POSE_NAMES

    if interp is None:
        logger.warning("bind_one: no interpretation for referent param '%s'", param.prompt_name)
        return unresolved

    description = interp.referent_description
    if description is None:
        if interp.value:
            inferred_type = (
                param.field_type.__name__
                if isinstance(param.field_type, type) else None
            )
            description = ReferentDescription(name=interp.value, semantic_type=inferred_type)
        else:
            logger.warning(
                "bind_one: referent param '%s' has neither referent_description nor value",
                param.prompt_name,
            )
            return unresolved

    expected_type = param.field_type if isinstance(param.field_type, type) else None
    result = resolver.resolve(description, expected_type=expected_type)

    if result.warning:
        logger.warning("bind_one: %s", result.warning)

    if not result.candidates:
        logger.warning(
            "bind_one: no candidates for '%s' (name=%r, type=%r)",
            param.prompt_name,
            description.name,
            description.semantic_type,
        )
        return unresolved

    body = result.candidates[0]

    # Pose field: return global_pose instead of the symbol itself
    if isinstance(param.field_type, type):
        if any(cls.__name__ in _POSE_NAMES for cls in param.field_type.__mro__):
            try:
                return body.global_pose
            except AttributeError:
                logger.warning("bind_one: grounded body for '%s' has no global_pose", param.prompt_name)
                return unresolved

    # Type[X] field: return the Symbol subclass itself
    if param.wf is not None and param.wf.is_type_type:
        if description.semantic_type:
            cls = resolve_symbol_class(description.semantic_type)
            if cls is not None:
                return cls

    # Type check
    if expected_type is not None and not isinstance(body, expected_type):
        logger.warning(
            "bind_one: grounded value for '%s' has type %s, expected %s",
            param.prompt_name, type(body).__name__, expected_type.__name__,
        )
        return unresolved

    return body


# ── Discrete coercion ──────────────────────────────────────────────────────────


def coerce_enum(value: str, enum_type: type) -> Any:
    """Convert a string to an enum member (case-insensitive).

    Raises ValueError if no member matches.
    """
    try:
        return enum_type[value]
    except KeyError:
        pass
    value_upper = value.upper()
    for member in enum_type:
        if member.name.upper() == value_upper:
            return member
    raise ValueError(
        f"'{value}' is not a valid member of {enum_type.__name__} "
        f"{list(enum_type.__members__)}."
    )


def coerce_primitive(value: str, field_type: Any) -> Any:
    """Cast an LLM string output to bool, int, float, or str."""
    origin = typing.get_origin(field_type)
    if origin is typing.Union:
        args = [a for a in typing.get_args(field_type) if a is not type(None)]
        unwrapped = args[0] if len(args) == 1 else field_type
    else:
        unwrapped = field_type

    if unwrapped is bool:
        return value.lower() in ("true", "1", "yes")
    if unwrapped is int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if unwrapped is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value
