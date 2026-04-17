"""Small value-conversion helpers for LLM slot outputs."""
from __future__ import annotations

import logging
import typing
from typing import TYPE_CHECKING
from typing_extensions import Any, Dict, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llmr.match_inspection import MatchBinding
    from llmr.pycram_bridge.introspector import FieldKind, PycramIntrospector
    from llmr.schemas.slots import SlotValue
    from llmr.world.grounder import EntityGrounder


def resolve_binding_value(
    binding: "MatchBinding",
    introspector: "PycramIntrospector",
    slot_by_name: Dict[str, "SlotValue"],
    grounder: "EntityGrounder",
    resolved_params: Dict[str, Any],
    unresolved: Any,
) -> Any:
    """Resolve one free KRROOD Match binding from LLM output and SymbolGraph."""
    from llmr.pycram_bridge.introspector import FieldKind

    field_type = binding.field_type
    kind = introspector.classify_type(field_type)

    if kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF):
        slot = slot_by_name.get(binding.prompt_name)
        if slot is None:
            return unresolved
        return resolve_entity_slot(
            slot,
            grounder,
            kind,
            binding.prompt_name,
            expected_type=field_type,
            resolved_params=resolved_params,
            unresolved=unresolved,
        )

    if kind == FieldKind.ENUM:
        slot = slot_by_name.get(binding.prompt_name)
        if slot is not None and slot.value:
            return coerce_enum(slot.value, field_type)
        return unresolved

    if kind == FieldKind.COMPLEX:
        return unresolved

    if kind == FieldKind.PRIMITIVE or kind is None:
        slot = slot_by_name.get(binding.prompt_name)
        if slot is not None and slot.value is not None:
            return coerce_primitive(slot.value, field_type)

    return unresolved


def resolve_entity_slot(
    sv: "SlotValue",
    grounder: "EntityGrounder",
    kind: "FieldKind",
    field_name: str,
    expected_type: Optional[type] = None,
    resolved_params: Optional[Dict[str, Any]] = None,
    unresolved: Any = None,
) -> Any:
    """Ground an ENTITY/POSE/TYPE_REF slot to a Symbol instance via EntityGrounder."""
    from llmr.pycram_bridge.introspector import FieldKind
    from llmr.schemas.entities import EntityDescriptionSchema
    from llmr.world.grounder import (
        ground_expected_entity,
        grounder_can_return_type,
        resolve_symbol_class,
    )

    ed = sv.entity_description
    if ed is not None:
        grounding_ed = ed
    elif sv.value:
        grounding_ed = EntityDescriptionSchema(name=sv.value)
    else:
        logger.warning(
            "resolve_entity_slot: field '%s' has neither entity_description nor value.",
            field_name,
        )
        return unresolved

    grounding = grounder.ground(grounding_ed)
    if grounding.warning:
        logger.warning("Grounding warning for '%s': %s", field_name, grounding.warning)
    if not grounding.bodies:
        if (
            kind == FieldKind.ENTITY
            and isinstance(expected_type, type)
            and not grounder_can_return_type(grounder, expected_type)
        ):
            expected_entity = ground_expected_entity(
                expected_type,
                grounding_ed,
                resolved_params or {},
                symbol_graph=getattr(grounder, "symbol_graph", None),
            )
            if expected_entity is not None:
                return expected_entity
        logger.warning(
            "resolve_entity_slot: no bodies found for field '%s' (name=%r, type=%r).",
            field_name,
            grounding_ed.name,
            grounding_ed.semantic_type,
        )
        return unresolved

    body = grounding.bodies[0]
    if kind == FieldKind.ENTITY and isinstance(expected_type, type):
        if not isinstance(body, expected_type):
            expected_entity = ground_expected_entity(
                expected_type,
                grounding_ed,
                resolved_params or {},
                symbol_graph=getattr(grounder, "symbol_graph", None),
            )
            if expected_entity is not None:
                return expected_entity
            logger.warning(
                "resolve_entity_slot: grounded value for field '%s' has type %s, expected %s.",
                field_name,
                type(body).__name__,
                expected_type.__name__,
            )
            return unresolved

    if kind == FieldKind.POSE:
        try:
            return body.global_pose
        except AttributeError:
            logger.warning("Grounded body for '%s' has no global_pose.", field_name)
            return unresolved

    if kind == FieldKind.TYPE_REF:
        if ed is not None and ed.semantic_type:
            cls = resolve_symbol_class(
                ed.semantic_type,
                symbol_graph=getattr(grounder, "symbol_graph", None),
            )
            if cls is not None:
                return cls
        return body

    return body


def coerce_enum(value: str, enum_type: type) -> Any:
    """Convert a string to the matching enum member."""
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
        "coerce_enum: '%s' is not a valid member of %s %s; falling back to %s.",
        value,
        enum_type.__name__,
        list(enum_type.__members__),
        first.name,
    )
    return first


def coerce_primitive(value: str, field_type: Any) -> Any:
    """Cast an LLM string output to bool, int, float, or str as required."""
    origin = typing.get_origin(field_type)
    if origin is typing.Union:
        args = [arg for arg in typing.get_args(field_type) if arg is not type(None)]
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
