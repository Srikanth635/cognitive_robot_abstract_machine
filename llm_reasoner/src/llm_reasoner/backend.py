"""
LLMBackend — a GenerativeBackend that uses an LLM as a reasoning engine.

Design
------
1. No `world: Any` field — world context comes from SymbolGraph (via groundable_type)
   or a caller-injected world_context_provider callable.

2. No hard SDT imports anywhere in this module.
   Robot components (Manipulator, Camera) come from the `context` dict supplied
   by the caller.

3. Entity slots are symbolically grounded via EntityGrounder (llm_reasoner.world.grounder)
   after the LLM returns an EntityDescriptionSchema.

4. Slot filling is driven by PycramIntrospector (llm_reasoner.pycram_bridge.introspector)
   so each free slot is resolved the right way (ENTITY grounding, ENUM coercion,
   COMPLEX reconstruction, CONTEXT injection, PRIMITIVE coercion).

Variable assignment pattern (mirrors ProbabilisticBackend):
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from krrood.symbol_graph.symbol_graph import Symbol

T = TypeVar("T")

# Sentinel — means "could not resolve, skip this field"
_UNRESOLVED = object()


# ── LLMBackend ─────────────────────────────────────────────────────────────────

@dataclass
class LLMBackend(GenerativeBackend):
    """
    A GenerativeBackend that uses an LLM to fill underspecified Match slots.

    :param instruction:           Natural-language instruction describing the action.
    :param llm:                   LangChain BaseChatModel (injected, no global singletons).
    :param groundable_type:       Symbol subclass for scene objects (e.g. Body from SDT).
                                  Passed by the caller — never imported here.
    :param context:               Caller-supplied dict for robot components:
                                    - ``"manipulators"`` : {arm_name: Manipulator}
                                    - ``"manipulator"``  : single Manipulator fallback
    :param world_context_provider: Optional callable returning a world-context string.
                                  When provided, replaces SymbolGraph serialization.
    """

    instruction: str
    llm: "BaseChatModel"
    groundable_type: "type[Symbol]"
    context: Dict[str, Any] = field(default_factory=dict)
    world_context_provider: Optional[Callable[[], str]] = field(default=None)

    # ── Core interface ─────────────────────────────────────────────────────────

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        """Fill all free slots in the Match expression via LLM reasoning."""

        # ── 1. Parse free / fixed slots from the Match variable graph ──────────
        free_slots: List[Tuple[str, Any]] = []
        fixed_slots: Dict[str, Any] = {}
        field_types: Dict[str, Any] = {}
        # name_from_variable_access_path may return 'ClassName.field_name'.
        # We normalise to bare field names for all lookups but keep the full
        # path so we can call _get_mapped_variable_by_name() with it later.
        _full_name_map: Dict[str, str] = {}   # short_name → full access path

        for attr_match in expression.matches_with_variables:
            fname_raw = attr_match.name_from_variable_access_path
            fname = _field_short_name(fname_raw)   # strip 'ClassName.' prefix
            value = attr_match.assigned_variable._value_
            ftype = attr_match.assigned_variable._type_
            field_types[fname] = ftype
            _full_name_map[fname] = fname_raw

            if isinstance(value, type(Ellipsis)):
                free_slots.append((fname, ftype))
            else:
                fixed_slots[fname] = value

        if not free_slots:
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()
            return

        # ── 2. World context ───────────────────────────────────────────────────
        world_context = self._get_world_context()

        # ── 3. Introspect action class for field metadata ──────────────────────
        from llm_reasoner.pycram_bridge.introspector import FieldKind, PycramIntrospector
        try:
            action_schema = PycramIntrospector().introspect(expression.type)
            field_specs = {f.name: f for f in action_schema.fields}
        except Exception as exc:
            logger.debug(
                "LLMBackend: introspection failed for %s: %s — field-kind fallback active.",
                expression.type.__name__, exc,
            )
            field_specs = {}

        # ── 4. Separate CONTEXT fields (injected from self.context, not LLM) ──
        context_field_names = {
            name
            for name, _ in free_slots
            if (
                (fspec := field_specs.get(name)) is not None
                and fspec.kind == FieldKind.CONTEXT
            )
        }
        llm_free_slot_names = [
            name for name, _ in free_slots if name not in context_field_names
        ]

        # ── 5. Run the slot filler (LLM call with dynamic prompt) ─────────────
        output = None
        if llm_free_slot_names:
            from llm_reasoner.reasoning.slot_filler import run_slot_filler
            output = run_slot_filler(
                instruction=self.instruction,
                action_cls=expression.type,
                free_slot_names=llm_free_slot_names,
                fixed_slots=fixed_slots,
                world_context=world_context,
                llm=self.llm,
            )
            if output is None:
                logger.warning(
                    "LLMBackend: slot filler returned None for %s — yielding nothing.",
                    expression.type.__name__,
                )
                return

        # ── 6. Resolve each free slot ──────────────────────────────────────────
        from llm_reasoner.world.grounder import EntityGrounder
        grounder = EntityGrounder(self.groundable_type)

        slot_by_name = (
            {sv.field_name: sv for sv in output.slots} if output else {}
        )
        resolved_params: Dict[str, Any] = {}  # used for arm → manipulator lookup

        for field_name, field_type in free_slots:
            fspec = field_specs.get(field_name)
            kind = fspec.kind if fspec is not None else None

            resolved = _UNRESOLVED

            if field_name in context_field_names:
                resolved = _resolve_context_field(
                    field_name=field_name,
                    fspec=fspec,
                    context=self.context,
                    resolved_params=resolved_params,
                )

            elif kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF):
                sv = slot_by_name.get(field_name)
                if sv is not None:
                    resolved = _resolve_entity_slot(sv, grounder, kind, field_name)

            elif kind == FieldKind.ENUM:
                sv = slot_by_name.get(field_name)
                if sv is not None and sv.value:
                    resolved = _coerce_enum(sv.value, fspec.raw_type)

            elif kind == FieldKind.COMPLEX:
                if fspec is not None and fspec.sub_fields:
                    resolved = _reconstruct_complex(
                        field_name=field_name,
                        fspec=fspec,
                        slot_by_name=slot_by_name,
                        context=self.context,
                        resolved_params=resolved_params,
                    )

            elif kind == FieldKind.PRIMITIVE or kind is None:
                sv = slot_by_name.get(field_name)
                if sv is not None and sv.value is not None:
                    resolved = _coerce_primitive(sv.value, field_type)

            if resolved is _UNRESOLVED:
                logger.debug(
                    "LLMBackend: field '%s' unresolved — leaving as default.",
                    field_name,
                )
                continue

            resolved_params[field_name] = resolved
            try:
                # Use the original full access path (e.g. 'PickUpAction.arm') since
                # _get_mapped_variable_by_name matches on name_from_variable_access_path.
                mapped_var = expression._get_mapped_variable_by_name(
                    _full_name_map.get(field_name, field_name)
                )
                mapped_var._value_ = resolved
            except Exception as exc:
                logger.warning(
                    "LLMBackend: cannot set field '%s': %s", field_name, exc
                )

        expression._update_kwargs_from_literal_values()
        yield expression.construct_instance()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_world_context(self) -> str:
        if self.world_context_provider is not None:
            try:
                return self.world_context_provider()
            except Exception as exc:
                logger.warning(
                    "LLMBackend: world_context_provider raised %s — falling back to SymbolGraph.",
                    exc,
                )
        from llm_reasoner.world.serializer import serialize_world_from_symbol_graph
        return serialize_world_from_symbol_graph(self.groundable_type)


# ── Per-slot resolvers (module-level, reusable) ────────────────────────────────

def _resolve_entity_slot(
    sv: Any,
    grounder: Any,
    kind: Any,
    field_name: str,
) -> Any:
    """Ground an ENTITY/POSE/TYPE_REF slot to a Symbol instance via EntityGrounder."""
    from llm_reasoner.pycram_bridge.introspector import FieldKind
    from llm_reasoner.world.grounder import resolve_symbol_class
    from llm_reasoner.schemas.entities import EntityDescriptionSchema

    # Build EntityDescriptionSchema for the grounder — use the LLM's entity
    # description if available, otherwise fall back to the value string.
    ed = sv.entity_description
    if ed is not None:
        grounding_ed = ed  # already an EntityDescriptionSchema from the LLM output
    elif sv.value:
        grounding_ed = EntityDescriptionSchema(name=sv.value)
    else:
        logger.warning(
            "_resolve_entity_slot: field '%s' has neither entity_description nor value.",
            field_name,
        )
        return _UNRESOLVED

    grounding = grounder.ground(grounding_ed)
    if grounding.warning:
        logger.warning("Grounding warning for '%s': %s", field_name, grounding.warning)
    if not grounding.bodies:
        logger.warning(
            "_resolve_entity_slot: no bodies found for field '%s' (name=%r, type=%r).",
            field_name, grounding_ed.name, grounding_ed.semantic_type,
        )
        return _UNRESOLVED

    body = grounding.bodies[0]

    if kind == FieldKind.POSE:
        try:
            return body.global_pose
        except AttributeError:
            logger.warning("Grounded body for '%s' has no global_pose.", field_name)
            return _UNRESOLVED

    if kind == FieldKind.TYPE_REF:
        # TYPE_REF fields expect the *class* (e.g. Type[SemanticAnnotation]),
        # resolved from SymbolGraph class diagram.
        if ed is not None and ed.semantic_type:
            cls = resolve_symbol_class(ed.semantic_type)
            if cls is not None:
                return cls
        return body  # fallback: return the instance

    return body


def _reconstruct_complex(
    field_name: str,
    fspec: Any,
    slot_by_name: Dict[str, Any],
    context: Dict[str, Any],
    resolved_params: Dict[str, Any],
) -> Any:
    """Build a complex dataclass (e.g. GraspDescription) from dotted SlotValue entries.

    For each sub-field:
      - CONTEXT kind   → inject from context dict (Manipulator, Camera, etc.)
      - ENUM kind      → coerce string from the dotted SlotValue
      - PRIMITIVE kind → use string value from the dotted SlotValue
      - Missing optional sub-fields → let the dataclass default handle them
    """
    from llm_reasoner.pycram_bridge.introspector import FieldKind

    kwargs: Dict[str, Any] = {}
    for sub in fspec.sub_fields:
        sub_key = f"{field_name}.{sub.name}"

        if sub.kind == FieldKind.CONTEXT:
            val = _resolve_context_field(
                field_name=sub.name,
                fspec=sub,
                context=context,
                resolved_params=resolved_params,
            )
            if val is not _UNRESOLVED:
                kwargs[sub.name] = val
            continue  # never fall through to slot_by_name for CONTEXT fields

        if sub_key not in slot_by_name:
            continue  # missing optional sub-field — use dataclass default

        sv = slot_by_name[sub_key]
        raw_val = sv.value
        if raw_val is None:
            continue
        if sub.kind == FieldKind.ENUM:
            kwargs[sub.name] = _coerce_enum(raw_val, sub.raw_type)
        elif sub.kind == FieldKind.PRIMITIVE:
            kwargs[sub.name] = _coerce_primitive(raw_val, sub.raw_type)
        else:
            kwargs[sub.name] = raw_val

    try:
        return fspec.raw_type(**kwargs)
    except Exception as exc:
        logger.warning(
            "_reconstruct_complex: cannot build %s for '%s': %s",
            fspec.raw_type.__name__, field_name, exc,
        )
        return _UNRESOLVED


def _resolve_context_field(
    field_name: str,
    fspec: Optional[Any],
    context: Dict[str, Any],
    resolved_params: Dict[str, Any],
) -> Any:
    """Return a context-injected value (Manipulator, Camera, etc.).

    No SDT imports — robot components are looked up by name convention in the
    caller-supplied context dict.
    """
    # Direct key lookup
    val = context.get(field_name)
    if val is not None:
        return val

    # Arm-specific manipulator lookup
    type_name = ""
    if fspec is not None and isinstance(fspec.raw_type, type):
        type_name = fspec.raw_type.__name__

    if "manipulator" in type_name.lower() or "manipulator" in field_name.lower():
        manipulators: Dict[str, Any] = context.get("manipulators", {})
        # Try to match by already-resolved arm value
        for k, v in resolved_params.items():
            if "arm" in k.lower() and hasattr(v, "name"):
                manip = manipulators.get(v.name)
                if manip is not None:
                    return manip
        # Fallback: first available manipulator
        if manipulators:
            return next(iter(manipulators.values()))
        # Last resort: single "manipulator" key
        return context.get("manipulator") or _UNRESOLVED

    logger.debug(
        "_resolve_context_field: '%s' not found in context (keys: %s).",
        field_name, list(context.keys()),
    )
    return _UNRESOLVED


def _coerce_enum(value: str, enum_type: type) -> Any:
    """Convert a string to the matching enum member (exact, then case-insensitive)."""
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
        "_coerce_enum: '%s' is not a valid member of %s %s — falling back to %s.",
        value, enum_type.__name__, list(enum_type.__members__), first.name,
    )
    return first


def _coerce_primitive(value: str, field_type: Any) -> Any:
    """Coerce a string value to the expected primitive Python type."""
    unwrapped = _unwrap_field_type(field_type)
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
    return value  # str or unknown → return as-is


def _unwrap_field_type(t: Any) -> Any:
    """Strip Optional[X] / Union[X, None] → X."""
    import typing as _typing
    if _typing.get_origin(t) is _typing.Union:
        args = [a for a in _typing.get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


def _field_short_name(name: str) -> str:
    """Strip 'ClassName.' prefix from the EQL access path name.

    ``name_from_variable_access_path`` returns the last element of the EQL
    access path, which can be prefixed with the action class name
    (e.g. ``'PickUpAction.object_designator'``).  PycramIntrospector and the
    LLM both work with bare field names (``'object_designator'``), so we strip
    everything up to and including the last dot.
    """
    return name.rsplit(".", 1)[-1] if "." in name else name
