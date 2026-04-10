"""AutoActionHandler — builds a pycram action instance from ActionSlotSchema.

Works entirely from pycram class introspection; no per-action handler code needed.

Resolution pipeline (per field)
--------------------------------
ENTITY   → EntityGrounder.ground(entity_desc_matching_role).bodies[0]
POSE     → same grounding, then .global_pose on the body
ENUM     → coerce schema.parameters[field] string → enum member;
           if missing, call EnumResolver (LLM)
COMPLEX  → recursively build from sub-fields using the flat parameters dict
CONTEXT  → context[field.name] or context["manipulators"][resolved_arm]
PRIMITIVE → schema.parameters.get(field.name, field.default)
TYPE_REF → resolve class name from SymbolGraph class diagram
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing_extensions import Any, Dict, List, Optional

from krrood.llmr_decoupled.pipeline.clarification import ClarificationNeededError, ClarificationRequest
from krrood.llmr_decoupled.pipeline.dispatcher import ActionHandler, ActionSpec
from krrood.llmr_decoupled.pipeline.entity_grounder import body_display_name
from krrood.llmr_decoupled.pycram_bridge.introspector import NO_DEFAULT, ActionSchema, FieldKind, FieldSpec
from krrood.llmr_decoupled.pycram_bridge.enum_resolver import EnumResolver
from krrood.llmr_decoupled.workflows.schemas.common import ActionSlotSchema, EntityDescriptionSchema

logger = logging.getLogger(__name__)


@dataclass
class AutoActionHandler(ActionHandler):
    """Generic action handler driven entirely by pycram class introspection.

    Registered with :class:`~krrood.llmr_decoupled.pipeline.dispatcher.ActionDispatcher`
    by :class:`~krrood.llmr_decoupled.pycram_bridge.PycramBridge` for each discovered action.

    :param action_schema: Introspected description of the pycram action.
    :param grounder: Shared EntityGrounder from the pipeline.
    :param context: Caller-supplied dict. Expected keys:

        - ``world_context`` (str): Serialized world state for LLM calls.
        - ``manipulators`` (Dict[str, Manipulator]): Arm-name → Manipulator instance.
          E.g. ``{"LEFT": left_manip, "RIGHT": right_manip}``.
          Single-manipulator robots can use ``{"manipulator": manip}``.
    """

    action_schema: ActionSchema = field(default=None)
    _enum_resolver: EnumResolver = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._enum_resolver = EnumResolver()

    # ── Main entry point ───────────────────────────────────────────────────────

    def execute(self, schema: ActionSlotSchema) -> ActionSpec:
        """Resolve all fields and return an :class:`ActionSpec` with resolved kwargs.

        The returned ``ActionSpec.parameters`` contains the fully resolved pycram
        objects (enum instances, GraspDescription, etc.) and ``grounded_entities``
        contains grounded Symbol instances, keyed by their pycram field name.

        Call :func:`~krrood.llmr_decoupled.pycram_bridge.PycramBridge.to_pycram_action` to
        convert the spec into a concrete pycram action instance.
        """
        resolved_params: Dict[str, Any] = {}
        grounded_entities: Dict[str, Any] = {}
        world_context: str = self.context.get("world_context", "")

        logger.debug(
            "AutoActionHandler.execute: action=%s entities=%s parameters=%s",
            self.action_schema.action_type,
            [(e.role, e.name) for e in schema.entities],
            list(schema.parameters.keys()),
        )

        for fspec in self.action_schema.fields:
            value = self._resolve_field(
                fspec, schema, resolved_params, world_context
            )
            if value is dataclasses.MISSING:
                if not fspec.is_optional:
                    raise RuntimeError(
                        f"Required field '{fspec.name}' could not be resolved "
                        f"for {self.action_schema.action_type}."
                    )
                continue  # optional unresolved field — omit (use class default)

            if fspec.kind in (FieldKind.ENTITY, FieldKind.POSE):
                grounded_entities[fspec.name] = value
            else:
                resolved_params[fspec.name] = value

        return ActionSpec(
            action_type=self.action_schema.action_type,
            parameters=resolved_params,
            grounded_entities=grounded_entities,
        )

    # ── Per-field resolution ───────────────────────────────────────────────────

    def _resolve_field(
        self,
        fspec: FieldSpec,
        schema: ActionSlotSchema,
        resolved_so_far: Dict[str, Any],
        world_context: str,
    ) -> Any:
        """Dispatch to the correct resolver based on *fspec.kind*."""
        if fspec.kind == FieldKind.ENTITY:
            return self._resolve_entity(fspec, schema)

        if fspec.kind == FieldKind.POSE:
            return self._resolve_pose(fspec, schema)

        if fspec.kind == FieldKind.ENUM:
            return self._resolve_enum(fspec, schema, resolved_so_far, world_context)

        if fspec.kind == FieldKind.COMPLEX:
            return self._resolve_complex(fspec, schema, resolved_so_far, world_context)

        if fspec.kind == FieldKind.CONTEXT:
            return self._resolve_from_context(fspec, resolved_so_far)

        if fspec.kind == FieldKind.TYPE_REF:
            return self._resolve_type_ref(fspec, schema)

        if fspec.kind == FieldKind.PRIMITIVE:
            val = schema.parameters.get(fspec.name, dataclasses.MISSING)
            if val is dataclasses.MISSING and fspec.default is not NO_DEFAULT:
                return fspec.default
            return val

        return dataclasses.MISSING

    # ── ENTITY ─────────────────────────────────────────────────────────────────

    def _resolve_entity(self, fspec: FieldSpec, schema: ActionSlotSchema) -> Any:
        entity_desc = self._find_entity(fspec.name, schema.entities)
        if entity_desc is None:
            if fspec.is_optional:
                return dataclasses.MISSING
            raise ClarificationNeededError(
                ClarificationRequest(
                    entity_name=fspec.name,
                    entity_role=fspec.name,
                    available_names=[],
                    message=(
                        f"No entity with role '{fspec.name}' was found in the instruction. "
                        f"Please specify the {fspec.name.replace('_', ' ')}."
                    ),
                )
            )

        grounding = self.grounder.ground(entity_desc)
        if grounding.warning:
            logger.warning("Grounding warning for '%s': %s", fspec.name, grounding.warning)
        if not grounding.bodies:
            raise ClarificationNeededError(
                ClarificationRequest(
                    entity_name=entity_desc.name or fspec.name,
                    entity_role=fspec.name,
                    available_names=[],
                    message=(
                        f"Cannot find '{entity_desc.name}' (role={fspec.name}) in the world. "
                        "Check that the object exists."
                    ),
                )
            )

        body = grounding.bodies[0] if len(grounding.bodies) == 1 else grounding.bodies
        logger.debug(
            "Grounded '%s' → %s",
            fspec.name,
            body_display_name(body) if not isinstance(body, list) else [body_display_name(b) for b in body],
        )
        return body

    # ── POSE ───────────────────────────────────────────────────────────────────

    def _resolve_pose(self, fspec: FieldSpec, schema: ActionSlotSchema) -> Any:
        """Ground the entity and return its ``global_pose``."""
        body = self._resolve_entity(fspec, schema)
        if body is dataclasses.MISSING:
            return dataclasses.MISSING
        try:
            if isinstance(body, list):
                return body[0].global_pose
            return body.global_pose
        except AttributeError:
            logger.warning("Grounded body for '%s' has no global_pose.", fspec.name)
            return dataclasses.MISSING

    # ── ENUM ───────────────────────────────────────────────────────────────────

    def _resolve_enum(
        self,
        fspec: FieldSpec,
        schema: ActionSlotSchema,
        resolved_so_far: Dict[str, Any],
        world_context: str,
    ) -> Any:
        raw = schema.parameters.get(fspec.name)

        if raw is None:
            if fspec.is_optional and fspec.default is not NO_DEFAULT:
                return fspec.default
            # Ask LLM to resolve
            logger.debug("Calling EnumResolver for required field '%s'.", fspec.name)
            member_name = self._enum_resolver.resolve(
                param_name=fspec.name,
                description=fspec.docstring,
                members=fspec.enum_members,
                world_context=world_context,
                known_params={
                    k: (v.name if hasattr(v, "name") else str(v))
                    for k, v in resolved_so_far.items()
                },
            )
            if member_name is None:
                if fspec.is_optional:
                    return dataclasses.MISSING
                raise RuntimeError(
                    f"EnumResolver could not resolve required field '{fspec.name}'."
                )
            raw = member_name

        # Convert string → enum member
        try:
            return fspec.raw_type[raw]
        except (KeyError, TypeError):
            # Try case-insensitive match
            raw_upper = str(raw).upper()
            for member in fspec.raw_type:
                if member.name.upper() == raw_upper:
                    return member
            logger.warning(
                "Cannot coerce '%s' to %s; using first member.", raw, fspec.raw_type
            )
            return next(iter(fspec.raw_type))

    # ── COMPLEX ────────────────────────────────────────────────────────────────

    def _resolve_complex(
        self,
        fspec: FieldSpec,
        schema: ActionSlotSchema,
        resolved_so_far: Dict[str, Any],
        world_context: str,
    ) -> Any:
        """Recursively build a complex type (e.g. GraspDescription) from sub-fields."""
        if not fspec.sub_fields:
            logger.warning(
                "COMPLEX field '%s' has no sub_fields — cannot build.", fspec.name
            )
            return dataclasses.MISSING

        kwargs: Dict[str, Any] = {}
        for sub in fspec.sub_fields:
            value = self._resolve_field(sub, schema, {**resolved_so_far, **kwargs}, world_context)
            if value is not dataclasses.MISSING:
                kwargs[sub.name] = value

        try:
            return fspec.raw_type(**kwargs)
        except Exception as exc:
            if fspec.is_optional:
                logger.warning(
                    "Cannot build COMPLEX field '%s': %s — skipping.", fspec.name, exc
                )
                return dataclasses.MISSING
            raise RuntimeError(
                f"Cannot construct {fspec.raw_type.__name__} for field '{fspec.name}': {exc}"
            ) from exc

    # ── CONTEXT ────────────────────────────────────────────────────────────────

    def _resolve_from_context(
        self, fspec: FieldSpec, resolved_so_far: Dict[str, Any]
    ) -> Any:
        """Look up a context-injected value.

        For ``Manipulator`` fields, tries to look up by arm name first::

            context["manipulators"]["RIGHT"] → right manipulator
        """
        # Direct lookup
        val = self.context.get(fspec.name)
        if val is not None:
            return val

        # Arm-specific manipulator lookup
        type_name = fspec.raw_type.__name__ if isinstance(fspec.raw_type, type) else ""
        if "manipulator" in type_name.lower() or "manipulator" in fspec.name.lower():
            manipulators: Dict[str, Any] = self.context.get("manipulators", {})
            # Try to find the arm that was resolved
            for k, v in resolved_so_far.items():
                if "arm" in k.lower() and hasattr(v, "name"):
                    manip = manipulators.get(v.name)
                    if manip is not None:
                        return manip
            # Fallback: first available manipulator
            if manipulators:
                return next(iter(manipulators.values()))
            # Last resort: single "manipulator" key
            return self.context.get("manipulator")

        logger.debug(
            "Context field '%s' not found in context dict (keys: %s).",
            fspec.name, list(self.context.keys()),
        )
        return dataclasses.MISSING

    # ── TYPE_REF ───────────────────────────────────────────────────────────────

    def _resolve_type_ref(self, fspec: FieldSpec, schema: ActionSlotSchema) -> Any:
        """Resolve ``Type[X]`` fields — returns the class itself, found via SymbolGraph."""
        # Look for a matching entity description with role = field.name
        entity_desc = self._find_entity(fspec.name, schema.entities)
        type_hint = (
            entity_desc.semantic_type if entity_desc else schema.parameters.get(fspec.name)
        )
        if not type_hint:
            if fspec.is_optional:
                return dataclasses.MISSING
            raise RuntimeError(
                f"TYPE_REF field '{fspec.name}' has no semantic_type in entity description."
            )

        from krrood.llmr_decoupled.pipeline.entity_grounder import resolve_symbol_class
        cls = resolve_symbol_class(type_hint)
        if cls is None and fspec.is_optional:
            return dataclasses.MISSING
        if cls is None:
            raise RuntimeError(
                f"Cannot resolve '{type_hint}' to a Symbol subclass for field '{fspec.name}'."
            )
        return cls

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_entity(
        role: str, entities: List[EntityDescriptionSchema]
    ) -> Optional[EntityDescriptionSchema]:
        """Return the first entity whose role matches *role*.

        Falls back through progressively looser matching so that LLM role-naming
        variations (e.g. ``'object'`` vs ``'object_designator'``) still resolve:

        1. Exact match.
        2. Normalised substring match (strips underscores, case-insensitive).
        3. Single-entity shortcut — if only one entity exists, assume it fills
           this role (avoids failure when the LLM omits the role field).
        """
        # 1. Exact
        for e in entities:
            if e.role == role:
                return e

        # 2. Normalised substring
        role_norm = role.lower().replace("_", "")
        for e in entities:
            entity_role_norm = (e.role or "").lower().replace("_", "")
            if entity_role_norm and (
                entity_role_norm in role_norm or role_norm in entity_role_norm
            ):
                logger.debug(
                    "_find_entity: fuzzy match '%s' → role='%s'", role, e.role
                )
                return e

        # 3. Single-entity shortcut
        if len(entities) == 1:
            logger.debug(
                "_find_entity: single-entity fallback for role '%s'.", role
            )
            return entities[0]

        return None
