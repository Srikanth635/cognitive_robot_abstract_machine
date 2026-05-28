"""Type bridge — converts between KRROOD/PyCRAM type representations and LLM-facing JSON.

Merged from: resolution/action_match.py + resolution/deserializer.py

Responsibilities:
  - Snapshot a KRROOD Match expression into a plain ActionTemplate (action_match logic)
  - Bind resolved values back into the Match for KRROOD construction
  - Hydrate LLM JSON output into native PyCRAM Python objects (deserializer logic)
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
import sys
from dataclasses import MISSING, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union, get_args, get_origin

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import Symbol
from krrood.utils import own_dataclass_fields, get_default_values_for_dataclass, is_builtin_type

from pycram.view_manager import ViewManager

from agentic_llmr.platform.world import find_body_by_name, get_active_world

if TYPE_CHECKING:
    from krrood.class_diagrams.wrapped_field import WrappedField

logger = logging.getLogger(__name__)

_UNRESOLVED = object()
_POSE_NAMES: frozenset[str] = frozenset({"Pose", "HomogeneousTransformationMatrix"})


# ── ActionParameter / ActionTemplate data structures ──────────────────────────


@dataclass
class ActionParameter:
    """One leaf variable in a Match expression."""

    attribute_name: str       # leaf field name, e.g. 'arm'
    prompt_name: str          # root-stripped path used in prompts
    field_type: Any           # resolved Python type from the KRROOD variable
    value: Any                # current value; _UNRESOLVED when free
    is_free: bool
    is_optional_default: bool = False  # dataclass default not present in the Match
    _variable: Any = field(repr=False, default=None)
    wf: Optional["WrappedField"] = field(default=None, repr=False)


@dataclass
class ActionTemplate:
    """Plain-data snapshot of a KRROOD Match expression."""

    action_type: type
    action_name: str
    parameters: list[ActionParameter]
    _expression: Any = field(repr=False)

    @property
    def free_parameters(self) -> list[ActionParameter]:
        """Parameters the agent must resolve (value is _UNRESOLVED)."""
        return [p for p in self.parameters if p.is_free]

    @property
    def free_parameter_names(self) -> list[str]:
        """Prompt-facing names of all free parameters."""
        return [p.prompt_name for p in self.free_parameters]

    @property
    def fixed_bindings(self) -> dict[str, Any]:
        """Prompt-name → value map for parameters already bound in the Match."""
        return {
            p.prompt_name: p.value
            for p in self.parameters
            if not p.is_free and not p.is_optional_default
        }

    @property
    def optional_default_parameters(self) -> list[ActionParameter]:
        """Parameters with dataclass defaults not present in the Match expression."""
        return [p for p in self.parameters if p.is_optional_default]

    @property
    def optional_default_bindings(self) -> dict[str, Any]:
        """Prompt-name → default-value map for optional parameters."""
        return {p.prompt_name: p.value for p in self.parameters if p.is_optional_default}


# ── Type predicates ────────────────────────────────────────────────────────────


def is_referent(wf: Optional["WrappedField"], field_type: Any) -> bool:
    """True when the parameter resolves to a Symbol instance via SymbolGraph."""
    if wf is not None:
        if wf.is_type_type:
            return True
        if wf.is_enum or wf.is_builtin_type:
            return False
        endpoint = wf.type_endpoint
    else:
        endpoint = field_type

    if not isinstance(endpoint, type):
        return False
    if any(cls.__name__ in _POSE_NAMES for cls in endpoint.__mro__):
        return True
    try:
        return issubclass(endpoint, Symbol)
    except TypeError:
        return False


def is_nested(wf: Optional["WrappedField"], field_type: Any) -> bool:
    """True when the parameter is a non-Symbol dataclass (sub-fields handle resolution)."""
    if wf is not None:
        if wf.is_enum or wf.is_builtin_type or wf.is_type_type:
            return False
        endpoint = wf.type_endpoint
    else:
        endpoint = field_type

    if not isinstance(endpoint, type) or not dataclasses.is_dataclass(endpoint):
        return False
    try:
        return not issubclass(endpoint, Symbol)
    except TypeError:
        return True


# ── KRROOD Match snapshot ──────────────────────────────────────────────────────


def _extract_match_parameters(
    match: Any,
    action_cls: type,
    wf_by_name: dict,
) -> list:
    """Build ActionParameter list from the explicitly bound/free variables in a Match expression."""
    return [
        ActionParameter(
            attribute_name=(attr_name := attr_match.attribute_name),
            prompt_name=_strip_root(attr_match.name_from_variable_access_path, action_cls),
            field_type=(variable := attr_match.assigned_variable)._type_,
            value=(value := _read_variable(variable)),
            is_free=value is _UNRESOLVED,
            _variable=variable,
            wf=wf_by_name.get(attr_name),
        )
        for attr_match in match.matches_with_variables
    ]


def _collect_required_free_parameters(
    action_cls: type,
    wf_by_name: dict,
    captured_attr_names: set,
) -> list:
    """Return free ActionParameters for required fields absent from the Match expression."""
    result = []
    for name in sorted(_required_field_names(action_cls) - captured_attr_names):
        wf = wf_by_name.get(name)
        field_type = wf.type_endpoint if wf is not None else object
        result.append(ActionParameter(
            attribute_name=name,
            prompt_name=name,
            field_type=field_type,
            value=_UNRESOLVED,
            is_free=True,
            _variable=None,
            wf=wf,
        ))
    return result


def _collect_top_level_defaults(
    action_cls: type,
    wf_by_name: dict,
    all_prompt_names: set,
) -> list:
    """Return optional-default ActionParameters for own fields with defaults not already in the template."""
    result = []
    top_defaults = get_default_values_for_dataclass(action_cls)
    for wf in wf_by_name.values():
        if wf.name in all_prompt_names or wf.name not in top_defaults:
            continue
        val = top_defaults[wf.name]
        result.append(ActionParameter(
            attribute_name=wf.name,
            prompt_name=wf.name,
            field_type=type(val),
            value=val,
            is_free=False,
            is_optional_default=True,
            wf=wf,
        ))
    return result


def _collect_nested_defaults(
    parameters: list,
    wf_by_name: dict,
    all_prompt_names: set,
) -> list:
    """Return optional-default ActionParameters for fields inside nested dataclass types.

    Scans dotted prompt names in *parameters* to find nested types, then surfaces
    any of their defaulted fields not already present in *all_prompt_names*.
    """
    result = []
    seen_prefixes: set = set()
    for p in list(parameters):
        parts = p.prompt_name.split(".")
        if len(parts) < 2 or parts[0] in seen_prefixes:
            continue
        prefix = parts[0]
        seen_prefixes.add(prefix)
        top_wf = wf_by_name.get(prefix)
        if top_wf is None:
            continue
        try:
            nested_own = {f.name for f in own_dataclass_fields(top_wf.type_endpoint)}
            nested_wc = ClassDiagram([top_wf.type_endpoint]).get_wrapped_class(top_wf.type_endpoint)
            nested_wf_by_name = {wf.name: wf for wf in nested_wc.fields if wf.name in nested_own}
            nested_defaults = get_default_values_for_dataclass(top_wf.type_endpoint)
            for nested_wf in nested_wf_by_name.values():
                dotted = f"{prefix}.{nested_wf.name}"
                if dotted in all_prompt_names or nested_wf.name not in nested_defaults:
                    continue
                val = nested_defaults[nested_wf.name]
                result.append(ActionParameter(
                    attribute_name=nested_wf.name,
                    prompt_name=dotted,
                    field_type=type(val),
                    value=val,
                    is_free=False,
                    is_optional_default=True,
                    wf=nested_wf,
                ))
        except Exception:
            pass
    return result


def snapshot_match(match: Any) -> ActionTemplate:
    """Snapshot *match* into an ActionTemplate, attaching WrappedField to each parameter.

    Also discovers own dataclass fields with defaults that were not included in the
    Match — for both the top-level action class and nested dataclass types — and
    appends them as optional-default parameters so they appear in the LLM context.
    """
    action_cls = match.type
    try:
        own_names = {f.name for f in own_dataclass_fields(action_cls)}
        wc = ClassDiagram([action_cls]).get_wrapped_class(action_cls)
        wf_by_name: dict[str, Any] = {wf.name: wf for wf in wc.fields if wf.name in own_names}
    except Exception:
        own_names, wf_by_name = set(), {}

    parameters = _extract_match_parameters(match, action_cls, wf_by_name)
    parameters += _collect_required_free_parameters(
        action_cls, wf_by_name, {p.attribute_name for p in parameters}
    )
    # all_prompt_names is snapshotted here — both default collectors use this same set
    # (preserves original invariant: top-level defaults are not excluded from nested scan)
    all_prompt_names = {p.prompt_name for p in parameters}
    parameters += _collect_top_level_defaults(action_cls, wf_by_name, all_prompt_names)
    parameters += _collect_nested_defaults(parameters, wf_by_name, all_prompt_names)

    return ActionTemplate(
        action_type=action_cls,
        action_name=getattr(action_cls, "__name__", str(action_cls)),
        parameters=parameters,
        _expression=match,
    )


def bind_parameter(param: ActionParameter, value: Any) -> bool:
    """Write *value* into the KRROOD variable behind *param*."""
    try:
        param._variable._value_ = value
        param.value = value
        param.is_free = False
        return True
    except Exception as exc:
        logger.warning("bind_parameter: cannot set '%s': %s", param.attribute_name, exc)
        return False


def underspecified_match(action_cls: type) -> Match:
    """Return Match(action_cls) with every required public field left free (...)."""
    match = Match(action_cls)
    try:
        own_names = _required_field_names(action_cls)
        wc = ClassDiagram([action_cls]).get_wrapped_class(action_cls)
        kwargs = {
            wf.name: _free_match_value(wf)
            for wf in wc.fields
            if wf.name in own_names and not wf.is_optional
        }
        if kwargs:
            match(**kwargs)
    except Exception:
        pass
    return match


def missing_required_parameters(template: ActionTemplate) -> list[str]:
    """Return names of required parameters still unbound after resolution."""
    try:
        required = _required_field_names(template.action_type)
    except Exception:
        return []
    free_names = {p.attribute_name for p in template.parameters if p.is_free}
    return sorted(required & free_names)


# ── Internal helpers (action_match) ───────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _required_field_names(cls: type) -> frozenset:
    """Names of own, public, required (no-default) fields on *cls*."""
    return frozenset(
        f.name for f in own_dataclass_fields(cls)
        if not f.name.startswith("_")
        and f.default is MISSING
        and f.default_factory is MISSING  # type: ignore[misc]
    )


def _free_match_value(wf: Any) -> Any:
    """Return Ellipsis for leaf fields, or a nested Match for nested dataclass fields."""
    if not is_nested(wf, wf.type_endpoint):
        return ...
    nested_cls = wf.type_endpoint
    nested_match = Match(nested_cls)
    try:
        own_names = _required_field_names(nested_cls)
        wc = ClassDiagram([nested_cls]).get_wrapped_class(nested_cls)
        kwargs = {
            sub_wf.name: _free_match_value(sub_wf)
            for sub_wf in wc.fields
            if sub_wf.name in own_names and not sub_wf.is_optional
        }
        if kwargs:
            nested_match(**kwargs)
    except Exception:
        pass
    return nested_match


def _read_variable(variable: Any) -> Any:
    try:
        value = vars(variable).get("_value_", _UNRESOLVED)
    except TypeError:
        value = getattr(variable, "_value_", _UNRESOLVED)
    if value is not _UNRESOLVED and value is not ...:
        return value
    evaluate = getattr(variable, "evaluate", None)
    if callable(evaluate):
        try:
            value = next(iter(evaluate()))
            if value is not ...:
                variable._value_ = value
                return value
        except Exception:
            pass
    return _UNRESOLVED


def _strip_root(name: str, action_cls: type) -> str:
    prefix = f"{action_cls.__name__}."
    return name[len(prefix):] if name.startswith(prefix) else name


# ── Deserializer — JSON → native PyCRAM objects ───────────────────────────────


def _is_body_type(t: Any) -> bool:
    return "Body" in {cls.__name__ for cls in getattr(t, "__mro__", [])}


def _is_manipulator_type(t: Any) -> bool:
    return "Manipulator" in {cls.__name__ for cls in getattr(t, "__mro__", [])}


def _is_body_instance(v: Any) -> bool:
    return "Body" in {cls.__name__ for cls in type(v).__mro__}


def _resolve_annotation(annotation: Any, owner_cls: type) -> Optional[type]:
    """Resolve a string annotation to its actual type.

    ``from __future__ import annotations`` stores all annotations as strings.
    We resolve them against the owning class's module globals, then fall back
    to sys.modules so that plain names resolve correctly.
    Returns None if the annotation cannot be resolved (e.g. complex generics
    like ``Optional[PlanNode]``).
    """
    if isinstance(annotation, type):
        return annotation

    if not isinstance(annotation, str):
        return None

    name = annotation.strip("'\"")

    # Skip complex generics like Optional[X], Union[X, Y], List[X], etc.
    if re.search(r"[\[\]|,]", name):
        return None

    module_globals = vars(sys.modules.get(owner_cls.__module__, object))
    import builtins as _builtins
    resolved = module_globals.get(name) or getattr(_builtins, name, None)
    return resolved if isinstance(resolved, type) else None


def _resolve_target_type(raw_type: Any, owner_cls: type) -> Optional[type]:
    """Resolve a raw type annotation to a concrete Python type, unwrapping Optional/Union if needed."""
    target = _resolve_annotation(raw_type, owner_cls) if not isinstance(raw_type, type) else raw_type
    if target is None:
        origin = get_origin(raw_type)
        if origin is Union:
            args = [a for a in get_args(raw_type) if a is not type(None)]
            if args:
                target = args[0] if isinstance(args[0], type) else None
    return target


def _hydrate_manipulator(context: Dict[str, Any]) -> Any:
    """Inject the arm's end-effector view from context. Raises if arm is absent."""
    arm = context.get("arm")
    if arm is None:
        raise ValueError("Cannot hydrate Manipulator: 'arm' was not provided in context.")
    _, robot_view = get_active_world()
    return ViewManager.get_end_effector_view(arm, robot_view)


def _hydrate_enum(target_type: type, value: Any) -> Any:
    """Convert a string or existing member to the target Enum type."""
    if isinstance(value, target_type):
        return value
    for member in target_type:
        if member.name.upper() == str(value).upper():
            return member
    raise ValueError(f"'{value}' is not a valid member of Enum {target_type.__name__}.")


def _hydrate_body(value: Any) -> Any:
    """Look up a Body in the active world by instance, string name, or dict with body_name key."""
    if _is_body_instance(value):
        return value
    name = value.get("body_name") or value.get("name") or str(value) if isinstance(value, dict) else str(value)
    body = find_body_by_name(name)
    if body is None:
        raise ValueError(f"Body '{name}' not found in the active world.")
    return body


def _hydrate_dataclass(
    target_type: type, value: dict, context: Dict[str, Any], owner_cls: type
) -> Any:
    """Recursively hydrate a dict into a dataclass instance, injecting Manipulator fields from context."""
    kwargs = {}
    for f in dataclasses.fields(target_type):
        if f.name in value:
            kwargs[f.name] = hydrate_value(f.type, value[f.name], context, target_type)
        elif _resolve_annotation(f.type, target_type) is not None:
            resolved = _resolve_annotation(f.type, target_type)
            if (
                resolved is not None
                and isinstance(resolved, type)
                and _is_manipulator_type(resolved)
                and "arm" in context
            ):
                kwargs[f.name] = hydrate_value(f.type, None, context, target_type)
    return target_type(**kwargs)


def hydrate_value(raw_type: Any, value: Any, context: Dict[str, Any], owner_cls: type) -> Any:
    """Dispatch hydration of *value* to the appropriate type-specific handler.

    Dispatch order is load-bearing: Manipulator injection must precede the
    ``value is None`` guard because the dataclass hydrator passes None as a
    sentinel to trigger Manipulator injection on nested fields.
    """
    target_type = _resolve_target_type(raw_type, owner_cls)
    if target_type is None:
        return value

    # Manipulator injection MUST come before the `value is None` early-return
    if _is_manipulator_type(target_type):
        return _hydrate_manipulator(context)

    if value is None:
        return None

    if issubclass(target_type, Enum):
        return _hydrate_enum(target_type, value)

    if _is_body_type(target_type):
        return _hydrate_body(value)

    if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
        return _hydrate_dataclass(target_type, value, context, owner_cls)

    # Scalar builtin — use KRROOD's is_builtin_type so the check isn't a hardcoded tuple
    if is_builtin_type(target_type):
        try:
            return target_type(value)
        except (TypeError, ValueError):
            return value

    return value


def hydrate_action_kwargs(action_cls: Type, raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dict of raw JSON arguments into native PyCRAM objects for action_cls."""
    hydrated: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

    # First pass: hydrate Enums so they populate context for nested objects
    for f in dataclasses.fields(action_cls):
        if f.name not in raw_kwargs:
            continue
        resolved = _resolve_annotation(f.type, action_cls)
        if resolved is not None and issubclass(resolved, Enum):
            hydrated_val = hydrate_value(f.type, raw_kwargs[f.name], {}, action_cls)
            context[f.name] = hydrated_val
            hydrated[f.name] = hydrated_val

    # Second pass: hydrate everything else
    for f in dataclasses.fields(action_cls):
        if f.name not in raw_kwargs or f.name in hydrated:
            continue
        hydrated[f.name] = hydrate_value(f.type, raw_kwargs[f.name], context, action_cls)

    return hydrated
