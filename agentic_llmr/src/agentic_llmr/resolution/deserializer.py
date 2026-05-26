"""Deserializer — converts LLM JSON output into native PyCRAM Python objects."""

import re
import sys
from enum import Enum
from dataclasses import is_dataclass, fields
from typing import Any, Dict, Optional, Type, Union, get_origin, get_args

from krrood.utils import is_builtin_type

from agentic_llmr.integrations.world_manager import get_active_world
from agentic_llmr.resolution.scene import find_body_by_name
from pycram.view_manager import ViewManager


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

    # Strip surrounding quotes added by some introspection paths
    name = annotation.strip("'\"")

    # Skip complex generics like Optional[X], Union[X, Y], List[X], etc.
    if re.search(r"[\[\]|,]", name):
        return None

    module_globals = vars(sys.modules.get(owner_cls.__module__, object))
    import builtins as _builtins
    resolved = module_globals.get(name) or getattr(_builtins, name, None)
    return resolved if isinstance(resolved, type) else None


def hydrate_value(raw_type: Any, value: Any, context: Dict[str, Any], owner_cls: type) -> Any:
    """Recursively hydrates a single value based on its target type annotation."""

    target_type = _resolve_annotation(raw_type, owner_cls) if not isinstance(raw_type, type) else raw_type

    # Handle Optional[X] / Union[X, None] on already-resolved typing generics
    if target_type is None:
        origin = get_origin(raw_type)
        if origin is Union:
            args = [a for a in get_args(raw_type) if a is not type(None)]
            if args:
                target_type = args[0] if isinstance(args[0], type) else None

    if target_type is None:
        return value

    # Manipulator — auto-inject from context arm.
    # This check MUST come before `value is None` because the dataclass hydration loop
    # intentionally passes None as the sentinel value to trigger injection here.
    if _is_manipulator_type(target_type):
        arm = context.get("arm")
        if arm is None:
            raise ValueError("Cannot hydrate Manipulator: 'arm' was not provided in context.")
        _, robot_view = get_active_world()
        return ViewManager.get_end_effector_view(arm, robot_view)

    if value is None:
        return None

    # Enum
    if issubclass(target_type, Enum):
        if isinstance(value, target_type):
            return value
        for member in target_type:
            if member.name.upper() == str(value).upper():
                return member
        raise ValueError(f"'{value}' is not a valid member of Enum {target_type.__name__}.")

    # Body — accept either a Body instance, a plain string name, or a dict with body_name key
    if _is_body_type(target_type):
        if _is_body_instance(value):
            return value
        if isinstance(value, dict):
            name = value.get("body_name") or value.get("name") or str(value)
        else:
            name = str(value)
        body = find_body_by_name(name)
        if body is None:
            raise ValueError(f"Body '{name}' not found in the active world.")
        return body

    # Dataclass (e.g. GraspDescription) — recurse
    if is_dataclass(target_type) and isinstance(value, dict):
        kwargs = {}
        for f in fields(target_type):
            if f.name in value:
                kwargs[f.name] = hydrate_value(f.type, value[f.name], context, target_type)
            elif _resolve_annotation(f.type, target_type) is not None:
                resolved = _resolve_annotation(f.type, target_type)
                if resolved is not None and isinstance(resolved, type) and _is_manipulator_type(resolved) and "arm" in context:
                    kwargs[f.name] = hydrate_value(f.type, None, context, target_type)
        return target_type(**kwargs)

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
    for f in fields(action_cls):
        if f.name not in raw_kwargs:
            continue
        resolved = _resolve_annotation(f.type, action_cls)
        if resolved is not None and issubclass(resolved, Enum):
            hydrated_val = hydrate_value(f.type, raw_kwargs[f.name], {}, action_cls)
            context[f.name] = hydrated_val
            hydrated[f.name] = hydrated_val

    # Second pass: hydrate everything else
    for f in fields(action_cls):
        if f.name not in raw_kwargs or f.name in hydrated:
            continue
        hydrated[f.name] = hydrate_value(f.type, raw_kwargs[f.name], context, action_cls)

    return hydrated
