"""KRROOD Match construction utilities for action schemas."""
from __future__ import annotations

from typing_extensions import Any, Dict, Iterable

from krrood.entity_query_language.query.match import Match


def required_match(action_cls: type) -> Match[Any]:
    """Return ``Match(action_cls)`` with required public schema fields free."""
    match = Match(action_cls)
    try:
        from llmr.pycram_bridge import PycramIntrospector

        fields = PycramIntrospector().introspect(action_cls).fields
    except Exception:
        return match

    kwargs = _required_match_kwargs(fields)
    if kwargs:
        match(**kwargs)
    return match


def _required_match_kwargs(fields: Iterable[Any]) -> Dict[str, Any]:
    """Build Match kwargs for required public fields."""
    kwargs: Dict[str, Any] = {}
    for field in fields:
        if field.is_optional or field.name.startswith("_"):
            continue
        kwargs[field.name] = _free_match_value(field)
    return kwargs


def _free_match_value(field: Any) -> Any:
    """Return ``...`` or a nested Match value for one required field."""
    from llmr.pycram_bridge.introspector import FieldKind

    if field.kind != FieldKind.COMPLEX:
        return ...

    nested_match = Match(field.raw_type)
    nested_kwargs = _required_match_kwargs(field.sub_fields)
    if nested_kwargs:
        nested_match(**nested_kwargs)
    return nested_match
