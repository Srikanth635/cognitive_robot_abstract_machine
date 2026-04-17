"""KRROOD Match inspection utilities used by llmr backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any, List, Optional

from krrood.entity_query_language.query.match import Match

from llmr._utils import slot_prompt_name


@dataclass(frozen=True)
class MatchBinding:
    """One leaf variable discovered in a KRROOD Match expression."""

    attribute_name: str
    prompt_name: str
    variable: Any
    field_type: Any
    value: Any

    @property
    def is_free(self) -> bool:
        return isinstance(self.value, type(Ellipsis))


def match_bindings(expression: Match[Any], unresolved: Any) -> List[MatchBinding]:
    """Collect one binding per KRROOD Match leaf variable."""
    bindings: List[MatchBinding] = []
    for attr_match in expression.matches_with_variables:
        variable = attr_match.assigned_variable
        full_path = attr_match.name_from_variable_access_path
        prompt_name = slot_prompt_name(full_path, expression.type)
        bindings.append(
            MatchBinding(
                attribute_name=attr_match.attribute_name,
                prompt_name=prompt_name,
                variable=variable,
                field_type=variable._type_,
                value=assigned_variable_value(variable, unresolved),
            )
        )
    return bindings


def assigned_variable_value(assigned_variable: Any, unresolved: Any) -> Any:
    """Return a concrete KRROOD variable value, evaluating selectable variables."""
    try:
        value = vars(assigned_variable).get("_value_", unresolved)
    except TypeError:
        value = getattr(assigned_variable, "_value_", unresolved)
    if value is not unresolved:
        return value

    evaluate = getattr(assigned_variable, "evaluate", None)
    if not callable(evaluate):
        return unresolved
    try:
        value = next(iter(evaluate()))
    except Exception:
        return unresolved
    assigned_variable._value_ = value
    return value


def unresolved_required_fields(expression: Match[Any], introspector: Any) -> List[str]:
    """Return required action fields that are still unset in a Match expression.

    KRROOD may represent a required top-level field as a nested ``Match`` whose
    unresolved leaves live below it, for example ``action.slot.member``. In
    that case the top-level slot is present even though no direct variable named
    ``slot`` appears in ``matches_with_variables``.
    """
    try:
        required = {
            field.name
            for field in introspector.introspect(expression.type).fields
            if not field.is_optional
        }
    except Exception:
        return []

    unresolved: List[str] = []
    seen: set[str] = set()
    for attr_match in expression.matches_with_variables:
        field_name = attr_match.attribute_name
        top_level_name = _top_level_field_name(attr_match)
        seen.add(field_name)
        if top_level_name:
            seen.add(top_level_name)
        value = attr_match.assigned_variable._value_
        if field_name in required and isinstance(value, type(Ellipsis)):
            unresolved.append(field_name)

    expression._update_kwargs_from_literal_values()
    for field_name in sorted(required):
        if field_name not in expression.kwargs:
            if field_name in seen:
                continue
            unresolved.append(field_name)
            continue
        if _match_value_is_resolved(expression.kwargs[field_name]):
            continue
        unresolved.append(field_name)

    return list(dict.fromkeys(unresolved))


def _top_level_field_name(attr_match: Any) -> Optional[str]:
    """Return the first field name in a KRROOD variable access path, if present."""
    try:
        access_path = attr_match.assigned_variable._access_path_
    except AttributeError:
        return None

    try:
        steps = iter(access_path)
    except TypeError:
        steps = iter((access_path,))

    for step in steps:
        attribute_name = getattr(step, "_attribute_name_", None)
        if attribute_name:
            return attribute_name
    return None


def _match_value_is_resolved(value: Any) -> bool:
    """Return whether a Match kwarg value can be constructed without ellipses."""
    if isinstance(value, type(Ellipsis)):
        return False
    if isinstance(value, Match):
        return all(_match_value_is_resolved(item) for item in value.kwargs.values())
    if isinstance(value, dict):
        return all(_match_value_is_resolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_match_value_is_resolved(item) for item in value)
    return True
