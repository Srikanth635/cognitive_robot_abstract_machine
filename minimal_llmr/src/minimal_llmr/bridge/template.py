"""ActionTemplate — plain-data snapshot of a KRROOD Match expression.

Public:  snapshot_match, bind_parameter,
         underspecified_match, missing_required_parameters,
         is_referent, is_nested
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import MISSING, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import Symbol
from krrood.utils import own_dataclass_fields

if TYPE_CHECKING:
    from krrood.class_diagrams.wrapped_field import WrappedField

logger = logging.getLogger(__name__)

_UNRESOLVED = object()
_POSE_NAMES: frozenset[str] = frozenset({"Pose", "HomogeneousTransformationMatrix"})


# ── Data structures ────────────────────────────────────────────────────────────


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
        return [p for p in self.parameters if p.is_free]

    @property
    def free_parameter_names(self) -> list[str]:
        return [p.prompt_name for p in self.free_parameters]

    @property
    def fixed_bindings(self) -> dict[str, Any]:
        return {
            p.prompt_name: p.value
            for p in self.parameters
            if not p.is_free and not p.is_optional_default
        }

    @property
    def optional_default_parameters(self) -> list[ActionParameter]:
        return [p for p in self.parameters if p.is_optional_default]

    @property
    def optional_default_bindings(self) -> dict[str, Any]:
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


# ── Public API ─────────────────────────────────────────────────────────────────


def snapshot_match(match: Any) -> ActionTemplate:
    """Snapshot *match* into an ActionTemplate, attaching WrappedField to each parameter.

    Also discovers own dataclass fields with defaults that were not included in the
    Match — for both the top-level action class and nested dataclass types — and
    appends them as optional-default parameters so they appear in the LLM context.
    Uses wf.field.default / wf.field.default_factory from the already-computed
    wf_by_name; no extra imports or helpers needed.
    """
    action_cls = match.type
    try:
        own_names = {f.name for f in own_dataclass_fields(action_cls)}
        wc = ClassDiagram([action_cls]).get_wrapped_class(action_cls)
        wf_by_name: dict[str, Any] = {wf.name: wf for wf in wc.fields if wf.name in own_names}
    except Exception:
        own_names, wf_by_name = set(), {}

    # Explicitly bound / free parameters from the Match
    parameters = [
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

    # Optional default parameters — reuse wf_by_name, read defaults via wf.field
    all_prompt_names = {p.prompt_name for p in parameters}

    for wf in wf_by_name.values():
        if wf.name in all_prompt_names:
            continue
        if wf.field.default is not MISSING:
            val = wf.field.default
        elif wf.field.default_factory is not MISSING:
            val = wf.field.default_factory()
        else:
            continue
        parameters.append(ActionParameter(
            attribute_name=wf.name,
            prompt_name=wf.name,
            field_type=type(val),
            value=val,
            is_free=False,
            is_optional_default=True,
            wf=wf,
        ))

    # Same for nested dataclass types identified by dotted prefixes in the template
    seen_prefixes: set[str] = set()
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
            for nested_wf in nested_wf_by_name.values():
                dotted = f"{prefix}.{nested_wf.name}"
                if dotted in all_prompt_names:
                    continue
                if nested_wf.field.default is not MISSING:
                    val = nested_wf.field.default
                elif nested_wf.field.default_factory is not MISSING:
                    val = nested_wf.field.default_factory()
                else:
                    continue
                parameters.append(ActionParameter(
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



# ── Internal helpers ───────────────────────────────────────────────────────────


def _required_field_names(cls: type) -> set[str]:
    """Names of own, public, required (no-default) fields on *cls*."""
    return {
        f.name for f in own_dataclass_fields(cls)
        if not f.name.startswith("_")
        and f.default is MISSING
        and f.default_factory is MISSING  # type: ignore[misc]
    }


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
    # Ellipsis is KRROOD's sentinel for a free (unbound) variable
    if value is not _UNRESOLVED and value is not ...:
        return value
    evaluate = getattr(variable, "evaluate", None)
    if callable(evaluate):
        try:
            value = next(iter(evaluate()))
            if value is not ...:   # evaluate() also yields ... for free variables
                variable._value_ = value
                return value
        except Exception:
            pass
    return _UNRESOLVED


def _strip_root(name: str, action_cls: type) -> str:
    prefix = f"{action_cls.__name__}."
    return name[len(prefix):] if name.startswith(prefix) else name
