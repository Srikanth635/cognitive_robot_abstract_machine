"""AgenticLLMBackend — KRROOD GenerativeBackend facade over ReActAgent."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.utils import T

from agentic_llmr.core.orchestrator import ReActAgent
from agentic_llmr.resolution.action_match import (
    ActionTemplate,
    bind_parameter,
    snapshot_match,
)
from agentic_llmr.resolution.deserializer import hydrate_value
from agentic_llmr.resolution.schema_docs import build_action_documentation

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# ── Context rendering ──────────────────────────────────────────────────────────


def _render_bound_value(val: Any) -> str:
    """Render a fixed-binding value in a form the LLM can use directly in tool calls.

    SDT Symbol instances (Body, Joint, …) are shown by their human-readable display
    name instead of their Python repr, which would expose opaque UUIDs to the agent.
    """
    try:
        from krrood.symbol_graph.symbol_graph import Symbol
        if isinstance(val, Symbol):
            from agentic_llmr.resolution.scene import symbol_display_name
            return f'"{symbol_display_name(val)}"  (body name in active world)'
    except Exception:
        pass
    return repr(val)


def _build_template_context(template: ActionTemplate) -> str:
    """Render an ActionTemplate into a concise context string for the agent.

    Covers:
    - already-bound parameters (no agent work needed)
    - free parameters declared in the Match expression
    - required fields missing from the Match entirely (gap params, _variable=None)
    - optional parameters with defaults (override only if scene demands it)
    - full action schema so the agent knows enum values and nested dict keys
    """
    lines = [f"Action Class: {template.action_name}"]

    if template.fixed_bindings:
        lines.append("\nAlready-bound parameters (do NOT re-resolve these):")
        for name, value in template.fixed_bindings.items():
            lines.append(f"  {name} = {_render_bound_value(value)}")

    if template.free_parameters:
        lines.append("\nFree parameters that must be resolved from the scene:")
        for param in template.free_parameters:
            ft = param.field_type
            if isinstance(ft, type) and issubclass(ft, Enum):
                members = " | ".join(e.name for e in ft)
                lines.append(f"  {param.prompt_name}  (Enum: {members})")
            elif isinstance(ft, type) and dataclasses.is_dataclass(ft):
                lines.append(
                    f"  {param.prompt_name}  ({ft.__name__} dict"
                    " — see schema below for required keys)"
                )
            else:
                type_label = getattr(ft, "__name__", str(ft))
                lines.append(f"  {param.prompt_name}  (type: {type_label})")
    else:
        lines.append("\nAll parameters are already bound — confirm and return the JSON.")

    if template.optional_default_bindings:
        lines.append("\nOptional parameters with defaults (override only if scene context demands it):")
        for name, value in template.optional_default_bindings.items():
            lines.append(f"  {name} = {value!r}")

    lines.append("\n--- Full Parameter Schema (use this to build your JSON output) ---")
    lines.append(build_action_documentation(template.action_type))
    if _action_has_manipulator_field(template.action_type):
        lines.append(
            "\nNote: Manipulator is auto-injected from arm — do NOT include it in your JSON."
        )

    return "\n".join(lines)


def _action_has_manipulator_field(action_cls: type) -> bool:
    """Return True if action_cls (or any of its nested dataclass fields) has a Manipulator-typed field."""
    from agentic_llmr.resolution.deserializer import _is_manipulator_type
    from agentic_llmr.resolution.deserializer import _resolve_annotation

    def _scan(cls: type, seen: set) -> bool:
        if cls in seen or not dataclasses.is_dataclass(cls):
            return False
        seen.add(cls)
        for f in dataclasses.fields(cls):
            resolved = _resolve_annotation(f.type, cls) if not isinstance(f.type, type) else f.type
            if resolved is None:
                continue
            if _is_manipulator_type(resolved):
                return True
            if dataclasses.is_dataclass(resolved) and _scan(resolved, seen):
                return True
        return False

    return _scan(action_cls, set())


# ── JSON parsing ───────────────────────────────────────────────────────────────


def _parse_json_response(agent_response: str) -> Any:
    """Extract and parse JSON from an agent response string."""
    json_match = re.search(r"```json\s*(.*?)\s*```", agent_response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    return json.loads(agent_response)


def _lookup(parameters: Dict[str, Any], prompt_name: str) -> Any:
    """Retrieve a value from the agent's parameters dict using a dotted prompt_name."""
    parts = prompt_name.split(".")
    node: Any = parameters
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# ── KRROOD-native construction ─────────────────────────────────────────────────


def _build_enum_context(template: ActionTemplate, raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-hydrate Enum fields into a context dict so Manipulator injection works."""
    context: Dict[str, Any] = {}
    for param in template.free_parameters:
        if isinstance(param.field_type, type) and issubclass(param.field_type, Enum):
            raw = _lookup(raw_params, param.prompt_name)
            if raw is not None:
                try:
                    context[param.attribute_name] = hydrate_value(
                        param.field_type, raw, {}, template.action_type
                    )
                except Exception:
                    pass
    return context


def _construct_via_krrood(
    template: ActionTemplate,
    raw_params: Dict[str, Any],
) -> Any:
    """Hydrate the agent's JSON into the Match, then call KRROOD construct_instance().

    Parameters with a backing MappedVariable (_variable is not None) are bound
    via bind_parameter so KRROOD's own machinery propagates the value.
    Parameters that were missing from the Match entirely (_variable is None) are
    injected directly into expression.kwargs before construction.
    """
    context = _build_enum_context(template, raw_params)
    extra_kwargs: Dict[str, Any] = {}

    for param in template.free_parameters:
        raw = _lookup(raw_params, param.prompt_name)
        if raw is None:
            continue
        try:
            hydrated = hydrate_value(param.field_type, raw, context, template.action_type)
        except Exception as exc:
            print(f"[Backend] hydrate_value failed for '{param.prompt_name}': {exc}")
            continue

        if param._variable is not None:
            bind_parameter(param, hydrated)
        else:
            # Gap parameter — no MappedVariable; inject directly into Match kwargs
            extra_kwargs[param.attribute_name] = hydrated

    expression = template._expression
    expression._update_kwargs_from_literal_values()
    for name, val in extra_kwargs.items():
        expression.kwargs[name] = val

    return expression.construct_instance()


# ── Backend ────────────────────────────────────────────────────────────────────


@dataclass
class AgenticLLMBackend(GenerativeBackend):
    """Thin KRROOD backend that delegates resolution to the ReActAgent."""

    llm: "BaseChatModel"
    instruction: Optional[str] = field(kw_only=True, default=None)
    _agent: ReActAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = ReActAgent(llm=self.llm)

    def _evaluate(self, expression: Any) -> Iterable[T]:
        print("[AgenticLLMBackend] Routing input to ReActAgent...")

        if isinstance(expression, str):
            # Raw NL instruction — no Match template available
            instruction_text = self.instruction or "Follow the instruction and output execution parameters."
            context_str = f"Raw Instruction: {expression}"
            template = None
        else:
            # KRROOD Match — snapshot to extract bound/free parameters
            instruction_text = self.instruction or "Resolve the free parameters for the provided action."
            template = None
            try:
                template = snapshot_match(expression)
                context_str = _build_template_context(template)
                print(
                    f"[AgenticLLMBackend] Action: {template.action_name} | "
                    f"free: {template.free_parameter_names} | "
                    f"bound: {list(template.fixed_bindings)}"
                )
            except Exception as exc:
                action_class_name = getattr(expression.type, "__name__", str(expression.type))
                context_str = f"Target Action Class: {action_class_name}"
                print(f"[AgenticLLMBackend] snapshot_match failed ({exc}), falling back to class name only.")

        agent_result_str = self._agent.resolve_action(
            instruction=instruction_text,
            template_context=context_str,
        )
        print("[AgenticLLMBackend] Agent finished. Constructing action instance...")

        if template is not None:
            # Match path — use KRROOD-native construction
            try:
                payload = _parse_json_response(agent_result_str)
                items = payload if isinstance(payload, list) else [payload]
                instances = []
                for item in items:
                    raw_params = item.get("parameters", item)
                    instances.append(_construct_via_krrood(template, raw_params))
                result = instances if len(instances) > 1 else instances[0]
                yield result
                return
            except Exception as exc:
                print(f"[AgenticLLMBackend] KRROOD construction failed ({exc}), falling back to hydrator.")

        # Fallback / raw-string path — use hydrator-based construction
        try:
            action_instance = self._agent.parse_and_hydrate_action(agent_result_str)
            yield action_instance
        except Exception as e:
            print(f"[AgenticLLMBackend] Error parsing agent response: {e}")
            yield agent_result_str  # type: ignore
