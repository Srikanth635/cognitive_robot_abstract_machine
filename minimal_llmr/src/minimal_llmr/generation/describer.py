"""Task 2: Generation — describe free action parameters using an LLM.

The LLM receives the instruction, action schema, and world context and returns
a ParameterInterpretation for each free parameter.  Output is still
language-level: entity parameters are described as ReferentDescription
objects, not yet grounded to Symbol instances.

Key functions:
  describe_parameters()   — main entry point (LLM call + repair)

"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap

from typing import TYPE_CHECKING, Any, Optional

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.utils import own_dataclass_fields

from minimal_llmr.bridge.template import is_referent, is_nested
from minimal_llmr.core.schemas import ParameterInterpretation, ParameterInterpretations

if TYPE_CHECKING:
    from krrood.class_diagrams.wrapped_field import WrappedField
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ── System prompt ──────────────────────────────────────────────────────────────

_DESCRIBER_SYSTEM = """\
You are a robot action parameter resolver with strong spatial and physical reasoning.

You receive:
  1. A natural-language instruction from a human operator.
  2. The target robot action type, its description, and all free parameter slots.
  3. Already-fixed parameter values (do not change these).
  4. The current world state: scene objects, positions, and semantic annotations.

Your task: for every FREE parameter, reason carefully and return a ParameterInterpretation.
Ensure all parameter choices are mutually consistent — parameters that refer to the same
physical entity or role (e.g. an arm identifier and its corresponding manipulator object)
must agree. The full set of interpretations should form a coherent, non-contradictory plan.

IMPORTANT: For spatial and geometric reasoning fields you MUST quote actual numeric
values (xyz coordinates, distances, bounding-box dimensions) directly from the Scene
Objects table in the world context. Generic descriptions without numbers are not
acceptable for those fields.

────────────────────────────────────────────────────
REFERENT PARAMETERS  (objects / surfaces in the world)
────────────────────────────────────────────────────
Return a ParameterInterpretation with referent_description populated:
  param_name         = the role name exactly as listed
  referent_description:
    name             = exact name from Available Semantic Types or body_name
    semantic_type    = EXACT type name from Available Semantic Types; null if no match
    spatial_context  = spatial qualifier from instruction ("on the table") or null
    attributes       = discriminating key/value attributes (color, size) or null
  reasoning:
    causal           = why this specific world object is the correct referent —
                       what property, position, or semantic type makes it match
                       the instruction (cause → effect on the action outcome)
    counterfactual   = at least one alternative object that was considered and
                       a specific reason it was rejected (wrong type, wrong
                       location, not present in world, etc.)
    spatial          = MUST quote actual xyz values from the Scene Objects table.
                       State the object's xyz, the robot/arm xyz if visible,
                       and the computed distance or relative direction. Example:
                       "milk xyz=(1.2, 0.4, 0.85), robot base xyz=(0.0, 0.0, 0.0),
                       distance≈1.3m along X — within right arm workspace."
    geometric        = MUST quote actual size values from the Scene Objects table.
                       State the object's (depth, width, height) and explain which
                       dimension is relevant. Example: "milk size=(0.07, 0.07, 0.20)m
                       — narrow square cross-section, height 0.20m confirms upright
                       orientation matches a standard milk carton."

────────────────────────────────────────────────────
DISCRETE PARAMETERS  (enum members / primitive values)
────────────────────────────────────────────────────
Return a ParameterInterpretation with:
  param_name = the parameter name exactly as listed
  value      = chosen value as a string (exact enum member name or primitive)
  reasoning:
    causal           = why this value produces the correct physical behaviour
                       for the action (spatial alignment, reachability, physics)
    counterfactual   = at least one alternative value that was considered and
                       why it would lead to an incorrect or suboptimal outcome
    spatial          = MUST quote actual xyz values from the Scene Objects table.
                       State the object's xyz and derive the relative direction
                       to the robot. Example: "milk xyz=(1.2, 0.4, 0.85) is
                       directly ahead along X (Δx=1.2, Δy=0.4) — FRONT approach
                       minimises lateral deviation."
    geometric        = MUST quote actual size values from the Scene Objects table.
                       State the relevant dimension and explain its effect. Example:
                       "milk size=(0.07, 0.07, 0.20)m — 0.07m narrow face targeted
                       by FRONT approach; manipulation_offset=0.05m clears the face
                       with 2cm margin."

For ENUM parameters use EXACTLY one of the listed allowed values.
After all per-parameter interpretations, fill:
  overall_reasoning      = one-sentence summary of the resolution
  coherence_assessment   = how the full parameter set forms a consistent,
                           non-contradictory plan (e.g. arm and manipulator agree)
Return structured JSON.
"""


# ── Public entry point ─────────────────────────────────────────────────────────


def describe_parameters(
    instruction: Optional[str],
    action_cls: type,
    free_param_names: list[str],
    fixed_params: dict[str, Any],
    world_context: str,
    llm: "BaseChatModel",
    optional_defaults: Optional[dict[str, Any]] = None,
) -> Optional[ParameterInterpretations]:
    """Generate parameter interpretations for *free_param_names* via LLM.

    Returns None on LLM failure.
    """
    stripped_names = [_strip_root(n, action_cls) for n in free_param_names]
    stripped_fixed = {_strip_root(k, action_cls): v for k, v in fixed_params.items()}

    user_message, expected_names = _compose_prompt(
        instruction=instruction,
        action_cls=action_cls,
        free_param_names=stripped_names,
        fixed_params=stripped_fixed,
        world_context=world_context,
        optional_defaults=optional_defaults,
    )

    structured_llm = llm.with_structured_output(ParameterInterpretations)
    try:
        return _invoke_with_repair(
            structured_llm=structured_llm,
            action_cls=action_cls,
            user_message=user_message,
            expected_names=expected_names,
        )
    except Exception:
        logger.exception("describe_parameters: LLM call failed for %s", action_cls.__name__)
        return None


# ── Field docstring extraction ─────────────────────────────────────────────────


def _resolve_leaf_wf(top_wf: "WrappedField", dotted_name: str) -> "Optional[WrappedField]":
    """Walk a dotted path into nested dataclasses and return the leaf WrappedField.

    For 'grasp_description.approach_direction', top_wf is the wf for
    grasp_description (a GraspDescription dataclass); we descend into
    GraspDescription and return the wf for approach_direction so that
    enum members and type info are available for prompt construction.
    """
    current_endpoint = top_wf.type_endpoint
    current_wf: Optional[WrappedField] = None
    for part in dotted_name.split(".")[1:]:
        try:
            own_names = {f.name for f in own_dataclass_fields(current_endpoint)}
            cd = ClassDiagram([current_endpoint])
            wc = cd.get_wrapped_class(current_endpoint)
            wf_map = {wf.name: wf for wf in wc.fields if wf.name in own_names}
            current_wf = wf_map.get(part)
            if current_wf is None:
                return None
            current_endpoint = current_wf.type_endpoint
        except Exception:
            return None
    return current_wf


def _get_leaf_docstring(top_wf: "WrappedField", dotted_name: str) -> str:
    """Return the docstring of the leaf field from its containing nested class.

    For 'grasp_description.approach_direction', descends into GraspDescription
    and returns the docstring attached to approach_direction there, not the
    docstring of the grasp_description field on the parent action class.
    Handles arbitrary nesting depth.
    """
    parts = dotted_name.split(".")
    current_endpoint = top_wf.type_endpoint
    for part in parts[1:-1]:
        try:
            own_names = {f.name for f in own_dataclass_fields(current_endpoint)}
            wc = ClassDiagram([current_endpoint]).get_wrapped_class(current_endpoint)
            wf_map = {wf.name: wf for wf in wc.fields if wf.name in own_names}
            sub_wf = wf_map.get(part)
            if sub_wf is None:
                return ""
            current_endpoint = sub_wf.type_endpoint
        except Exception:
            return ""
    return _extract_field_docstrings(current_endpoint).get(parts[-1], "")


def _extract_field_docstrings(cls: type) -> dict[str, str]:
    """Extract attribute-level docstrings via AST (string literal after annotation)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    except Exception:
        return {}

    docs: dict[str, str] = {}
    class_body: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_body = node.body
            break

    for i, node in enumerate(class_body):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if i + 1 < len(class_body):
            nxt = class_body[i + 1]
            if (
                isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
            ):
                docs[node.target.id] = nxt.value.value.strip()
    return docs


# ── Prompt construction ────────────────────────────────────────────────────────


def _compose_prompt(
    instruction: Optional[str],
    action_cls: type,
    free_param_names: list[str],
    fixed_params: dict[str, Any],
    world_context: str,
    optional_defaults: Optional[dict[str, Any]] = None,
) -> tuple[str, list[str]]:
    """Build the describer user message and list of expected param_names."""
    docstrings: dict[str, str] = {}
    wf_by_name: dict[str, Any] = {}
    referent_specs: list[Any] = []
    discrete_specs: list[Any] = []
    unknown_names: list[str] = []
    expected: list[str] = []

    try:
        own_names = {f.name for f in own_dataclass_fields(action_cls)}
        cd = ClassDiagram([action_cls])
        wc = cd.get_wrapped_class(action_cls)
        docstrings = _extract_field_docstrings(action_cls)
        wf_by_name = {wf.name: wf for wf in wc.fields if wf.name in own_names}

        for name in free_param_names:
            top_name = name.split(".")[0]
            wf = wf_by_name.get(name) or wf_by_name.get(top_name)
            if wf is None:
                unknown_names.append(name)
                expected.append(name)
                continue

            # Dotted sub-field names: resolve the leaf wf for correct type info,
            # then classify as referent or discrete using the leaf type.
            if "." in name:
                leaf_wf = _resolve_leaf_wf(wf, name) or wf
                leaf_doc = _get_leaf_docstring(wf, name)
                if is_referent(leaf_wf, None):
                    referent_specs.append((name, leaf_wf, leaf_doc))
                else:
                    discrete_specs.append((name, leaf_wf, leaf_doc))
                expected.append(name)
            elif is_nested(wf, None):
                continue  # sub-fields appear as dotted names
            elif is_referent(wf, None):
                referent_specs.append((name, wf, docstrings.get(top_name, "")))
                expected.append(name)
            else:
                discrete_specs.append((name, wf, docstrings.get(top_name, "")))
                expected.append(name)

    except Exception as exc:
        logger.debug("_compose_prompt: introspection failed for %s: %s", action_cls.__name__, exc)
        for name in free_param_names:
            unknown_names.append(name)
            expected.append(name)

    expected = list(dict.fromkeys(expected))

    lines: list[str] = []
    if instruction:
        lines.append(f"Instruction: {instruction!r}")
    lines.append(f"Action type: {action_cls.__name__}")

    import inspect as _inspect
    action_doc = " ".join((_inspect.getdoc(action_cls) or "").split())
    if action_doc:
        lines += ["", f"Action description: {action_doc}"]

    if expected:
        lines += [
            "",
            "Required free parameter names:",
            *[f"  - {n}" for n in expected],
            (
                f"Return exactly {len(expected)} ParameterInterpretation entries, "
                "one for each param_name above."
            ),
        ]

    if referent_specs:
        lines += [
            "",
            "── Referent parameters (world objects) ───────────────────────────────",
            "For each: return a ParameterInterpretation with referent_description populated.",
        ]
        for name, wf, doc in referent_specs:
            type_label = getattr(wf.type_endpoint, "__name__", str(wf.type_endpoint))
            doc_str = f": {doc}" if doc else ""
            lines.append(f"  - {name} ({type_label}){doc_str}")

    if discrete_specs:
        lines += [
            "",
            "── Discrete parameters (enum members / primitives) ────────────────────",
            "For each: return a ParameterInterpretation with value = chosen string.",
        ]
        for name, wf, doc in discrete_specs:
            members = list(wf.type_endpoint.__members__) if wf.is_enum else []
            if members:
                doc_str = f" — {doc}" if doc else ""
                lines.append(f"  - {name} (allowed values: {' | '.join(members)}){doc_str}")
            else:
                type_label = getattr(wf.type_endpoint, "__name__", str(wf.type_endpoint))
                doc_str = f": {doc}" if doc else ""
                lines.append(f"  - {name} ({type_label}){doc_str}")

    if unknown_names:
        lines += [
            "",
            "── Additional free parameters (no type info — fill by best judgement) ──",
            *[f"  - {n}" for n in unknown_names],
        ]

    if fixed_params:
        lines += [
            "",
            "── Already-fixed parameters (honour these, do not change) ───────────────",
            *[f"  - {k} = {v!r}" for k, v in fixed_params.items()],
        ]

    if optional_defaults:
        lines += [
            "",
            "── Current defaults (context only — do not return these as interpretations) ──",
        ]
        for name, val in optional_defaults.items():
            if "." in name:
                top_wf = wf_by_name.get(name.split(".")[0])
                doc = _get_leaf_docstring(top_wf, name) if top_wf else ""
            else:
                doc = docstrings.get(name, "")
            doc_str = f": {doc}" if doc else ""
            lines.append(f"  - {name} ({type(val).__name__}, default={val!r}){doc_str}")

    lines += ["", world_context]
    return "\n".join(lines), expected


# ── Repair logic ───────────────────────────────────────────────────────────────


def _invoke_with_repair(
    structured_llm: Any,
    action_cls: type,
    user_message: str,
    expected_names: list[str],
) -> ParameterInterpretations:
    output: ParameterInterpretations = structured_llm.invoke([
        {"role": "system", "content": _DESCRIBER_SYSTEM},
        {"role": "user", "content": user_message},
    ])
    _normalize_names(output, action_cls)

    returned = {interp.param_name for interp in output.interpretations}
    missing = [n for n in expected_names if n not in returned]
    if not missing:
        return output

    logger.warning(
        "describe_parameters: LLM omitted %s for %s — attempting repair",
        missing, action_cls.__name__,
    )
    repair_message = "\n".join([
        user_message,
        "",
        "Correction: the previous response omitted required parameters.",
        "Missing param_names:",
        *[f"  - {n}" for n in missing],
        "",
        "Return a complete ParameterInterpretations. Include one entry for each:",
        *[f"  - {n}" for n in expected_names],
        "",
        "Previous response:",
        output.model_dump_json(),
    ])
    repaired: ParameterInterpretations = structured_llm.invoke([
        {"role": "system", "content": _DESCRIBER_SYSTEM},
        {"role": "user", "content": repair_message},
    ])
    _normalize_names(repaired, action_cls)

    merged = {i.param_name: i for i in output.interpretations}
    for interp in repaired.interpretations:
        merged[interp.param_name] = interp
    return repaired.model_copy(update={"interpretations": list(merged.values())})


def _normalize_names(output: ParameterInterpretations, action_cls: type) -> None:
    for interp in output.interpretations:
        interp.param_name = _strip_root(interp.param_name, action_cls)


def _strip_root(name: str, action_cls: type) -> str:
    prefix = f"{action_cls.__name__}."
    return name[len(prefix):] if name.startswith(prefix) else name
