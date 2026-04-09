"""Slot-filler prompt builder.

Two modes:
- ``build_slot_filler_prompt(action_types)`` — plain dict mode (fallback when
  pycram introspection is not available).
- ``build_slot_filler_prompt_from_schemas(schemas)`` — richer mode driven by
  :class:`~krrood.llmr.pycram_bridge.introspector.ActionSchema` objects, giving
  the LLM the exact field names, types, and docstrings for each action.
"""

from __future__ import annotations

from typing_extensions import Dict, List, Optional, TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from krrood.llmr.pycram_bridge.introspector import ActionSchema


# ── Shared system preamble ─────────────────────────────────────────────────────

_SYSTEM_PREAMBLE = """\
You are a robot action slot-filler.

Given a natural language instruction and a world context, extract a structured
action description from the instruction.

OUTPUT STRUCTURE
────────────────
action_type : The action type string (exactly one of the supported types below).

entities    : List of world entities that need to be looked up in the world.
              Each entity has:
                role          — the exact parameter name in the action constructor
                                (e.g. "object_designator", "target_location")
                name          — noun phrase from the instruction
                semantic_type — ontological type hint from the world context
                                (null if unknown)
                spatial_context — spatial qualifier ("on the table")
                attributes    — discriminating attributes "color": "red"

parameters  : Non-entity action parameters (arm choice, enum values, primitives)
              keyed by their exact parameter name.

manner      : Optional execution style from the instruction ("carefully").
              Null if not mentioned.

constraints : Optional list of explicit constraints ("without spilling it").
              Null if none.

ENTITY vs PARAMETER RULE
────────────────────────
Put a value in `entities` if it refers to an object, surface, or region in
the world that must be looked up (a Body, Region, or location).
Put a value in `parameters` if it is a choice between discrete options
(arm, approach direction, technique) or a primitive value (number, bool).

EXTRACTION RULES
────────────────
- Extract ONLY what is EXPLICITLY stated in the instruction. This is the most
  important rule — do not guess, infer, or supply "reasonable defaults".
- If a parameter is not mentioned in the instruction, OMIT it from `parameters`
  entirely. Leave the `parameters` dict empty or sparse rather than inventing values.
- For enum parameters: only include them if the instruction names the value
  (e.g. "use the LEFT arm" → arm=LEFT). If the arm / direction / etc. is not
  mentioned, DO NOT include it.
- Entity `name` field: use the head noun only — strip articles (a, an, the)
  and all modifiers (adjectives, compound qualifiers). Put descriptors like
  color or size in `attributes` instead.
  Examples: "the milk" → "milk", "a red cup" → name="cup" attributes="color":"red",
  "the breakfast cereal" → "cereal", "cooking pot" → "cooking_pot".
- Set action_type to EXACTLY one of the supported action type strings below.

PARAMETERS LISTED BELOW ARE FOR NAME REFERENCE ONLY — use the exact key names
when you DO extract them. Do NOT fill them in unless the instruction states them.
"""

_HUMAN_TEMPLATE = """\
## World Context
{world_context}

## Instruction
{instruction}

Extract the action slot schema.
"""


# ── Introspection-driven prompt ────────────────────────────────────────────────

def build_slot_filler_prompt_from_schemas(
    schemas: "List[ActionSchema]",
) -> ChatPromptTemplate:
    """Build a slot-filler prompt fully informed by pycram action introspection.

    Tells the LLM exactly which entity roles and parameter names to extract for
    each action type, eliminating guesswork.
    """
    from krrood.llmr.pycram_bridge.introspector import FieldKind

    action_block_lines = ["SUPPORTED ACTIONS", "─────────────────"]
    for schema in schemas:
        action_block_lines.append(f"\n### {schema.action_type}")
        action_block_lines.append(f"Description: {schema.docstring}")

        entity_fields  = [f for f in schema.fields if f.kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF)]
        param_fields   = [f for f in schema.fields if f.kind in (FieldKind.ENUM, FieldKind.PRIMITIVE)]
        complex_params = _collect_complex_subparams(schema)

        if entity_fields:
            action_block_lines.append("Entities (put in `entities` list):")
            for f in entity_fields:
                note = " — use .global_pose on the grounded body" if f.kind == FieldKind.POSE else ""
                type_name = f.raw_type.__name__ if isinstance(f.raw_type, type) else str(f.raw_type)
                action_block_lines.append(
                    f"  role='{f.name}' ({type_name}){note}: {f.docstring}"
                )

        all_params = [(f.name, f.docstring, f.enum_members) for f in param_fields] + complex_params
        if all_params:
            action_block_lines.append(
                "Optional parameters (ONLY add to `parameters` if the instruction "
                "explicitly names the value — omit otherwise):"
            )
            for fname, fdoc, fmembers in all_params:
                # List valid member names only as a format hint, not as defaults to pick from
                members_note = f" (valid values: {', '.join(fmembers)})" if fmembers else ""
                action_block_lines.append(f"  '{fname}'{members_note}: {fdoc}")

    action_block = "\n".join(action_block_lines)
    system = _SYSTEM_PREAMBLE + "\n" + action_block
    return ChatPromptTemplate.from_messages(
        [("system", system), ("human", _HUMAN_TEMPLATE)]
    )


def _collect_complex_subparams(schema: "ActionSchema") -> List[tuple]:
    """Flatten COMPLEX field sub-fields into (name, docstring, enum_members) tuples."""
    from krrood.llmr.pycram_bridge.introspector import FieldKind

    result = []
    for f in schema.fields:
        if f.kind == FieldKind.COMPLEX:
            for sub in f.sub_fields:
                if sub.kind in (FieldKind.ENUM, FieldKind.PRIMITIVE):
                    result.append((sub.name, sub.docstring, sub.enum_members))
    return result


# ── Legacy plain-dict prompt ───────────────────────────────────────────────────

def build_slot_filler_prompt(action_types: Dict[str, str]) -> ChatPromptTemplate:
    """Build a slot-filler prompt from a plain ``{action_type: description}`` dict.

    Used when pycram introspection is not available or when the caller provides
    action types manually.
    """
    if action_types:
        action_block = "SUPPORTED ACTIONS\n─────────────────\n" + "\n".join(
            f"  - {name}: {desc}" for name, desc in action_types.items()
        )
    else:
        action_block = "SUPPORTED ACTIONS\n─────────────────\n  (none registered)"

    system = _SYSTEM_PREAMBLE + "\n" + action_block
    return ChatPromptTemplate.from_messages(
        [("system", system), ("human", _HUMAN_TEMPLATE)]
    )


# Default prompt — replaced at runtime
slot_filler_prompt = build_slot_filler_prompt({})
