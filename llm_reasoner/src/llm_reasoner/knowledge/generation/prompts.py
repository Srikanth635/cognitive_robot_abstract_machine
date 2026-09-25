"""
Prompts for eliciting a hypothesis.

WHY NONE OF THE RESOLVER'S PROMPT TEXT IS REUSED
------------------------------------------------
Its whole vocabulary is built for a different job. "Ellipsis marks a blank",
"choose one candidate by its integer token", "Situation states what is currently
true" -- all of it presumes one partially specified object being completed against
a world. Generation has no world, nothing to select from, and produces a graph
rather than an object. Sharing the text would mean an edit for one half quietly
changing what the other asks for.

What IS shared is the rendering machinery: describing a type by its docstring,
hinting a JSON shape for a value, appending a rejection to a prompt for repair.
Those operate on schema data and know nothing about queries.

WHY THE REPAIR MESSAGE IS THE INTERESTING PART
-----------------------------------------------
A rejected hypothesis does not come back as a parse error. It comes back as a
constraint violation stated in the ontology's own vocabulary -- "this transition
fires on an event that does not end the phase it leaves", naming the fragment at
fault. That tells a model both what is wrong and where, which a syntax error
cannot, and it is the reason the acceptance suite was built to report subjects
rather than booleans.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

from llm_reasoner.knowledge.generation.schema import (
    ClassSchema,
    FieldKind,
    FieldSchema,
    GenerationSchema,
)

SYSTEM_PROMPT = """\
You describe what a kind of action involves, in general.

What you are producing:
- A hypothesis about how a kind of manipulation is organised: its phases, what
  each phase is for, what marks each phase as finished, who or what takes part,
  and the quantities involved.
- It is GENERIC. There is no particular scene, no particular object, no
  particular robot. Never name a specific cup, table or room, and never state a
  measured value as though you had observed one.

How to answer:
- Emit a JSON object with one key, "fragments", whose value is an array.
- Every fragment is an object with "id" and "type", plus that type's fields.
- "id" is any short string unique within your answer. Cross-references between
  fragments use these ids, never nested objects.
- Build bottom-up: a fragment must appear before, or alongside, anything that
  refers to it. Two fragments may refer to each other.
- Exactly one fragment has the root type you are told to produce.
- Use only the types and field names listed. Add no others.
- Enum fields take the member name as a string.
- Omit an optional field rather than filling it with null or a guess.

On quantities:
- Give ranges, not measurements. What a vessel weighs in general is 0.2 to 0.5
  kg; asserting 0.34 kg claims an observation you have not made.
- Always state a unit. An unlabelled number cannot be used.

On honesty:
- Where the action itself does not determine a value, say so with a wide range
  rather than inventing a plausible one.
- Mark as an assumption anything the plan does not itself bring about, so that
  whoever executes it knows to check rather than to trust.

Answer with the JSON object only. No prose, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def hint_for(field: FieldSchema) -> str:
    """:return: The expected JSON shape for one field."""
    if field.kind is FieldKind.CONCEPT and field.options:
        base = "one ontology concept: " + " | ".join(field.options)
    elif field.kind is FieldKind.CONCEPT:
        base = "an IRI or bare name of a term from the reference vocabulary"
    elif field.options:
        base = "one of: " + " | ".join(field.options)
    elif field.kind is FieldKind.FRAGMENT:
        base = f"id of a {field.type_name} fragment"
        if field.options and len(field.options) <= 12:
            base += " -- one of: " + ", ".join(field.options)
        elif field.options:
            base += f" -- any of {len(field.options)} kinds, e.g. " + ", ".join(
                field.options[:6]
            )
    elif field.type_name == "bool":
        base = "true or false"
    elif field.type_name in ("int", "float"):
        base = "a number"
    else:
        base = "a string"
    if field.collection:
        base = f"array of [{base}]"
    if field.required:
        base += f"   (required, at least {field.required})"
    elif field.has_default and field.default is not None:
        # a scalar's own class-level default, e.g. is_failure_event defaulting
        # to false: stating it gives the model a concrete prior for the ordinary
        # case instead of only "omit if optional", which a plausible-sounding
        # value (a threshold crossing "sounds like" a failure) can override
        base += f"   (optional; defaults to {_render_default(field.default)})"
    return base


def _render_default(value: Any) -> str:
    """Render a field's default the way the model is asked to write values."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return value.name
    return str(value)


def render_field(field: FieldSchema) -> str:
    lines = [f"    {field.name}: {hint_for(field)}"]
    if field.docstring:
        lines.append(f'      "{" ".join(field.docstring.split())}"')
    return "\n".join(lines)


def render_class(schema: ClassSchema, subtypes: Sequence[str] = ()) -> str:
    """
    Render one type for the prompt.

    Subtypes are deliberately NOT listed here. Every field that accepts this type
    already names the concrete kinds it will take, which is where a model is actually
    choosing; repeating them under the class costs a third of the prompt and adds
    nothing. ``subtypes`` is kept in the signature for callers that want a standalone
    rendering of a type outside a field context.
    """
    lines = [schema.name + (f"  (a kind of {schema.parent})" if schema.parent else "")]
    if schema.docstring:
        lines.append(f'  "{" ".join(schema.docstring.split())}"')
    if subtypes:
        lines.append("  more specific kinds: " + ", ".join(subtypes))
    lines.extend(render_field(f) for f in schema.fields)
    return "\n".join(lines)


@dataclass
class GenerationPromptBuilder:
    """
    Assembles the user prompt from the derived schema.

    Kept separate from the schema itself so that how a model is asked can change without
    touching what it is asked for -- the schema is derived from the classes and is not a
    place for presentation decisions.
    """

    include_docstrings: bool = True
    """
    Whether to carry each class's own explanation into the prompt.

    Worth the tokens. The docstrings say what a phase or a subgoal IS, which is the part
    that decides whether a model fills the structure sensibly rather than merely
    legally, and nothing was written twice to produce them.
    """

    max_subtypes_listed: int = 20

    def build(
        self,
        schema: GenerationSchema,
        action_label: str,
        degraded: bool = False,
        notes: Sequence[str] = (),
    ) -> str:
        """
        :param schema: The derived output shape.
        :param action_label: The action being described, in plain words.
        :param degraded: Whether only a fallback profile matched, so no
            action-specific vocabulary was loaded.
        :param notes: Extra guidance from the caller.
        """
        sections = [
            f"Describe what is involved in: {action_label}",
            f"Produce exactly one fragment of type {schema.root}, plus every "
            "fragment it refers to, directly or indirectly.",
            "## Ontology classification\n\n"
            "- Select the most specific available type for every fragment.\n"
            "- The fragment `type` states what the fragment is. `label` does "
            "not create a subtype.\n"
            '- "label" is only a human-readable description of that typed '
            "fragment.\n"
            "- Do not encode a new concept only in a label. If the vocabulary "
            "contains the concept, use its named type or concept option.",
        ]

        if degraded:
            sections.append(
                "## No specific vocabulary for this action\n\n"
                "Only the general apparatus is loaded, so there are no named "
                "phase or event types for this action in particular. Use the "
                "general types and give each fragment a clear label. Do not "
                "invent type names."
            )

        root_schema = schema.get(schema.root)
        if root_schema is not None:
            sections.append(
                "## The fragment you are building toward\n\n"
                + render_class(root_schema)
            )

        sections.append("## Types you may use\n\n" + self.render_vocabulary(schema))

        if notes:
            sections.append("## Also\n\n" + "\n".join(f"- {n}" for n in notes))

        sections.append(
            "## Output\n\n"
            '{"fragments": [ {"id": "...", "type": "...", ...}, ... ]}\n\n'
            f"Exactly one fragment of type {schema.root}. Ids unique within "
            "your answer. References by id only."
        )
        return "\n\n".join(sections)

    def render_vocabulary(self, schema: GenerationSchema) -> str:
        """
        Render every type, with more specific kinds listed under their base.

        Listing subtypes under their base is what turns a flat vocabulary into a
        choice a model can make well: seeing that a phase may be a load phase or
        a tilt-to-flow phase is the difference between naming a phase and
        classifying one.
        """
        return "\n\n".join(render_class(c) for c in schema.classes)

    @staticmethod
    def build_repair(
        previous_prompt: str,
        raw_response: str,
        violations: Iterable[Any],
        reason: Optional[str] = None,
        max_reported: int = 12,
    ) -> str:
        """
        Ask again, saying precisely what was wrong and where.

        The client is single-turn: ``complete`` and ``complete_structured``
        carry no conversation state, so this prompt is the ONLY place the model
        can see what it wrote last time. Showing the FULL previous response is
        therefore not generosity, it is a functional requirement -- a model
        cannot correct a mistake in text that was cut off before the mistake
        appears, and a fragment named only in an error message, with the
        response itself truncated well before it, is exactly that: a repair
        instruction pointing at something the model can no longer see.

        Violations are reported as the constraint suite states them, subject
        included, because a model correcting "the transition out of the load
        phase fires on flow onset" can find the fragment, while one told "invalid
        output" can only try again.

        :param violations: Constraint violations, or empty when the failure was
            structural and never reached validation.
        :param reason: A decoding or parsing failure, when there was one.
        """
        listed = list(violations)[:max_reported]
        parts = [
            previous_prompt,
            "## Your previous answer was rejected\n\n```\n" + raw_response + "\n```",
        ]
        if reason:
            parts.append(f"Reason: {reason}")
        if listed:
            parts.append(
                "These specific problems were found:\n\n"
                + "\n".join(f"- {v}" for v in listed)
                + (
                    "\n- ... and more of the same kind"
                    if len(list(violations)) > max_reported
                    else ""
                )
            )
        parts.append(
            "Answer again, correcting these. Keep everything that was right; "
            "change only what the problems name. JSON object only."
        )
        return "\n\n".join(parts)


__all__ = [
    "GenerationPromptBuilder",
    "SYSTEM_PROMPT",
    "hint_for",
    "render_class",
    "render_field",
]
