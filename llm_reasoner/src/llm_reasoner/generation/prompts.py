"""
What a language model is told when it is asked for a hypothesis, and when it is asked to
correct one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing_extensions import Any, Mapping, Sequence

from llm_reasoner.generation.schema import (
    FieldKind,
    FieldSchema,
    GenerationSchema,
    TypeSchema,
)
from llm_reasoner.generation.wire import CONCEPT_PREFIXES
from llm_reasoner.ontologies.terms import ConceptReference

SYSTEM_PROMPT = """\
You describe how a manipulation action is organised, as a structure of typed fragments
drawn from an ontology of predictive manipulation.

What you produce:
- The phases of the action, the subgoal each phase pursues and the events that end it,
  the participants by their roles, the forces and conditions involved, what the
  controller predicts at each event, how it compares and corrects, and the quantities.
- Participants are roles such as "the source container", never particular objects.
  Quantities are ranges with units, never single measurements you have not made.

How to answer:
- Emit one JSON object with the key "fragments", whose value is a list.
- Every fragment is an object with "id", "type" and the fields of that type. Any
  fragment may also have a "label" naming it in plain words.
- "id" is a short string unique within your answer. Fragments refer to one another by
  these ids, never by nesting.
- Use only the types and fields listed. A field holding many values takes a list.
- Choose the most specific type that fits; a label never replaces a type.
- Enumeration fields take the member name. Concept fields take a prefixed term such as
  aicor:Pouring.
- Leave out a field you cannot fill rather than writing null.

Answer with the JSON object only, without prose or code fences.
"""


# %% rendering


def written_value(value: Any) -> str:
    """
    :return: A fixed value as the model writes it.
    """
    if isinstance(value, ConceptReference):
        for prefix, namespace in CONCEPT_PREFIXES.items():
            if value.iri.startswith(namespace):
                return f"{prefix}:{value.iri[len(namespace):]}"
        return value.iri
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(written_value(item) for item in value) + "]"
    return repr(value)


def field_hint(field_schema: FieldSchema) -> str:
    """
    :return: What one value of a field looks like, in words.
    """
    if field_schema.kind == FieldKind.FRAGMENT:
        hint = f"id of a {field_schema.value_type.__name__} or a more specific type"
    elif field_schema.kind == FieldKind.ENUMERATION:
        hint = "one of " + " | ".join(field_schema.options)
    elif field_schema.kind == FieldKind.CONCEPT:
        hint = "a prefixed term such as aicor:Pouring"
    else:
        hint = f"a {field_schema.kind}"
    return f"list of {hint}" if field_schema.many else hint


def rendered_type(type_schema: TypeSchema) -> str:
    """
    :return: A fragment type, its meaning and the fields it adds to its bases.
    """
    bases = [
        base.__name__
        for base in type_schema.python_type.__bases__
        if "term" in vars(base)
    ]
    heading = f"### {type_schema.name}" + (
        f" (a kind of {', '.join(bases)})" if bases else ""
    )
    own = vars(type_schema.python_type).get("__annotations__", {})
    lines = [heading, type_schema.description]
    lines.extend(
        f"- {field_schema.name}: {field_hint(field_schema)}. {field_schema.description}"
        for field_schema in type_schema.fields
        if field_schema.name in own
    )
    return "\n".join(line for line in lines if line)


# %% prompts


@dataclass
class GenerationPrompt:
    """
    Builds the prompts of one generation.
    """

    def initial(
        self,
        schema: GenerationSchema,
        instruction: str,
        fixed_values: Mapping[str, Any],
    ) -> str:
        """
        :param schema: The fragment types the answer may use.
        :param instruction: The action to describe, in plain words.
        :param fixed_values: Values the root fragment must have, by field name.
        :return: The prompt asking for a hypothesis.
        """
        root = schema.root.__name__
        sections = [
            f"Describe the organisation of this action: {instruction}",
            f"Produce exactly one fragment of type {root}, together with every "
            "fragment it refers to, directly or indirectly.",
        ]
        if fixed_values:
            sections.append(
                f"## Fixed values\n\nThe {root} fragment must have:\n"
                + "\n".join(
                    f"- {name} = {written_value(value)}"
                    for name, value in fixed_values.items()
                )
            )
        sections.append(
            "## Types you may use\n\n"
            + "\n\n".join(
                rendered_type(type_schema)
                for _, type_schema in sorted(schema.types.items())
            )
        )
        sections.append(
            '## Output\n\n{"fragments": [{"id": "...", "type": "...", ...}, ...]}'
        )
        return "\n\n".join(sections)

    @staticmethod
    def repair(previous_prompt: str, raw_response: str, problems: Sequence[str]) -> str:
        """
        :param previous_prompt: The prompt the rejected answer responded to.
        :param raw_response: The rejected answer, in full, since the model keeps no
            memory of it.
        :param problems: What is wrong, each naming the fragment at fault.
        :return: The prompt asking the model to correct its answer.
        """
        return "\n\n".join(
            [
                previous_prompt,
                "## Your previous answer was rejected\n\n" + raw_response,
                "These problems were found:\n"
                + "\n".join(f"- {problem}" for problem in problems),
                "Answer again, correcting these problems and keeping everything "
                "else. JSON object only.",
            ]
        )


def described_violation(violation: Any, identifier: str) -> str:
    """
    :return: A rule violation as the model is told it, naming the fragment's id.
    """
    fragment = violation.fragment
    return (
        f"[{violation.rule}] {type(fragment).__name__} '{identifier}': "
        f"{violation.reason}"
    )
