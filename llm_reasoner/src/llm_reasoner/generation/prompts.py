"""
What a language model is told when it is asked for a hypothesis, and when it is asked to
correct one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Any, Callable, Mapping, Sequence

from llm_reasoner.knowledge.acceptance.derived import described
from llm_reasoner.knowledge.acceptance.rules import Violation
from llm_reasoner.knowledge.acceptance.shapes import StructureShape, StructureShapes
from llm_reasoner.generation.schema import (
    AnswerKey,
    FieldKind,
    FieldSchema,
    GenerationSchema,
    TypeSchema,
)
from llm_reasoner.generation.wire import prefixed_name
from llm_reasoner.knowledge.ontologies.terms import ConceptReference

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
- A type marked "Defined" is met by its fields: a fragment of another type that states
  what the definition requires belongs to it as well. An estimate whose physical
  dimension is a force rate is therefore also a force rate parameter.
- Enumeration fields take the member name. Concept fields take a prefixed term:
  aicor:<Name> or dul:<Name> for a term CRAM-AICOR or DUL declares, such as
  aicor:Pouring, and hypothesis:<Name> for a concept they lack, such as a category of
  grasp. Never invent an aicor: or dul: term.
- Leave out a field you cannot fill rather than writing null.

Answer with the JSON object only, without prose or code fences.
"""


# %% rendering


def written_value(value: Any) -> str:
    """
    :return: A fixed value as the model writes it.
    """
    if isinstance(value, ConceptReference):
        return prefixed_name(value.iri)
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
        hint = (
            "a prefixed term: aicor:<Name> or dul:<Name> that CRAM-AICOR or DUL "
            "declares, or hypothesis:<Name> for a concept they lack"
        )
    else:
        hint = f"a {field_schema.kind}"
    return f"list of {hint}" if field_schema.many else hint


def rendered_type(type_schema: TypeSchema) -> str:
    """
    :return: A fragment type, its meaning, what the ontology requires of it beyond its
        bases, and the fields it adds to them.
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
    definition = type_schema.python_type.term.definition
    if definition is not None:
        lines.append(f"Defined: {described(definition, False)}.")
    lines.extend(
        f"Requires: {described(restriction, False)}."
        for restriction in type_schema.python_type.term.restrictions
    )
    lines.extend(
        f"- {field_schema.name}: {field_hint(field_schema)}. {field_schema.description}"
        for field_schema in type_schema.fields
        if field_schema.name in own
    )
    return "\n".join(line for line in lines if line)


def rendered_shape(shape: StructureShape) -> str:
    """
    :return: A structural rule as the model reads it: the type it constrains and what it
        requires.
    """
    return f"- {shape.target.__name__}: {shape.description}"


# %% prompts


@dataclass
class GenerationPrompt:
    """
    Builds the prompts of one generation.
    """

    shapes: StructureShapes = field(default_factory=StructureShapes.shipped)
    """
    The structural rules stated to the model beside the types they constrain.
    """

    def system(self, schema: GenerationSchema) -> str:
        """
        :param schema: The fragment types the answer may use.
        :return: The system prompt, which depends only on the type generated, so it
            stays the same for every attempt and every instruction and can be cached.
        """
        root = schema.root.__name__
        return "\n\n".join(
            [
                SYSTEM_PROMPT,
                f"Produce exactly one fragment of type {root}, together with the "
                "fragments that make it up. Every fragment you write is part of the "
                f"{root}, so it does not list them.",
                "## Types you may use\n\n"
                + "\n\n".join(
                    rendered_type(type_schema)
                    for _, type_schema in sorted(schema.types.items())
                ),
                "## Rules across fragments\n\n"
                "Each rule relates several fragments, so it is stated here rather "
                "than with one type.\n\n"
                + "\n".join(
                    rendered_shape(shape)
                    for shape in self.shapes.applicable_to(
                        type_schema.python_type for type_schema in schema.types.values()
                    )
                ),
                '## Output\n\n{"fragments": [{"id": "...", "type": "...", ...}, ...]}',
            ]
        )

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
        :return: The prompt asking for a hypothesis of the instruction.
        """
        root = schema.root.__name__
        sections = [f"Describe the organisation of this action: {instruction}"]
        if fixed_values:
            sections.append(
                f"## Fixed values\n\nThe {root} fragment must have:\n"
                + "\n".join(
                    f"- {name} = {written_value(value)}"
                    for name, value in fixed_values.items()
                )
            )
        return "\n\n".join(sections)

    @staticmethod
    def repair(initial_prompt: str, answer: str, problems: Sequence[str]) -> str:
        """
        :param initial_prompt: The first prompt of the request, which earlier repairs
            are not added to.
        :param answer: The answer being repaired, in full, since the model keeps no
            memory of it.
        :param problems: What is wrong with that answer, each naming the fragment at
            fault.
        :return: The prompt asking the model for a patch correcting the answer.
        """
        return "\n\n".join(
            [
                initial_prompt,
                "## Your answer so far\n\n" + answer,
                "These problems were found:\n"
                + "\n".join(f"- {problem}" for problem in problems),
                "Correct them with a patch rather than a new answer: a JSON object "
                f'whose "{AnswerKey.FRAGMENTS}" list holds only the fragments you add '
                "or change, each written in full with all its fields, and whose "
                f'"{AnswerKey.REMOVE}" list holds the ids of the fragments to delete. '
                "Every fragment you do not mention stays as it is. JSON object only.",
            ]
        )


def described_violation(
    violation: Violation, identifier_of: Callable[[Any], str]
) -> str:
    """
    :param violation: The violation.
    :param identifier_of: The answer's identifier of a fragment.
    :return: A rule violation as the model is told it, naming the ids of the fragment
        at fault and of the fragments the violation concerns.
    """
    fragment = violation.fragment
    described = (
        f"[{violation.rule}] {type(fragment).__name__} "
        f"'{identifier_of(fragment)}': {violation.reason}"
    )
    if not violation.involved:
        return described
    return (
        described
        + "; concerns "
        + ", ".join(
            f"{type(involved).__name__} '{identifier_of(involved)}'"
            for involved in violation.involved
        )
    )
