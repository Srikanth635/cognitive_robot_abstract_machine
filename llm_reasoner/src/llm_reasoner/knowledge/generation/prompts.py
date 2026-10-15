"""
What a language model is told when it is asked for a hypothesis, and when it is asked to
correct one: tagged sections holding text or data, and rules each stated once under its
own name, as every prompt of llm_reasoner is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, StrEnum

from typing_extensions import Any, Callable, Dict, List, Mapping, Sequence

from llm_reasoner.knowledge.acceptance.derived import described
from llm_reasoner.knowledge.acceptance.rules import Violation
from llm_reasoner.knowledge.acceptance.shapes import StructureShape, StructureShapes
from llm_reasoner.knowledge.generation.briefs import GenerationBrief
from llm_reasoner.knowledge.generation.schema import (
    AnswerKey,
    FieldKind,
    FieldSchema,
    GenerationSchema,
    TypeSchema,
)
from llm_reasoner.knowledge.generation.wire import prefixed_name
from llm_reasoner.knowledge.ontologies.terms import ConceptReference
from llm_reasoner.prompt_structure import (
    SECTION_SEPARATOR,
    PromptBlock,
    PromptRule,
    render_blocks,
)

JSON_INDENT = 2
"""
The indentation of the JSON data a prompt holds.
"""

# %% sections


class GenerationSection(StrEnum):
    """
    The sections the prompts of a generation are made of, each named by its tag.
    """

    ROLE = "role"
    """
    What the model is asked to do.
    """

    INPUTS = "inputs"
    """
    What each section of a request means.
    """

    RULES = "rules"
    """
    How the model answers.
    """

    TYPES = "types"
    """
    The fragment types an answer may use, as data.
    """

    STRUCTURE_RULES = "structure_rules"
    """
    The rules relating several fragments, as data.
    """

    OUTPUT = "output"
    """
    The shape of the answer.
    """

    INSTRUCTION = "instruction"
    """
    The action to describe, in plain words.
    """

    FIXED_VALUES = "fixed_values"
    """
    The values the root fragment must have, as data.
    """

    REJECTED_ANSWER = "rejected_answer"
    """
    The answer so far, which is being corrected.
    """

    PROBLEMS = "problems"
    """
    What is wrong with the answer so far, as data.
    """


INPUT_MEANINGS = {
    GenerationSection.TYPES: "the fragment types an answer may use: each type's name, "
    "the types it is a kind of, its meaning, its definition when it has one, what it "
    "requires, and the fields it adds, each with what it takes, whether it takes "
    "several values, and its meaning.",
    GenerationSection.STRUCTURE_RULES: "rules relating several fragments, each with "
    "the type it constrains.",
    GenerationSection.OUTPUT: "the shape of the answer.",
    GenerationSection.INSTRUCTION: "the action to describe, in plain words.",
    GenerationSection.FIXED_VALUES: "the values the root fragment must have, by field.",
    GenerationSection.REJECTED_ANSWER: "your answer so far, which is being corrected.",
    GenerationSection.PROBLEMS: "what is wrong with that answer, each naming the "
    "fragment at fault.",
}
"""
What each section of a generation's prompts means, in the order they may appear.
"""

# %% rules


class GenerationRuleName(StrEnum):
    """
    The rules a generation's answer follows, each named for what it governs.
    """

    ROOT = "root"
    ANSWER_FORMAT = "answer_format"
    FRAGMENT_KEYS = "fragment_keys"
    REFERENCES = "references"
    TYPES_AND_FIELDS = "types_and_fields"
    SPECIFIC_TYPE = "specific_type"
    DEFINED_TYPE = "defined_type"
    ENUMERATIONS = "enumerations"
    CONCEPTS = "concepts"
    ABSENT_FIELDS = "absent_fields"
    FIXED_VALUES = "fixed_values"
    REPAIR = "repair"


class TypeKey(StrEnum):
    """
    The keys of a type in the types section.
    """

    NAME = "name"
    KIND_OF = "kind_of"
    MEANING = "meaning"
    DEFINED = "defined"
    REQUIRES = "requires"
    FIELDS = "fields"


class FieldKey(StrEnum):
    """
    The keys of a field of a type in the types section.
    """

    NAME = "name"
    TAKES = "takes"
    MANY = "many"
    MEANING = "meaning"


class TakesKey(StrEnum):
    """
    The keys saying what one value of a field is.
    """

    FRAGMENT = "fragment"
    """
    The id of a fragment of the named type or of a more specific one.
    """

    ENUM = "enum"
    """
    One of the listed member names.
    """

    CONCEPT = "concept"
    """
    A prefixed term, in the listed forms.
    """

    LITERAL = "literal"
    """
    A JSON value of the named kind.
    """


class StructureRuleKey(StrEnum):
    """
    The keys of a rule in the structure rules section.
    """

    TYPE = "type"
    RULE = "rule"


CONCEPT_FORMS = "aicor:<Name> | dul:<Name> | hypothesis:<Name>"
"""
The forms a concept field's value takes.
"""


def generation_rules(root: str) -> List[PromptRule]:
    """
    :param root: The name of the type generated.
    :return: Every rule about the form of a generation's answer.
    """
    return [
        PromptRule(
            GenerationRuleName.ROOT,
            f"produce exactly one fragment of type {root}, with the fragments that "
            f"make it up; every fragment you write is part of the {root}, so it does "
            "not list them.",
        ),
        PromptRule(
            GenerationRuleName.ANSWER_FORMAT,
            f"reply with one JSON object following <{GenerationSection.OUTPUT}>; no "
            "prose, no code fences.",
        ),
        PromptRule(
            GenerationRuleName.FRAGMENT_KEYS,
            f'every fragment is an object with "{AnswerKey.IDENTIFIER}", '
            f'"{AnswerKey.TYPE}" and the fields of that type, and may have a "label" '
            "naming it in plain words.",
        ),
        PromptRule(
            GenerationRuleName.REFERENCES,
            f'"{AnswerKey.IDENTIFIER}" is a short string unique within your answer; '
            "fragments refer to one another by these ids, never by nesting.",
        ),
        PromptRule(
            GenerationRuleName.TYPES_AND_FIELDS,
            f"use only the types and fields <{GenerationSection.TYPES}> lists; a field "
            f'whose "{FieldKey.MANY}" is true takes a list, and a field taking a '
            f'"{TakesKey.FRAGMENT}" takes the id of a fragment of that type or of a '
            "more specific one.",
        ),
        PromptRule(
            GenerationRuleName.SPECIFIC_TYPE,
            "choose the most specific type that fits; a label never replaces a type.",
        ),
        PromptRule(
            GenerationRuleName.DEFINED_TYPE,
            f'a type with "{TypeKey.DEFINED}" is met by its fields: a fragment of '
            "another type that states what the definition requires belongs to it as "
            "well, so an estimate whose physical dimension is a force rate is also a "
            "force rate parameter.",
        ),
        PromptRule(
            GenerationRuleName.ENUMERATIONS,
            f'a field taking an "{TakesKey.ENUM}" takes one of its member names.',
        ),
        PromptRule(
            GenerationRuleName.CONCEPTS,
            f'a field taking a "{TakesKey.CONCEPT}" takes a prefixed term: '
            "aicor:<Name> or dul:<Name> for a term CRAM-AICOR or DUL declares, such as "
            "aicor:Pouring, and hypothesis:<Name> for a concept they lack, such as a "
            "category of grasp; never invent an aicor: or dul: term.",
        ),
        PromptRule(
            GenerationRuleName.ABSENT_FIELDS,
            "leave out a field you cannot fill rather than writing null.",
        ),
        PromptRule(
            GenerationRuleName.FIXED_VALUES,
            f"the {root} fragment has every value <{GenerationSection.FIXED_VALUES}> "
            "states.",
        ),
        PromptRule(
            GenerationRuleName.REPAIR,
            f"when <{GenerationSection.REJECTED_ANSWER}> and "
            f"<{GenerationSection.PROBLEMS}> follow the request, correct the answer "
            f'with a patch: a JSON object whose "{AnswerKey.FRAGMENTS}" list holds '
            "only the fragments you add or change, each written in full with all its "
            f'fields, and whose "{AnswerKey.REMOVE}" list holds the ids of the '
            "fragments to delete; every fragment you do not mention stays as it is.",
        ),
    ]


# %% types and rules as data


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


def field_takes(field_schema: FieldSchema) -> Dict[str, Any]:
    """
    :return: What one value of a field is, as data.
    """
    if field_schema.kind == FieldKind.FRAGMENT:
        return {TakesKey.FRAGMENT: field_schema.value_type.__name__}
    if field_schema.kind == FieldKind.ENUMERATION:
        return {TakesKey.ENUM: list(field_schema.options)}
    if field_schema.kind == FieldKind.CONCEPT:
        return {TakesKey.CONCEPT: CONCEPT_FORMS}
    return {TakesKey.LITERAL: field_schema.kind}


def type_entry(type_schema: TypeSchema) -> Dict[str, Any]:
    """
    :return: A fragment type as data: its name, the types it is a kind of, its meaning,
        its definition when it has one, what the ontology requires of it beyond its
        bases, and the fields it adds to them.
    """
    term = type_schema.python_type.term
    own = vars(type_schema.python_type).get("__annotations__", {})
    entry: Dict[str, Any] = {
        TypeKey.NAME: type_schema.name,
        TypeKey.KIND_OF: [
            base.__name__
            for base in type_schema.python_type.__bases__
            if "term" in vars(base)
        ],
        TypeKey.MEANING: type_schema.description,
    }
    if term.definition is not None:
        entry[TypeKey.DEFINED] = described(term.definition, False)
    entry[TypeKey.REQUIRES] = [
        described(restriction, False) for restriction in term.restrictions
    ]
    entry[TypeKey.FIELDS] = [
        {
            FieldKey.NAME: field_schema.name,
            FieldKey.TAKES: field_takes(field_schema),
            FieldKey.MANY: field_schema.many,
            FieldKey.MEANING: field_schema.description,
        }
        for field_schema in type_schema.fields
        if field_schema.name in own
    ]
    return entry


def structure_rule_entry(shape: StructureShape) -> Dict[str, str]:
    """
    :return: A structural rule as data: the type it constrains and what it requires.
    """
    return {
        StructureRuleKey.TYPE: shape.target.__name__,
        StructureRuleKey.RULE: shape.description,
    }


def as_json(data: Any) -> str:
    """
    :return: Data as the JSON a prompt holds.
    """
    return json.dumps(data, indent=JSON_INDENT)


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
        brief = GenerationBrief.for_root(schema.root)
        return render_blocks(
            [
                PromptBlock(section=GenerationSection.ROLE, body=brief.role),
                PromptBlock(
                    section=GenerationSection.INPUTS,
                    body="\n".join(
                        f"- <{section}>: {meaning}"
                        for section, meaning in INPUT_MEANINGS.items()
                    ),
                ),
                PromptBlock(
                    section=GenerationSection.RULES,
                    body="\n".join(
                        rule.render()
                        for rule in (*brief.rules, *generation_rules(root))
                    ),
                ),
                PromptBlock(
                    section=GenerationSection.TYPES,
                    body=as_json(
                        [
                            type_entry(type_schema)
                            for _, type_schema in sorted(schema.types.items())
                        ]
                    ),
                ),
                PromptBlock(
                    section=GenerationSection.STRUCTURE_RULES,
                    body=as_json(
                        [
                            structure_rule_entry(shape)
                            for shape in self.shapes.applicable_to(
                                type_schema.python_type
                                for type_schema in schema.types.values()
                            )
                        ]
                    ),
                ),
                PromptBlock(
                    section=GenerationSection.OUTPUT, body=as_json(schema.json_schema())
                ),
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
        blocks = [PromptBlock(section=GenerationSection.INSTRUCTION, body=instruction)]
        if fixed_values:
            blocks.append(
                PromptBlock(
                    section=GenerationSection.FIXED_VALUES,
                    body=as_json(
                        {
                            name: written_value(value)
                            for name, value in fixed_values.items()
                        }
                    ),
                )
            )
        return render_blocks(blocks)

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
        return SECTION_SEPARATOR.join(
            [
                initial_prompt,
                render_blocks(
                    [
                        PromptBlock(
                            section=GenerationSection.REJECTED_ANSWER, body=answer
                        ),
                        PromptBlock(
                            section=GenerationSection.PROBLEMS,
                            body=as_json(list(problems)),
                        ),
                    ]
                ),
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
