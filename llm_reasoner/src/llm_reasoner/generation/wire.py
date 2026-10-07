"""
Turning a language model's answer into instances of the generated vocabulary.

An answer is a flat list of fragments, each with an identifier and a type, that refer to
one another by identifier. Nesting cannot express the cycles phases and events form, so
fragments are created first and linked second. A malformed answer is not an exception:
its problems are collected and returned, because they are what the model is shown when
it is asked to try again.
"""

from __future__ import annotations

import json
import re
from operator import attrgetter
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Any, Mapping, Optional

from llm_reasoner.acceptance.vocabulary import Vocabulary
from llm_reasoner.generation.schema import FieldKind, FieldSchema, GenerationSchema
from llm_reasoner.ontologies import CRAM_AICOR_NAMESPACE, DUL_NAMESPACE
from llm_reasoner.ontologies.terms import ConceptReference, DescriptionEntity

JSON_OBJECT = re.compile(r"\{.*}", re.DOTALL)
"""
Locates the JSON object inside a response that may carry surrounding text.
"""

CONCEPT_PREFIXES: Mapping[str, str] = {
    "aicor": CRAM_AICOR_NAMESPACE,
    "dul": DUL_NAMESPACE,
}
"""
The namespace each prefix a model may write in a concept stands for.
"""

IDENTIFIER_KEY = "id"
"""
The key of a fragment's identifier.
"""

TYPE_KEY = "type"
"""
The key of a fragment's type.
"""

FRAGMENTS_KEY = "fragments"
"""
The key of the answer's list of fragments.
"""


# %% results


@dataclass
class DecodedHypothesis:
    """
    The fragments of one answer, linked into instances of the vocabulary.
    """

    root: DescriptionEntity
    """
    The fragment of the type being generated.
    """

    fragments: list[DescriptionEntity]
    """
    Every fragment, in the order the answer lists them.
    """

    identifiers: dict[int, str] = field(default_factory=dict)
    """
    The answer's identifier of each fragment, keyed by the fragment's identity.
    """

    def identifier_of(self, fragment: Any) -> str:
        """
        :return: The answer's identifier of a fragment, or an empty string for a value
            the answer did not list.
        """
        return self.identifiers.get(id(fragment), "")


@dataclass
class DecodingResult:
    """
    The outcome of decoding one answer: a hypothesis, or the problems preventing one.
    """

    hypothesis: Optional[DecodedHypothesis] = None
    """
    The decoded hypothesis, when the answer could be decoded.
    """

    problems: list[str] = field(default_factory=list)
    """
    What is wrong with the answer, each naming the fragment and field at fault.
    """

    partial: Optional[DecodedHypothesis] = None
    """
    The fragments of an answer with problems, linked wherever their values could be
    read, when the answer has its one root fragment; it is checked but never admitted.
    """


# %% decoder


@dataclass
class FragmentDecoder:
    """
    Decodes answers against a generation schema.
    """

    schema: GenerationSchema
    """
    The fragment types the answer may use.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary, read for the components a structure declares and the
    values classes entail.
    """

    def decode(self, raw_response: str) -> DecodingResult:
        """
        :return: The hypothesis the answer describes, or the problems preventing one
            together with what of the answer could be linked.
        """
        entries, problems = self.entries(raw_response)
        usable = self.usable_entries(entries)
        instances = {
            entry[IDENTIFIER_KEY]: self.schema.get(entry[TYPE_KEY]).python_type()
            for entry in usable
        }
        for entry in usable:
            problems.extend(self.link(entry, instances))
        roots = [
            instance
            for instance in instances.values()
            if isinstance(instance, self.schema.root)
        ]
        if len(roots) != 1:
            problems.append(
                f"the answer has {len(roots)} fragments of type "
                f"{self.schema.root.__name__}, but must have exactly one"
            )
            return DecodingResult(problems=problems)
        for instance in instances.values():
            self.fill_entailed_values(instance)
        self.declare_components(roots[0], list(instances.values()))
        hypothesis = DecodedHypothesis(
            root=roots[0],
            fragments=list(instances.values()),
            identifiers={
                id(instance): identifier for identifier, instance in instances.items()
            },
        )
        if problems:
            return DecodingResult(problems=problems, partial=hypothesis)
        return DecodingResult(hypothesis=hypothesis)

    def entries(self, raw_response: str) -> tuple[list[dict[str, Any]], list[str]]:
        """
        :return: The answer's fragment entries, and the problems with their identifiers
            and types.

        .. note::
            A fragment list written as one JSON string, as a model answering through a
            forced tool call sometimes writes it, is read as the list the string begins
            with; anything the model wrote after that list is ignored.
        """
        located = JSON_OBJECT.search(raw_response)
        if located is None:
            return [], ["the answer contains no JSON object"]
        payload = json.loads(located.group(0))
        entries = payload.get(FRAGMENTS_KEY) if isinstance(payload, dict) else None
        if isinstance(entries, str):
            entries, _ = json.JSONDecoder().raw_decode(entries.lstrip())
        if not isinstance(entries, list):
            return [], [f'the answer has no "{FRAGMENTS_KEY}" list']
        problems = []
        seen: set[str] = set()
        for position, entry in enumerate(entries):
            identifier = entry.get(IDENTIFIER_KEY) if isinstance(entry, dict) else None
            if not isinstance(identifier, str) or not identifier:
                problems.append(f"fragment {position} has no string identifier")
                continue
            if identifier in seen:
                problems.append(f"identifier '{identifier}' is used twice")
            seen.add(identifier)
            if self.schema.get(entry.get(TYPE_KEY, "")) is None:
                problems.append(
                    f"fragment '{identifier}' has unknown type '{entry.get(TYPE_KEY)}'"
                )
        return entries, problems

    def declaring_types(self, name: str) -> str:
        """
        :return: Where a field an answer put on the wrong type belongs, in words, or
            nothing when no type of the schema declares it.
        """
        declaring = sorted(
            type_schema.name
            for type_schema in self.schema.types.values()
            if type_schema.field_named(name) is not None
            and name in vars(type_schema.python_type).get("__annotations__", {})
        )
        if not declaring:
            return ""
        return f"; it is a field of {', '.join(declaring)}"

    def usable_entries(self, entries: list[Any]) -> list[dict[str, Any]]:
        """
        :return: The entries with an identifier not used before and a known type, which
            become fragments even when the answer has problems.
        """
        seen: set[str] = set()
        usable = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            identifier, type_name = entry.get(IDENTIFIER_KEY), entry.get(TYPE_KEY)
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in seen
                or not isinstance(type_name, str)
                or self.schema.get(type_name) is None
            ):
                continue
            seen.add(identifier)
            usable.append(entry)
        return usable

    def link(
        self, entry: dict[str, Any], instances: dict[str, DescriptionEntity]
    ) -> list[str]:
        """
        Set the fields of one fragment from its entry.

        :return: The problems with the entry's fields.
        """
        identifier = entry[IDENTIFIER_KEY]
        type_schema = self.schema.get(entry[TYPE_KEY])
        instance = instances[identifier]
        problems = []
        for name, written in entry.items():
            if name in (IDENTIFIER_KEY, TYPE_KEY):
                continue
            field_schema = type_schema.field_named(name)
            if field_schema is None and name in self.vocabulary.component_fields(
                type(instance)
            ):
                continue
            if field_schema is None:
                problems.append(
                    f"fragment '{identifier}' of type {type_schema.name} has no field "
                    f"'{name}'" + self.declaring_types(name)
                )
                continue
            values = written if isinstance(written, list) else [written]
            if not field_schema.many and len(values) > 1:
                problems.append(
                    f"field '{name}' of fragment '{identifier}' takes one value, not "
                    f"{len(values)}"
                )
                continue
            converted = [self.value(field_schema, value, instances) for value in values]
            problems.extend(
                f"field '{name}' of fragment '{identifier}': {problem}"
                for problem in converted
                if isinstance(problem, Problem)
            )
            if any(isinstance(value, Problem) for value in converted):
                continue
            setattr(instance, name, converted if field_schema.many else converted[0])
        return problems

    def fill_entailed_values(self, instance: DescriptionEntity) -> None:
        """
        Give a fragment the enumeration members and literals its classes entail, where
        the answer states nothing else.
        """
        type_schema = self.schema.get(type(instance).__name__)
        for name, value in self.vocabulary.entailed_values(type(instance)).items():
            field_schema = type_schema.field_named(name)
            if field_schema is None or field_schema.kind == FieldKind.FRAGMENT:
                continue
            stated = attrgetter(name)(instance)
            if not field_schema.many and stated is None:
                setattr(instance, name, value)
            elif field_schema.many and value not in stated:
                setattr(instance, name, [*stated, value])

    def declare_components(
        self, root: DescriptionEntity, fragments: list[DescriptionEntity]
    ) -> None:
        """
        Declare every fragment of the answer as a component of the root, through the
        most specific of the root's component fields accepting it; a fragment no
        component field accepts, such as a magnitude range, is a value and stays
        undeclared.
        """
        component_fields = self.vocabulary.component_fields(type(root))
        declared: dict[str, list[DescriptionEntity]] = {
            name: [] for name in component_fields
        }
        for fragment in fragments:
            if fragment is root:
                continue
            accepting = {
                name: component_type
                for name, component_type in component_fields.items()
                if isinstance(fragment, component_type)
            }
            for name, component_type in accepting.items():
                if not any(
                    other is not component_type and issubclass(other, component_type)
                    for other in accepting.values()
                ):
                    declared[name].append(fragment)
        for name, components in declared.items():
            if components:
                setattr(root, name, components)

    def value(
        self,
        field_schema: FieldSchema,
        written: Any,
        instances: dict[str, DescriptionEntity],
    ) -> Any:
        """
        :return: The Python value a written value stands for, or the problem with it.
        """
        if field_schema.kind == FieldKind.FRAGMENT:
            referenced = instances.get(written) if isinstance(written, str) else None
            if referenced is None:
                return Problem(f"'{written}' is not the identifier of a fragment")
            if type(referenced).__name__ not in field_schema.accepted_types:
                return Problem(
                    f"'{written}' is a {type(referenced).__name__}, not a "
                    f"{field_schema.value_type.__name__}; the field accepts only "
                    + ", ".join(field_schema.accepted_types)
                    + f". {field_schema.name}: {field_schema.description}"
                )
            return referenced
        if field_schema.kind == FieldKind.ENUMERATION:
            return self.member(field_schema.value_type, written)
        if field_schema.kind == FieldKind.CONCEPT:
            return self.concept(written)
        if field_schema.kind == FieldKind.NUMBER:
            if isinstance(written, bool) or not isinstance(written, (int, float)):
                return Problem(f"{written!r} is not a number")
            return float(written)
        if field_schema.kind == FieldKind.TRUTH_VALUE:
            return (
                written
                if isinstance(written, bool)
                else Problem(f"{written!r} is not true or false")
            )
        return (
            written if isinstance(written, str) else Problem(f"{written!r} is not text")
        )

    @staticmethod
    def member(enumeration: type[Enum], written: Any) -> Any:
        """
        :return: The enumeration member a written name stands for, or the problem.
        """
        for candidate in enumeration:
            if written in (candidate.name, candidate.value):
                return candidate
        return Problem(
            f"{written!r} is not one of "
            + ", ".join(candidate.name for candidate in enumeration)
        )

    @staticmethod
    def concept(written: Any) -> Any:
        """
        :return: The concept reference a prefixed name or IRI stands for, or the
            problem.
        """
        if isinstance(written, str) and written.startswith(("http://", "https://")):
            return ConceptReference(iri=written)
        prefix, _, local = (
            written.partition(":") if isinstance(written, str) else ("", "", "")
        )
        if prefix in CONCEPT_PREFIXES and local:
            return ConceptReference(iri=CONCEPT_PREFIXES[prefix] + local)
        return Problem(
            f"{written!r} is not a prefixed term such as aicor:Pouring or an IRI"
        )


@dataclass(frozen=True)
class Problem:
    """
    What is wrong with one written value.
    """

    description: str
    """
    The problem, in words.
    """

    def __str__(self) -> str:
        return self.description
