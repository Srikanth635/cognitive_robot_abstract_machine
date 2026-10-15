"""
The output shape a language model fills: every vocabulary class reachable from the type
being generated, with its fields, read from the generated classes by krrood.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum, StrEnum

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.utils import recursive_subclasses
from typing_extensions import Any, Optional

from llm_reasoner.attribute_docstrings import attribute_docstrings
from llm_reasoner.knowledge.ontologies import OntologyModuleFile
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.ontologies.terms import (
    is_inferred,
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
)

# %% answer keys


class AnswerKey(StrEnum):
    """
    The keys of an answer and of its fragments.
    """

    FRAGMENTS = "fragments"
    """
    The answer's list of fragments; in a repair, the fragments added or changed.
    """

    REMOVE = "remove"
    """
    In a repair, the identifiers of the fragments to delete.
    """

    IDENTIFIER = "id"
    """
    A fragment's identifier.
    """

    TYPE = "type"
    """
    A fragment's type.
    """


# %% field kinds


class FieldKind(StrEnum):
    """
    What a field of a fragment holds, as the model writes it.
    """

    FRAGMENT = "fragment"
    """
    The identifier of another fragment of the answer.
    """

    ENUMERATION = "enumeration"
    """
    The name of a member of a fixed list of options.
    """

    CONCEPT = "concept"
    """
    A term of a linked ontology, written as a prefixed name such as ``aicor:Pouring``.
    """

    NUMBER = "number"
    """
    A number.
    """

    TRUTH_VALUE = "truth_value"
    """
    True or false.
    """

    TEXT = "text"
    """
    Free text.
    """


LITERAL_KINDS: dict[type, FieldKind] = {
    float: FieldKind.NUMBER,
    int: FieldKind.NUMBER,
    bool: FieldKind.TRUTH_VALUE,
    str: FieldKind.TEXT,
}
"""
The field kind of each literal Python type.
"""

LABEL_FIELD = "label"
"""
The field every fragment has for a human-readable name.
"""


# %% schema


@dataclass(frozen=True)
class FieldSchema:
    """
    One field a fragment of some type may state.
    """

    name: str
    """
    The field name.
    """

    description: str
    """
    What the field means, from the ontology.
    """

    kind: FieldKind
    """
    What the field holds.
    """

    many: bool
    """
    Whether the field holds a list rather than one value.
    """

    value_type: type
    """
    The Python type of one value.
    """

    accepted_types: tuple[str, ...] = ()
    """
    The fragment types a fragment field may refer to.
    """

    options: tuple[str, ...] = ()
    """
    The member names an enumeration field may take.
    """


@dataclass(frozen=True)
class TypeSchema:
    """
    One fragment type the model may use.
    """

    python_type: type[DescriptionEntity]
    """
    The generated class.
    """

    description: str
    """
    What the type is, from the ontology.
    """

    fields: tuple[FieldSchema, ...]
    """
    Every field a fragment of the type may state, inherited ones included.
    """

    @property
    def name(self) -> str:
        """
        The type name the model writes.
        """
        return self.python_type.__name__

    def field_named(self, name: str) -> FieldSchema | None:
        """
        :return: The field of the given name, or ``None`` when the type has none.
        """
        return next((schema for schema in self.fields if schema.name == name), None)


@dataclass
class GenerationSchema:
    """
    The fragment types reachable from the type being generated.
    """

    root: type[DescriptionEntity]
    """
    The type of the fragment the answer is built toward.
    """

    types: dict[str, TypeSchema] = field(default_factory=dict)
    """
    Every usable fragment type, by name.
    """

    @classmethod
    def for_root(cls, root: type[DescriptionEntity]) -> GenerationSchema:
        """
        :return: The schema of every type reachable from the root through fields,
            including the component fields a model does not fill, together with every
            subclass of those types, as far as the root's module may use them.
        """
        vocabulary = Vocabulary()
        scope = TypeScope.of_root(root)
        schema = cls(root=root)
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for candidate in scope.generated_types(current):
                if candidate.__name__ in schema.types:
                    continue
                type_schema = type_schema_for(candidate, scope)
                schema.types[candidate.__name__] = type_schema
                frontier.extend(
                    field_schema.value_type
                    for field_schema in type_schema.fields
                    if field_schema.kind == FieldKind.FRAGMENT
                )
                frontier.extend(vocabulary.component_fields(candidate).values())
        return schema

    def module_files(self) -> set[OntologyModuleFile]:
        """
        :return: The ontology modules declaring the types and enumerations an answer may
            use.
        """
        vocabulary = Vocabulary()
        used_types = {
            used
            for type_schema in self.types.values()
            for used in (
                type_schema.python_type,
                *(
                    field_schema.value_type
                    for field_schema in type_schema.fields
                    if field_schema.kind == FieldKind.ENUMERATION
                ),
            )
        }
        return {
            module_file
            for module_file in map(vocabulary.module_file_of, used_types)
            if module_file is not None
        }

    def get(self, name: str) -> TypeSchema | None:
        """
        :return: The type of the given name, or ``None`` when the schema has none.
        """
        return self.types.get(name)

    def json_schema(self) -> dict[str, Any]:
        """
        :return: A JSON Schema constraining an answer to a list of fragments of known
            types, each with an identifier.
        """
        return {
            "type": "object",
            "required": [AnswerKey.FRAGMENTS],
            "additionalProperties": False,
            "properties": {
                AnswerKey.FRAGMENTS: {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [AnswerKey.IDENTIFIER, AnswerKey.TYPE],
                        "properties": {
                            AnswerKey.IDENTIFIER: {"type": "string"},
                            AnswerKey.TYPE: {
                                "type": "string",
                                "enum": sorted(self.types),
                            },
                        },
                    },
                },
                AnswerKey.REMOVE: {"type": "array", "items": {"type": "string"}},
            },
        }


# %% reading the generated classes


@dataclass(frozen=True)
class TypeScope:
    """
    The generated classes a hypothesis may use: those of the modules its root's module
    may be used with.
    """

    usable_modules: Optional[frozenset[OntologyModuleFile]] = None
    """
    The modules whose classes may be used; any module's when ``None``.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary, compare=False)
    """
    The generated vocabulary, which knows the module of each class.
    """

    @classmethod
    def of_root(cls, root: type[DescriptionEntity]) -> TypeScope:
        """
        :return: The scope of a hypothesis rooted in a class; unrestricted for a class
            generated from no registered module.
        """
        vocabulary = Vocabulary()
        module_file = vocabulary.module_file_of(root)
        return cls(
            usable_modules=(
                None if module_file is None else frozenset(module_file.usable_modules)
            ),
            vocabulary=vocabulary,
        )

    def admits(self, python_type: type) -> bool:
        """
        :return: Whether a hypothesis in this scope may use the class.
        """
        return (
            self.usable_modules is None
            or self.vocabulary.module_file_of(python_type) in self.usable_modules
        )

    def generated_types(
        self, base: type[DescriptionEntity]
    ) -> list[type[DescriptionEntity]]:
        """
        :return: The generated classes in scope that are the base or specialise it, each
            once, in name order.
        """
        return [
            candidate for candidate in generated_types(base) if self.admits(candidate)
        ]


UNRESTRICTED = TypeScope()
"""
The scope admitting every generated class.
"""


def type_schema_for(
    python_type: type[DescriptionEntity], scope: TypeScope = UNRESTRICTED
) -> TypeSchema:
    """
    :return: The schema of one generated class, read through krrood.
    """
    docstrings: dict[str, str] = {}
    for owner in reversed(python_type.__mro__):
        if isinstance(owner, type) and issubclass(owner, DescriptionEntity):
            docstrings.update(attribute_docstrings(owner))
    vocabulary = Vocabulary()
    fields = []
    for wrapped in WrappedClass(python_type).fields:
        name = wrapped.field.name
        term = PropertyTerm.of_field(python_type, name)
        if name != LABEL_FIELD and (
            term is None
            or is_inferred(python_type, name)
            or name in vocabulary.component_fields(python_type)
        ):
            continue
        many = term is not None and not term.functional
        fields.append(
            field_schema_for(
                name,
                docstrings.get(name, ""),
                vocabulary.value_type_of(python_type, name, wrapped.type_endpoint),
                many,
                scope,
            )
        )
    return TypeSchema(
        python_type=python_type,
        description=inspect.getdoc(python_type) or "",
        fields=tuple(fields),
    )


def generated_types(base: type[DescriptionEntity]) -> list[type[DescriptionEntity]]:
    """
    :return: The generated classes that are the base or specialise it, each once, in
        name order.
    """
    candidates = {base, *recursive_subclasses(base)}
    return sorted(
        (candidate for candidate in candidates if "term" in vars(candidate)),
        key=lambda candidate: candidate.__name__,
    )


def field_schema_for(
    name: str,
    description: str,
    value_type: type,
    many: bool,
    scope: TypeScope = UNRESTRICTED,
) -> FieldSchema:
    """
    :return: The schema of one field, from the value type krrood resolves for it.
    """
    if isinstance(value_type, type) and issubclass(value_type, DescriptionEntity):
        return FieldSchema(
            name=name,
            description=description,
            kind=FieldKind.FRAGMENT,
            many=many,
            value_type=value_type,
            accepted_types=tuple(
                accepted.__name__ for accepted in scope.generated_types(value_type)
            ),
        )
    if isinstance(value_type, type) and issubclass(value_type, Enum):
        return FieldSchema(
            name=name,
            description=description,
            kind=FieldKind.ENUMERATION,
            many=many,
            value_type=value_type,
            options=tuple(member.name for member in value_type),
        )
    if value_type is ConceptReference:
        return FieldSchema(
            name=name,
            description=description,
            kind=FieldKind.CONCEPT,
            many=many,
            value_type=value_type,
        )
    return FieldSchema(
        name=name,
        description=description,
        kind=LITERAL_KINDS[value_type],
        many=many,
        value_type=value_type,
    )
