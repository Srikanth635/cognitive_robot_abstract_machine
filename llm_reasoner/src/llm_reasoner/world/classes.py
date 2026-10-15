"""
What a class says about itself: its summary, the meaning and type of its fields, and
which of its fields hold content.
"""

from __future__ import annotations

import inspect
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from functools import cache, cached_property
from operator import attrgetter

from krrood.patterns.field_metadata import JSONMetadata
from typing_extensions import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    Union,
)

from llm_reasoner.attribute_docstrings import attribute_docstrings
from llm_reasoner.world.description import ClassSummary

PLAIN_TYPES = (bool, int, float, str, Enum, type(None))
"""
Types whose values are given as they are.
"""

PRIVATE_PREFIX = "_"
"""
The prefix of a field that is bookkeeping rather than content.
"""

SLOTS_ATTRIBUTE = "__slots__"
"""
The class attribute declaring that instances keep no attribute dictionary.
"""


class FoundationModule(StrEnum):
    """
    Modules whose classes every class specialises and which say nothing about what an
    individual is, so a lineage leaves them out.
    """

    BUILTINS = "builtins"
    ABSTRACT_BASE_CLASSES = "abc"
    TYPING = "typing"
    TYPING_EXTENSIONS = "typing_extensions"


@dataclass(frozen=True)
class FieldDefinition:
    """
    A field as its class declares it.
    """

    name: str
    """
    The name of the field.
    """

    type_name: str
    """
    The type the field declares, as written.
    """

    meaning: str
    """
    What the field holds, from its docstring; empty when it has none.
    """

    default: Optional[str] = None
    """
    The plain default the field declares, written out; ``None`` when it declares none or
    makes one with a factory.
    """


def is_individual(value: Any) -> bool:
    """
    :return: Whether a value is an instance of a dataclass, and so has fields to walk.
    """
    return is_dataclass(value) and not isinstance(value, type)


def identifier_of(declaration: Union[type, Callable[..., Any]]) -> str:
    """
    :return: The module and qualified name of a class or function, which tell it from
        others of its name.
    """
    return f"{declaration.__module__}.{declaration.__qualname__}"


def written_type(annotation: Any) -> str:
    """
    :return: A type annotation as written: the text of one given as text, the name of a
        class, and the representation of anything else.
    """
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


@dataclass(frozen=True)
class DocumentedClass:
    """
    What a class says about itself.
    """

    described: type
    """
    The class.
    """

    @classmethod
    @cache
    def of(cls, described: type) -> DocumentedClass:
        """
        :return: The documentation of a class, read once per class.
        """
        return cls(described=described)

    @cached_property
    def summary(self) -> str:
        """
        The first line of the docstring, empty when the class has none of its own.

        The signature a dataclass writes for an undocumented class does not count.
        """
        docstring = self.described.__doc__
        if not docstring or docstring.startswith(self.described.__name__ + "("):
            return ""
        return inspect.cleandoc(docstring).splitlines()[0]

    @cached_property
    def documentation(self) -> str:
        """
        The whole docstring, empty when the class has none of its own.

        The signature a dataclass writes for an undocumented class does not count.
        """
        docstring = self.described.__doc__
        if not docstring or docstring.startswith(self.described.__name__ + "("):
            return ""
        return inspect.cleandoc(docstring)

    @cached_property
    def meanings(self) -> Dict[str, str]:
        """
        The whole docstring of every field, inherited ones included, by field name.
        """
        meanings: Dict[str, str] = {}
        for ancestor in reversed(self.described.__mro__):
            if is_dataclass(ancestor):
                meanings.update(attribute_docstrings(ancestor))
        return meanings

    @cached_property
    def declared_content_fields(self) -> Tuple[Field, ...]:
        """
        The fields that may hold content: not private, and not excluded from
        serialisation, which is how a class marks a field as derived or as bookkeeping.
        """
        if not is_dataclass(self.described):
            return ()
        return tuple(
            declared
            for declared in fields(self.described)
            if not declared.name.startswith(PRIVATE_PREFIX)
            and self.is_serialised(declared)
        )

    @cached_property
    def keeps_attribute_dictionary(self) -> bool:
        """
        Whether instances keep their attributes in a dictionary, which tells a field the
        class fills in itself from one not yet set.
        """
        return any(
            SLOTS_ATTRIBUTE not in vars(ancestor)
            for ancestor in self.described.__mro__
            if ancestor is not object
        )

    def content_fields_of(self, individual: Any) -> List[Field]:
        """
        :return: The content fields an individual holds: those whoever made it sets, and
            those the class fills in itself once they are set.
        """
        filled_in = vars(individual) if self.keeps_attribute_dictionary else {}
        return [
            declared
            for declared in self.declared_content_fields
            if declared.init or declared.name in filled_in
        ]

    def content_field_named(self, individual: Any, name: str) -> Optional[Field]:
        """
        :return: The content field of the given name an individual holds, or ``None``
            when it holds none of that name.
        """
        return next(
            (
                declared
                for declared in self.content_fields_of(individual)
                if declared.name == name
            ),
            None,
        )

    def is_serialised(self, declared: Field) -> bool:
        """
        :return: Whether the field is part of what the class serialises.
        """
        metadata = JSONMetadata.of_field(self.described, declared.name)
        return metadata is None or metadata.serialize

    @staticmethod
    def value_of(individual: Any, declared: Field) -> Any:
        """
        :return: What a field of an individual holds.

        .. note:: A walk over classes it does not know can only reach a field by its
            name, so this is the one place an individual's field is read by name.
        """
        return attrgetter(declared.name)(individual)

    def field_definitions(self) -> List[FieldDefinition]:
        """
        :return: The content fields the class declares, with their types, meanings and
            plain defaults.
        """
        return [
            FieldDefinition(
                name=declared.name,
                type_name=self.type_name_of(declared),
                meaning=self.meanings.get(declared.name, ""),
                default=(
                    repr(declared.default)
                    if declared.default is not MISSING
                    and isinstance(declared.default, PLAIN_TYPES)
                    else None
                ),
            )
            for declared in self.declared_content_fields
        ]

    @staticmethod
    def type_name_of(declared: Field) -> str:
        """
        :return: The type a field declares, as written.
        """
        return written_type(declared.type)

    @cached_property
    def specialised_classes(self) -> Tuple[type, ...]:
        """
        The class and the classes it specialises, most specific first, without the
        foundation every class shares.
        """
        foundation = {module.value for module in FoundationModule}
        return tuple(
            ancestor
            for ancestor in self.described.__mro__
            if ancestor.__module__ not in foundation
        )

    @cached_property
    def lineage_names(self) -> FrozenSet[str]:
        """
        The names of the class and of every class it specialises, without the foundation
        every class shares.
        """
        return frozenset(ancestor.__name__ for ancestor in self.specialised_classes)

    def lineage(self, length: int) -> List[ClassSummary]:
        """
        :return: The class and the classes it specialises, most specific first, without
            the foundation every class shares.
        """
        return [
            ClassSummary(
                identifier=identifier_of(ancestor),
                name=ancestor.__name__,
                summary=DocumentedClass.of(ancestor).summary,
            )
            for ancestor in self.specialised_classes
        ][:length]
