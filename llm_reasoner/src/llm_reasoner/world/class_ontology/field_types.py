"""
What a field of a class holds: a plain value, an individual of a class, a member of an
enumeration, or a type that cannot be made out, and whether it may be left empty or
holds several.
"""

from __future__ import annotations

import ast
import collections.abc
import sys
import types
import typing
from dataclasses import Field, dataclass, field
from enum import StrEnum

from krrood.class_diagrams.exceptions import CouldNotResolveType
from krrood.class_diagrams.utils import get_type_hints_of_object
from typing_extensions import Annotated, Any, Dict, Optional, Union, Unpack

from llm_reasoner.world.classes import identifier_of, written_type


class PlainType(StrEnum):
    """
    The plain values a field may hold.
    """

    BOOLEAN = "bool"
    INTEGER = "int"
    FLOAT = "float"
    TEXT = "str"


PLAIN_TYPES = {
    bool: PlainType.BOOLEAN,
    int: PlainType.INTEGER,
    float: PlainType.FLOAT,
    str: PlainType.TEXT,
}
"""
The plain value each Python type holds.
"""

COLLECTION_ORIGINS = (
    list,
    set,
    frozenset,
    tuple,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Set,
    collections.abc.MutableSet,
    collections.abc.Iterable,
    collections.abc.Collection,
)
"""
The generic types whose values hold several of what they are parametrised with.
"""

UNION_ORIGINS = (Union, types.UnionType)

TYPE_VARIABLES = (typing.TypeVar, typing.TypeVarTuple)
"""
The kinds of type variable: one standing for a type, and one standing for several.
"""

"""
The forms a union of types is written in.
"""

COLLECTION_NAMES = frozenset(
    {
        "List",
        "list",
        "Set",
        "set",
        "FrozenSet",
        "frozenset",
        "Tuple",
        "tuple",
        "Sequence",
        "MutableSequence",
        "Iterable",
        "Collection",
    }
)
"""
The names a collection is written with when its type is read as text.
"""

ORIGINAL_BASES = "__orig_bases__"
"""
The class attribute holding the bases of a class as written, generic arguments included.
"""

TYPE_PARAMETERS = "__parameters__"
"""
The class attribute holding the type variables a generic class is parametrised with.
"""

OPTIONAL_NAME = "Optional"
"""
The name an optional type is written with when it is read as text.
"""

FOUNDATION_MODULES = frozenset(sys.stdlib_module_names) | {"typing_extensions"}
"""
The modules whose classes are part of the language rather than of the code base, which a
field holding one does not refer to as a described class.
"""


def unpacked(argument: Any) -> Any:
    """
    :return: The variadic type variable an argument unpacks, as a subclass passing one on
        unchanged writes it; the argument itself otherwise.
    """
    if typing.get_origin(argument) is Unpack:
        return typing.get_args(argument)[0]
    return argument


@dataclass(frozen=True)
class FieldType:
    """
    What a field holds.
    """

    written: str
    """
    The type as the field's annotation writes it in the source.
    """

    plain: Optional[PlainType] = None
    """
    The plain value it holds; none when it holds an individual or an unknown type.
    """

    referenced: Optional[type] = None
    """
    The class or enumeration whose individuals or members it holds; none for a plain
    value or a type that cannot be made out.
    """

    optional: bool = False
    """
    Whether it may be left empty.
    """

    collection: bool = False
    """
    Whether it holds several.
    """

    @property
    def target(self) -> Optional[str]:
        """
        The module and name of the class or enumeration it refers to, if any.
        """
        return None if self.referenced is None else identifier_of(self.referenced)


@dataclass
class FieldTypeReader:
    """
    Reads what the fields of a class hold, from their resolved type hints, or from the
    type as written when the class's hints cannot be resolved, as when one names a type
    imported only for type checking.
    """

    described: type
    """
    The class whose fields are read.
    """

    hints: Dict[str, Any] = field(init=False)
    """
    The resolved type of every field, empty when they cannot be resolved.
    """

    names: Dict[str, Any] = field(init=False)
    """
    What the names a type is written with stand for, in the modules of the class and of
    the classes it specialises.
    """

    bindings: Dict[Any, Any] = field(init=False)
    """
    The type each type variable stands for in the class, as its generic bases fix it; a
    variadic one only when they fix it to a single type.
    """

    def __post_init__(self) -> None:
        self.hints = self.resolved_hints(self.described)
        self.bindings = self.type_variable_bindings(self.described)
        self.names = {}
        for ancestor in reversed(self.described.__mro__):
            module = sys.modules.get(ancestor.__module__)
            if module is not None:
                self.names.update(vars(module))

    @staticmethod
    def resolved_hints(described: type) -> Dict[str, Any]:
        """
        :return: The resolved type of every field; empty when one cannot be resolved,
            which leaves the types as written.
        """
        try:
            return get_type_hints_of_object(described)
        except CouldNotResolveType:
            return {}

    @staticmethod
    def type_variable_bindings(described: type) -> Dict[Any, Any]:
        """
        :return: The type each type variable stands for in a class, as the generic bases
            of it and of the classes it specialises fix it, the most specific first; a
            variadic type variable only when it is fixed to a single type.
        """
        bindings: Dict[Any, Any] = {}
        for ancestor in described.__mro__:
            for base in vars(ancestor).get(ORIGINAL_BASES, ()):
                origin = typing.get_origin(base)
                if origin is None:
                    continue
                parameters = vars(origin).get(TYPE_PARAMETERS, ())
                arguments = typing.get_args(base)
                for index, parameter in enumerate(parameters):
                    following = len(parameters) - index - 1
                    if isinstance(parameter, typing.TypeVarTuple):
                        captured = arguments[index : len(arguments) - following]
                        if len(captured) == 1:
                            bindings.setdefault(parameter, unpacked(captured[0]))
                    elif index < len(arguments):
                        bindings.setdefault(parameter, arguments[index])
        return bindings

    def bound_type(self, variable: Any) -> Any:
        """
        :return: What a type variable stands for in the class, following the type
            variables it is fixed to; the last type variable when the bases fix it to no
            type, as when a subclass passes it on unchanged.
        """
        seen = {variable}
        bound_to: Any = variable
        while isinstance(bound_to, TYPE_VARIABLES) and bound_to in self.bindings:
            following = self.bindings[bound_to]
            if following in seen:
                break
            seen.add(following)
            bound_to = following
        return bound_to

    def read(self, declared: Field) -> FieldType:
        """
        :return: What a field holds.
        """
        written = (
            declared.type
            if isinstance(declared.type, str)
            else written_type(declared.type)
        )
        annotation = self.hints.get(declared.name, declared.type)
        if isinstance(annotation, str):
            return self.of_written(
                ast.parse(annotation, mode="eval").body, FieldType(written=written)
            )
        return self.of_annotation(annotation, FieldType(written=written))

    def of_annotation(self, annotation: Any, found: FieldType) -> FieldType:
        """
        :return: What a resolved type holds, adding to what was found around it.
        """
        if isinstance(annotation, typing.TypeVar):
            bound_to = self.bound_type(annotation)
            if not isinstance(bound_to, typing.TypeVar):
                return self.of_annotation(bound_to, found)
            bound = bound_to.__bound__
            return found if bound is None else self.of_annotation(bound, found)
        if isinstance(annotation, typing.TypeVarTuple):
            bound_to = self.bound_type(annotation)
            if isinstance(bound_to, TYPE_VARIABLES):
                return found
            return self.of_annotation(bound_to, found)
        origin = typing.get_origin(annotation)
        arguments = typing.get_args(annotation)
        if origin in (Annotated, Unpack):
            return self.of_annotation(arguments[0], found)
        if origin in UNION_ORIGINS:
            present = [argument for argument in arguments if argument is not type(None)]
            optional = FieldType(
                written=found.written,
                optional=found.optional or len(present) < len(arguments),
                collection=found.collection,
            )
            if len(present) != 1:
                return optional
            return self.of_annotation(present[0], optional)
        if origin in COLLECTION_ORIGINS and arguments:
            return self.of_annotation(
                arguments[0],
                FieldType(
                    written=found.written, optional=found.optional, collection=True
                ),
            )
        return self.of_class(annotation, found)

    def of_written(self, node: ast.expr, found: FieldType) -> FieldType:
        """
        :return: What a type written as text holds, adding to what was found around it.
        """
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            present = [
                side
                for side in (node.left, node.right)
                if not (isinstance(side, ast.Constant) and side.value is None)
            ]
            optional = FieldType(
                written=found.written, optional=True, collection=found.collection
            )
            return (
                optional if len(present) != 1 else self.of_written(present[0], optional)
            )
        if isinstance(node, ast.Subscript):
            outer = self.last_name(node.value)
            if outer == OPTIONAL_NAME:
                return self.of_written(
                    node.slice,
                    FieldType(
                        written=found.written,
                        optional=True,
                        collection=found.collection,
                    ),
                )
            if outer in COLLECTION_NAMES:
                inner = (
                    node.slice.elts[0]
                    if isinstance(node.slice, ast.Tuple)
                    else node.slice
                )
                return self.of_written(
                    inner,
                    FieldType(
                        written=found.written, optional=found.optional, collection=True
                    ),
                )
            return found
        name = self.last_name(node)
        if name is None:
            return found
        return self.of_annotation(self.names.get(name, PLAIN_NAMES.get(name)), found)

    @staticmethod
    def last_name(node: ast.expr) -> Optional[str]:
        """
        :return: The name a written type ends in, such as ``Pose`` for ``spatial.Pose``.
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @staticmethod
    def of_class(candidate: Any, found: FieldType) -> FieldType:
        """
        :return: What a field holding instances of a class holds: a plain value, an
            individual or member of the class, or nothing made out for a class of the
            language itself.
        """
        if candidate in PLAIN_TYPES:
            return FieldType(
                written=found.written,
                plain=PLAIN_TYPES[candidate],
                optional=found.optional,
                collection=found.collection,
            )
        if not isinstance(candidate, type) or (
            candidate.__module__.split(".")[0] in FOUNDATION_MODULES
        ):
            return found
        return FieldType(
            written=found.written,
            referenced=candidate,
            optional=found.optional,
            collection=found.collection,
        )


PLAIN_NAMES = {plain.value: python_type for python_type, plain in PLAIN_TYPES.items()}
"""
The Python type each plain type's name stands for, when a type is read as text.
"""
