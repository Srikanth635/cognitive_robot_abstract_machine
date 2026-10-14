"""
Reading what a property field's type hint says it holds.

Hints are resolved one field at a time, in the namespace of the module that declares the
field. Resolving a whole class at once fails as soon as any base class carries an
annotation that cannot be resolved, which would hide every other field.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import sys
import types
import typing
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin

from llm_reasoner.exceptions import LLMReasonerError

COLLECTION_TYPES = (list, set, frozenset, tuple)
"""
The container types that mark a property with many values.
"""

UNION_ORIGINS = (Union, types.UnionType)
"""
The forms in which a union of types can be written.
"""

# %% resolved ranges


@dataclass(frozen=True)
class PropertyRange:
    """
    What a field's type hint says the property holds.
    """

    element_types: tuple[type, ...] = ()
    """
    The classes a single value may belong to.

    Empty when the hint does not say.
    """

    collection: bool = False
    """
    Whether the hint is a container of values.
    """

    optional: bool = False
    """
    Whether the hint allows ``None``.
    """

    unresolved_names: tuple[str, ...] = ()
    """
    Names the hint uses that are defined nowhere it can see.

    When any exist, the other fields carry no information.
    """

    @property
    def resolved(self) -> bool:
        """
        Whether every name in the hint was found.
        """
        return not self.unresolved_names


@dataclass
class UndeclaredField(LLMReasonerError):
    """
    A range was requested for a field the class does not declare in its own body.
    """

    owner: type
    """
    The class that was asked about.
    """

    field_name: str
    """
    The field that it does not declare.
    """

    def __str__(self) -> str:
        return f"{self.owner.__qualname__} does not declare a field '{self.field_name}'"


# %% resolution


class NameCollector(ast.NodeVisitor):
    """
    Collects the names an annotation uses, including those inside quoted parts.
    """

    def __init__(self) -> None:
        self.names: list[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        """
        Record a name.
        """
        self.names.append(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        """
        Read a quoted part of the annotation as an annotation of its own.
        """
        if isinstance(node.value, str):
            self.visit(ast.parse(node.value, mode="eval"))


@dataclass(frozen=True)
class RangeResolver:
    """
    Resolves the range of the fields a class declares.
    """

    def resolve(self, owner: type, field_name: str) -> PropertyRange:
        """
        Return the range of a field written in the class's own body.

        :raises UndeclaredField: when the class does not annotate the field itself.
        """
        annotations = inspect.get_annotations(owner)
        if field_name not in annotations:
            raise UndeclaredField(owner, field_name)
        annotation = annotations[field_name]
        if not isinstance(annotation, str):
            return self._decompose(annotation)

        namespace = vars(sys.modules[owner.__module__])
        local_names = {owner.__name__: owner}
        unresolved = self._unresolved_names(annotation, namespace, local_names)
        if unresolved:
            return PropertyRange(unresolved_names=unresolved)
        hints = typing.get_type_hints(
            types.SimpleNamespace(__annotations__={field_name: annotation}),
            globalns=namespace,
            localns=local_names,
        )
        return self._decompose(hints[field_name])

    @staticmethod
    def _unresolved_names(
        annotation: str, namespace: dict[str, Any], local_names: dict[str, Any]
    ) -> tuple[str, ...]:
        """
        Return the names in an annotation that neither the module nor Python defines.
        """
        collector = NameCollector()
        collector.visit(ast.parse(annotation, mode="eval"))
        known = {*namespace, *local_names, *vars(builtins)}
        return tuple(dict.fromkeys(n for n in collector.names if n not in known))

    def _decompose(self, hint: Any) -> PropertyRange:
        """
        Split a resolved hint into its element types, cardinality and optionality.
        """
        members, optional = self._without_none(hint)
        collection = len(members) == 1 and self._is_collection(members[0])
        if collection:
            members = tuple(
                member
                for argument in get_args(members[0])
                for member in self._without_none(argument)[0]
            )
        return PropertyRange(
            element_types=tuple(
                dict.fromkeys(
                    member
                    for member in members
                    if isinstance(member, type) and member is not Any
                )
            ),
            collection=collection,
            optional=optional,
        )

    @staticmethod
    def _without_none(hint: Any) -> tuple[tuple[Any, ...], bool]:
        """
        Split a hint into the members of its union other than ``None``.

        :return: The members, and whether ``None`` was one of them.
        """
        alternatives = get_args(hint) if get_origin(hint) in UNION_ORIGINS else (hint,)
        return (
            tuple(a for a in alternatives if a is not type(None)),
            type(None) in alternatives,
        )

    @staticmethod
    def _is_collection(hint: Any) -> bool:
        """
        Return whether a hint is a container, written bare or with element types.
        """
        return hint in COLLECTION_TYPES or get_origin(hint) in COLLECTION_TYPES
