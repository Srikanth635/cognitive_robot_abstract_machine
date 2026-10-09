"""
The predicates and value functions a world model declares as krrood classes: which there
are, what they take, and what they say about given individuals.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import MISSING, dataclass

from krrood.entity_query_language.predicate import Predicate, SymbolicFunction
from typing_extensions import Any, List, Mapping, Optional, Union

from llm_reasoner.world.classes import DocumentedClass, FieldDefinition, identifier_of
from llm_reasoner.world.exceptions import (
    MissingArgument,
    NotAPredicate,
    UnknownArgument,
)
from llm_reasoner.world.naming import HandleRegistry
from llm_reasoner.world.sources import WorldSource

EVALUABLE_KINDS = (Predicate, SymbolicFunction)
"""
The krrood classes whose instances say something when called.
"""

TYPE_NAME_PATTERN = re.compile(r"\w+")
"""
The names a written type annotation mentions, such as ``Optional`` and ``Body`` in
``Optional[Body]``.
"""


@dataclass(frozen=True)
class IndividualArgument:
    """
    An individual passed to a predicate, named by its handle.
    """

    handle: str
    """
    The handle of the individual.
    """


@dataclass(frozen=True)
class PlainArgument:
    """
    A plain value passed to a predicate.
    """

    value: Any
    """
    The value.
    """


Argument = Union[IndividualArgument, PlainArgument]
"""
What a predicate is passed for one of its arguments.
"""


@dataclass(frozen=True)
class PredicateDefinition:
    """
    A predicate or value function as a listing names it.
    """

    identifier: str
    """
    The module and qualified name of its class.
    """

    name: str
    """
    The name of its class.
    """

    summary: str
    """
    The first line of its docstring.
    """

    arguments: List[FieldDefinition]
    """
    What it takes.
    """


@dataclass(frozen=True)
class PredicateCatalogue:
    """
    The predicates and value functions a world model declares.
    """

    source: WorldSource
    """
    The world whose kinds include the predicates.
    """

    def predicates(self, about: Optional[type] = None) -> List[PredicateDefinition]:
        """
        :param about: Only those taking an individual of this kind; every one if
            ``None``.
        :return: The predicates, ordered by identifier.
        """
        definitions = [
            self.definition(kind)
            for kind in self.source.kinds()
            if self.is_evaluable(kind)
        ]
        if about is not None:
            accepted = {ancestor.__name__ for ancestor in about.__mro__}
            definitions = [
                definition
                for definition in definitions
                if any(
                    accepted & set(TYPE_NAME_PATTERN.findall(argument.type_name))
                    for argument in definition.arguments
                )
            ]
        return sorted(definitions, key=lambda definition: definition.identifier)

    @staticmethod
    def is_evaluable(kind: type) -> bool:
        """
        :return: Whether a kind is a concrete predicate or value function.
        """
        return issubclass(kind, EVALUABLE_KINDS) and not inspect.isabstract(kind)

    @staticmethod
    def definition(kind: type) -> PredicateDefinition:
        """
        :return: The definition of a predicate: what it takes, read from its fields.
        """
        documented = DocumentedClass.of(kind)
        taken = {
            declared.name
            for declared in documented.declared_content_fields
            if declared.init
        }
        return PredicateDefinition(
            identifier=identifier_of(kind),
            name=kind.__name__,
            summary=documented.summary,
            arguments=[
                argument
                for argument in documented.field_definitions()
                if argument.name in taken
            ],
        )

    def evaluate(
        self,
        kind: type,
        arguments: Mapping[str, Argument],
        handles: HandleRegistry,
    ) -> Any:
        """
        :param kind: A predicate or value function.
        :param arguments: What it is passed, by argument name.
        :param handles: Resolves the individuals passed by handle.
        :return: What it says about the arguments.
        :raises NotAPredicate: When the kind is neither.
        :raises UnknownArgument: When an argument is not one it takes.
        :raises MissingArgument: When an argument it requires is not passed.
        """
        if not self.is_evaluable(kind):
            raise NotAPredicate(identifier=identifier_of(kind))
        taken = {
            declared.name: declared
            for declared in DocumentedClass.of(kind).declared_content_fields
            if declared.init
        }
        for name in arguments:
            if name not in taken:
                raise UnknownArgument(predicate=identifier_of(kind), argument=name)
        for name, declared in taken.items():
            required = (
                declared.default is MISSING and declared.default_factory is MISSING
            )
            if required and name not in arguments:
                raise MissingArgument(predicate=identifier_of(kind), argument=name)
        values = {
            name: (
                handles.individual_of(argument.handle)
                if isinstance(argument, IndividualArgument)
                else argument.value
            )
            for name, argument in arguments.items()
        }
        return kind._construct_normally_(**values)()
