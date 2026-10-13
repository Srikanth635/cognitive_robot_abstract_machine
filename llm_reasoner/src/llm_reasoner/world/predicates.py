"""
The predicates and value functions a world model declares, as krrood classes or as
functions: which there are, what they take, and what they say about given individuals.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import MISSING, Field, dataclass
from functools import cached_property

from krrood.entity_query_language.predicate import Predicate, SymbolicFunction
from typing_extensions import Any, FrozenSet, List, Mapping, Optional, Union

from llm_reasoner.world.classes import DocumentedClass, FieldDefinition, identifier_of
from llm_reasoner.world.exceptions import (
    AmbiguousPredicate,
    MissingArgument,
    NotAPredicate,
    UnknownArgument,
)
from llm_reasoner.world.functions import NAMEABLE_KINDS, WorldFunction
from llm_reasoner.world.kinds import KindCatalogue
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

PLAIN_ARGUMENT_TYPE_NAMES = frozenset(
    plain.__name__ for plain in (bool, int, float, str)
)
"""
The names of the types a function can be passed a plain value of.
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


# %% declared predicates


class DeclaredPredicate(ABC):
    """
    A predicate or value function the world model declares, as the catalogue lists and
    evaluates it.
    """

    @property
    @abstractmethod
    def identifier(self) -> str:
        """
        The module and qualified name of the predicate, which tell it from others of its
        name.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the predicate.
        """

    @property
    def is_offered(self) -> bool:
        """
        Whether the world can be asked it: everything it must be passed is something the
        world has.
        """
        return True

    @property
    @abstractmethod
    def taken(self) -> List[str]:
        """
        The names of the arguments it takes.
        """

    @property
    @abstractmethod
    def required(self) -> List[str]:
        """
        The names of the arguments it must be passed.
        """

    @abstractmethod
    def definition(self) -> PredicateDefinition:
        """
        :return: The predicate as a listing names it.
        """

    @abstractmethod
    def call(self, values: Mapping[str, Any]) -> Any:
        """
        :param values: What each argument is, by name.
        :return: What the predicate says about them.
        """

    def evaluate(
        self, arguments: Mapping[str, Argument], handles: HandleRegistry
    ) -> Any:
        """
        :param arguments: What it is passed, by argument name.
        :param handles: Resolves the individuals passed by handle.
        :return: What it says about the arguments.
        :raises UnknownArgument: When an argument is not one it takes.
        :raises MissingArgument: When an argument it requires is not passed.
        """
        for name in arguments:
            if name not in self.taken:
                raise UnknownArgument(predicate=self.identifier, argument=name)
        for name in self.required:
            if name not in arguments:
                raise MissingArgument(predicate=self.identifier, argument=name)
        return self.call(
            {
                name: (
                    handles.individual_of(argument.handle)
                    if isinstance(argument, IndividualArgument)
                    else argument.value
                )
                for name, argument in arguments.items()
            }
        )


@dataclass(frozen=True)
class PredicateClass(DeclaredPredicate):
    """
    A predicate or value function the world model declares as a krrood class, taking
    what its fields hold.
    """

    kind: type
    """
    The class.
    """

    @staticmethod
    def is_evaluable(kind: type) -> bool:
        """
        :return: Whether a kind is a concrete predicate or value function.
        """
        return issubclass(kind, EVALUABLE_KINDS) and not inspect.isabstract(kind)

    @property
    def identifier(self) -> str:
        return identifier_of(self.kind)

    @property
    def name(self) -> str:
        return self.kind.__name__

    @cached_property
    def taken_fields(self) -> List[Field]:
        """
        The fields whose values the class is made with.
        """
        return [
            declared
            for declared in DocumentedClass.of(self.kind).declared_content_fields
            if declared.init
        ]

    @property
    def taken(self) -> List[str]:
        return [declared.name for declared in self.taken_fields]

    @property
    def required(self) -> List[str]:
        return [
            declared.name
            for declared in self.taken_fields
            if declared.default is MISSING and declared.default_factory is MISSING
        ]

    def definition(self) -> PredicateDefinition:
        documented = DocumentedClass.of(self.kind)
        taken = set(self.taken)
        return PredicateDefinition(
            identifier=self.identifier,
            name=self.name,
            summary=documented.summary,
            arguments=[
                argument
                for argument in documented.field_definitions()
                if argument.name in taken
            ],
        )

    def call(self, values: Mapping[str, Any]) -> Any:
        return self.kind._construct_normally_(**values)()


@dataclass(frozen=True)
class FunctionPredicate(DeclaredPredicate):
    """
    A predicate or value function the world model declares as a function. It takes what
    an argument can stand for: an individual of a kind the world has, or a plain value.
    """

    function: WorldFunction
    """
    The function.
    """

    held_types: FrozenSet[str]
    """
    The names of the kinds the world has individuals of, and of the kinds those
    specialise.
    """

    def can_be_passed(self, parameter: inspect.Parameter) -> bool:
        """
        :return: Whether an argument can be passed to the parameter: it is named by
            call, and declares a plain type or the type of individuals the world has.
        """
        if parameter.kind not in NAMEABLE_KINDS:
            return False
        type_name = WorldFunction.type_name_of(parameter)
        return type_name in PLAIN_ARGUMENT_TYPE_NAMES or type_name in self.held_types

    @cached_property
    def passable(self) -> List[inspect.Parameter]:
        """
        The parameters an argument can be passed to, in order.
        """
        return [
            parameter
            for parameter in self.function.parameters
            if self.can_be_passed(parameter)
        ]

    @property
    def identifier(self) -> str:
        return self.function.identifier

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def is_offered(self) -> bool:
        return all(
            self.can_be_passed(parameter)
            for parameter in self.function.parameters
            if self.function.is_required(parameter)
        )

    @property
    def taken(self) -> List[str]:
        return [parameter.name for parameter in self.passable]

    @property
    def required(self) -> List[str]:
        return [
            parameter.name
            for parameter in self.passable
            if self.function.is_required(parameter)
        ]

    def definition(self) -> PredicateDefinition:
        return PredicateDefinition(
            identifier=self.identifier,
            name=self.name,
            summary=self.function.summary,
            arguments=[
                self.function.definition_of(parameter) for parameter in self.passable
            ],
        )

    def call(self, values: Mapping[str, Any]) -> Any:
        return self.function.call(values)


# %% the catalogue


@dataclass(frozen=True)
class PredicateCatalogue:
    """
    The predicates and value functions a world model declares.
    """

    source: WorldSource
    """
    The world whose kinds include the predicate classes and whose model declares the
    functions.
    """

    kinds: KindCatalogue
    """
    Finds the kind a name means.
    """

    def classes(self) -> List[PredicateClass]:
        """
        :return: The predicates and value functions declared as classes.
        """
        return [
            PredicateClass(kind=kind)
            for kind in self.source.kinds()
            if PredicateClass.is_evaluable(kind)
        ]

    def functions(self) -> List[FunctionPredicate]:
        """
        :return: The functions the world can be asked: those whose required arguments
            are all things the world has.
        """
        held_types = self.held_types()
        return [
            function
            for function in (
                FunctionPredicate(function=declared, held_types=held_types)
                for declared in self.source.functions()
            )
            if function.is_offered
        ]

    def held_types(self) -> FrozenSet[str]:
        """
        :return: The names of the kinds the world has individuals of, and of the kinds
            those specialise.
        """
        return frozenset(
            name
            for individual in self.source.individuals()
            for name in DocumentedClass.of(type(individual)).lineage_names
        )

    def predicates(self, about: Optional[type] = None) -> List[PredicateDefinition]:
        """
        :param about: Only those taking an individual of this kind; every one if
            ``None``.
        :return: The predicates, ordered by identifier.
        """
        definitions = [
            declared.definition() for declared in [*self.classes(), *self.functions()]
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

    def resolve(self, name: str) -> DeclaredPredicate:
        """
        :param name: The identifier of a predicate, or its plain name when no other
            predicate shares it.
        :return: The predicate it names.
        :raises AmbiguousPredicate: When several predicates share the plain name.
        :raises AmbiguousKind: When several classes share the plain name.
        :raises UnknownKind: When nothing loaded has the name.
        :raises NotAPredicate: When it names a class that is neither a predicate nor a
            value function.
        """
        functions = self.functions()
        for function in functions:
            if function.identifier == name:
                return function
        named = [function for function in functions if function.name == name]
        if named:
            sharing = [
                predicate for predicate in self.classes() if predicate.name == name
            ]
            candidates = [*named, *sharing]
            if len(candidates) > 1:
                raise AmbiguousPredicate(
                    name=name,
                    identifiers=sorted(
                        candidate.identifier for candidate in candidates
                    ),
                )
            return named[0]
        kind = self.kinds.resolve(name)
        if not PredicateClass.is_evaluable(kind):
            raise NotAPredicate(identifier=identifier_of(kind))
        return PredicateClass(kind=kind)

    def evaluate(
        self,
        name: str,
        arguments: Mapping[str, Argument],
        handles: HandleRegistry,
    ) -> Any:
        """
        :param name: The identifier of a predicate or value function, or its plain name.
        :param arguments: What it is passed, by argument name.
        :param handles: Resolves the individuals passed by handle.
        :return: What it says about the arguments.
        :raises NotAPredicate: When the name is neither.
        :raises UnknownArgument: When an argument is not one it takes.
        :raises MissingArgument: When an argument it requires is not passed.
        :raises PredicateFailed: When a function the world model declares raises.
        """
        return self.resolve(name).evaluate(arguments, handles)
