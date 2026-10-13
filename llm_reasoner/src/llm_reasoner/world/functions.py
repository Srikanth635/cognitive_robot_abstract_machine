"""
The functions a world model declares with krrood's ``symbolic_function``: which there
are, what each takes and says, and how one is called.

krrood keeps no record of these functions, so they are recognised by the code every
function ``symbolic_function`` returns runs.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from functools import cache, cached_property
from types import CodeType

from krrood.entity_query_language.predicate import symbolic_function
from typing_extensions import Any, Callable, Dict, List, Mapping

from llm_reasoner.world.classes import (
    PLAIN_TYPES,
    FieldDefinition,
    identifier_of,
    written_type,
)
from llm_reasoner.world.exceptions import PredicateFailed

VARIADIC_KINDS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)
"""
The kinds of parameter that stand for any number of arguments rather than for one.
"""

NAMEABLE_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)
"""
The kinds of parameter a call can pass an argument to by name.
"""

PARAMETER_MEANING = re.compile(
    r"^:param (\w+):[ \t]*(.*(?:\n[ \t]+\S.*)*)", re.MULTILINE
)
"""
A parameter and what its docstring says it is, which may continue on indented lines.
"""


@cache
def wrapper_code() -> CodeType:
    """
    :return: The code every function returned by ``symbolic_function`` runs, which tells
        such functions from all others.
    """
    return symbolic_function(lambda: None).__code__


@dataclass(frozen=True)
class WorldFunction:
    """
    A function the world model declares with ``symbolic_function``: it says something
    about individuals of the world.
    """

    function: Callable[..., Any]
    """
    The function as its author wrote it, without the wrapper ``symbolic_function`` puts
    around it.
    """

    @staticmethod
    def is_declared(member: Any) -> bool:
        """
        :return: Whether a member of a module is a function returned by
            ``symbolic_function``.
        """
        return inspect.isfunction(member) and member.__code__ is wrapper_code()

    @classmethod
    def of(cls, declared: Callable[..., Any]) -> WorldFunction:
        """
        :param declared: A function returned by ``symbolic_function``.
        :return: The function it wraps.
        """
        return cls(function=declared.__wrapped__)

    @cached_property
    def identifier(self) -> str:
        """
        The module and qualified name of the function, which tell it from others of its
        name.
        """
        return identifier_of(self.function)

    @cached_property
    def name(self) -> str:
        """
        The name of the function.
        """
        return self.function.__name__

    @cached_property
    def documentation(self) -> str:
        """
        The docstring of the function, empty when it has none.
        """
        return inspect.getdoc(self.function) or ""

    @cached_property
    def summary(self) -> str:
        """
        The first line of its docstring, empty when it has none.
        """
        lines = self.documentation.splitlines()
        return lines[0] if lines else ""

    @cached_property
    def meanings(self) -> Dict[str, str]:
        """
        What the docstring says each parameter is, by parameter name.
        """
        return {
            name: " ".join(meaning.split())
            for name, meaning in PARAMETER_MEANING.findall(self.documentation)
        }

    @cached_property
    def parameters(self) -> List[inspect.Parameter]:
        """
        The parameters of the function, in order, leaving out those that stand for any
        number of arguments.
        """
        return [
            parameter
            for parameter in inspect.signature(self.function).parameters.values()
            if parameter.kind not in VARIADIC_KINDS
        ]

    @staticmethod
    def is_required(parameter: inspect.Parameter) -> bool:
        """
        :return: Whether a call must pass an argument for the parameter.
        """
        return parameter.default is inspect.Parameter.empty

    @staticmethod
    def type_name_of(parameter: inspect.Parameter) -> str:
        """
        :return: The type a parameter declares, as written; empty when it declares none.
        """
        if parameter.annotation is inspect.Parameter.empty:
            return ""
        return written_type(parameter.annotation)

    def definition_of(self, parameter: inspect.Parameter) -> FieldDefinition:
        """
        :return: A parameter as a listing states it: its type, what the docstring says
            it is, and its default when that is a plain value.
        """
        has_plain_default = not self.is_required(parameter) and isinstance(
            parameter.default, PLAIN_TYPES
        )
        return FieldDefinition(
            name=parameter.name,
            type_name=self.type_name_of(parameter),
            meaning=self.meanings.get(parameter.name, ""),
            default=repr(parameter.default) if has_plain_default else None,
        )

    def call(self, values: Mapping[str, Any]) -> Any:
        """
        :param values: What each parameter is passed, by name.
        :return: What the function says about them.
        :raises PredicateFailed: When the function raises.

        .. note:: This is the one place the code of a world model runs, and it can fail
            in ways that say nothing about the question asked, such as a simulation it
            needs being absent. The failure is reported to whoever asked.
        """
        try:
            return self.function(**values)
        except Exception as error:
            raise PredicateFailed(
                predicate=self.identifier,
                error_type=type(error).__name__,
                reason=str(error),
            ) from error
