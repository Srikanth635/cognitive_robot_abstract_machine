"""
What the world injector raises when a question about the world cannot be answered as
asked.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Optional

from llm_reasoner.exceptions import LLMReasonerError


class WorldQuestionError(LLMReasonerError):
    """
    A question about the world that names something the world does not have.
    """


@dataclass
class UnknownKind(WorldQuestionError):
    """
    No loaded kind has the given name.
    """

    name: str
    """
    The name asked for.
    """

    def __post_init__(self):
        super().__init__(f"No loaded kind is named {self.name!r}")


@dataclass
class AmbiguousKind(WorldQuestionError):
    """
    Several loaded kinds share the given plain name, so only an identifier tells them
    apart.
    """

    name: str
    """
    The plain name asked for.
    """

    identifiers: List[str]
    """
    The identifiers of the kinds sharing it.
    """

    def __post_init__(self):
        super().__init__(
            f"{self.name!r} names several kinds; ask for one of {self.identifiers}"
        )


@dataclass
class NotAPredicate(WorldQuestionError):
    """
    The kind asked to be evaluated is not a predicate or a value function.
    """

    identifier: str
    """
    The identifier of the kind.
    """

    def __post_init__(self):
        super().__init__(f"{self.identifier} is not a predicate or a value function")


@dataclass
class UnknownHandle(WorldQuestionError):
    """
    The handle was never given out.
    """

    handle: str
    """
    The handle asked for.
    """

    kind_name: Optional[str] = None
    """
    The kind the handle is written for, when it is written as handles are.
    """

    def __post_init__(self):
        search = (
            f"; a search for the kind {self.kind_name!r} gives out the handles of its "
            "individuals"
            if self.kind_name is not None
            else ""
        )
        super().__init__(
            f"No individual was ever given the handle {self.handle!r}; handles are "
            f"given out only by answers that name individuals, such as a search{search}"
        )


@dataclass
class ExpiredHandle(WorldQuestionError):
    """
    The individual the handle named no longer exists.
    """

    handle: str
    """
    The handle asked for.
    """

    def __post_init__(self):
        super().__init__(f"The individual {self.handle!r} no longer exists")


@dataclass
class UnknownField(WorldQuestionError):
    """
    A field path passes through a field the individual reached does not have.
    """

    type_name: str
    """
    The kind of the individual reached.
    """

    field_name: str
    """
    The field it lacks.
    """

    def __post_init__(self):
        super().__init__(f"A {self.type_name} has no field {self.field_name!r}")


@dataclass
class InapplicableComparison(WorldQuestionError):
    """
    A condition's comparison applies to none of the values its path reaches, such as a
    word looked for in a value that is neither text nor a collection.
    """

    path: str
    """
    The path of the condition.
    """

    comparison: str
    """
    The comparison asked for.
    """

    reached_type_name: str
    """
    The kind of value the path reaches.
    """

    given_type_name: str
    """
    The kind of value compared with.
    """

    reached_fields: List[str]
    """
    The fields the value reached holds, which the path may continue to.
    """

    def __post_init__(self):
        continuation = (
            f"; a {self.reached_type_name} holds the fields {self.reached_fields}, "
            "which the path may continue to"
            if self.reached_fields
            else ""
        )
        super().__init__(
            f"{self.path!r} reaches a {self.reached_type_name}, which cannot be "
            f"compared as '{self.comparison}' with a {self.given_type_name}"
            f"{continuation}"
        )


@dataclass
class UnknownArgument(WorldQuestionError):
    """
    An argument given to a predicate is not one it takes.
    """

    predicate: str
    """
    The identifier of the predicate.
    """

    argument: str
    """
    The argument given.
    """

    def __post_init__(self):
        super().__init__(f"{self.predicate} takes no argument {self.argument!r}")


@dataclass
class MissingArgument(WorldQuestionError):
    """
    A predicate is evaluated without an argument it requires.
    """

    predicate: str
    """
    The identifier of the predicate.
    """

    argument: str
    """
    The argument missing.
    """

    def __post_init__(self):
        super().__init__(f"{self.predicate} requires the argument {self.argument!r}")
