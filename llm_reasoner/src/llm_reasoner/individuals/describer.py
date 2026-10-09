"""
Describing a world individual by walking its dataclass fields.

The walk knows nothing about the world model an individual belongs to: what a class is
comes from its docstring, what a field means from the field's docstring, and which
individuals hold another from the fields of the live symbols in krrood's symbol graph.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from functools import cache, cached_property
from itertools import islice
from operator import attrgetter

from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)
from krrood.patterns.field_metadata import JSONMetadata
from krrood.symbol_graph.symbol_graph import SymbolGraph
from krrood.symbolic_math.symbolic_math import SymbolicMathType
from typing_extensions import Any, Dict, List, Optional, Tuple

from llm_reasoner.individuals.description import (
    ClassSummary,
    CollectionValue,
    FieldDescription,
    FieldValue,
    IndividualDescription,
    NestedIndividual,
    NumericValue,
    PlainValue,
    Referrer,
    RepeatedIndividual,
    UndescribedValue,
    WorldStateDependentValue,
)

PLAIN_TYPES = (bool, int, float, str, Enum, type(None))
"""
Types whose values are given as they are.
"""

COLLECTION_TYPES = (list, tuple, set, frozenset)
"""
Types whose values are given as their size and first items.
"""

PRIVATE_PREFIX = "_"
"""
The prefix of a field that is bookkeeping rather than content.
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


# %% classes


@dataclass(frozen=True)
class DocumentedClass:
    """
    What a class says about itself: its summary, the meaning of its fields, and which
    of its fields hold content.
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
        The first line of the docstring, empty when the class has none of its own. The
        signature a dataclass writes for an undocumented class does not count.
        """
        docstring = self.described.__doc__
        if not docstring or docstring.startswith(self.described.__name__ + "("):
            return ""
        return inspect.cleandoc(docstring).splitlines()[0]

    @cached_property
    def meanings(self) -> Dict[str, str]:
        """
        The docstring of every field, inherited ones included, by field name.
        """
        meanings: Dict[str, str] = {}
        for ancestor in reversed(self.described.__mro__):
            if is_dataclass(ancestor):
                meanings.update(_attribute_docstrings(ancestor))
        return meanings

    @cached_property
    def content_fields(self) -> Tuple[Field, ...]:
        """
        The fields that hold content: set by whoever makes an individual rather than
        filled in by the class itself, not private, and not excluded from serialisation,
        which is how a class marks a field as derived or as bookkeeping.
        """
        if not is_dataclass(self.described):
            return ()
        return tuple(
            declared
            for declared in fields(self.described)
            if declared.init
            and not declared.name.startswith(PRIVATE_PREFIX)
            and self.is_serialised(declared)
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
            name, so this is the one place a field is read by name.
        """
        return attrgetter(declared.name)(individual)

    def lineage(self, length: int) -> List[ClassSummary]:
        """
        :return: The class and the classes it specialises, most specific first, without
            the foundation every class shares.
        """
        foundation = {module.value for module in FoundationModule}
        return [
            ClassSummary(
                name=ancestor.__name__, summary=DocumentedClass.of(ancestor).summary
            )
            for ancestor in self.described.__mro__
            if ancestor.__module__ not in foundation
        ][:length]


# %% limits


@dataclass(frozen=True)
class WalkLimits:
    """
    How far a description reaches.
    """

    depth: int = 3
    """
    How many levels of individuals below the first are described.
    """

    items_per_collection: int = 3
    """
    How many items of a collection are described.
    """

    individuals: int = 40
    """
    How many individuals one description describes at most.
    """

    lineage_length: int = 4
    """
    How many classes of an individual's lineage are named.
    """


# %% walk


def is_individual(value: Any) -> bool:
    """
    :return: Whether a value is an instance of a dataclass, and so has fields to walk.
    """
    return is_dataclass(value) and not isinstance(value, type)


@dataclass
class IndividualWalk:
    """
    One walk from an individual through the individuals its fields hold.
    """

    limits: WalkLimits
    """
    How far the walk reaches.
    """

    identifiers: Dict[int, str] = field(default_factory=dict)
    """
    The identifier of every individual described so far, by object identity.
    """

    def describe(self, individual: Any, depth: int) -> IndividualDescription:
        """
        :param individual: The individual to describe.
        :param depth: How many levels below the first individual it is.
        :return: Its description, describing what its fields hold in turn.
        """
        identifier = f"{type(individual).__name__} {len(self.identifiers) + 1}"
        self.identifiers[id(individual)] = identifier
        documented = DocumentedClass.of(type(individual))
        described_fields = []
        for declared in documented.content_fields:
            held = documented.value_of(individual, declared)
            if isinstance(held, COLLECTION_TYPES) and not held:
                continue
            described_fields.append(
                FieldDescription(
                    name=declared.name,
                    meaning=documented.meanings.get(declared.name, ""),
                    value=self.value(held, depth, declared),
                )
            )
        return IndividualDescription(
            identifier=identifier,
            lineage=documented.lineage(self.limits.lineage_length),
            fields=described_fields,
        )

    def value(
        self, held: Any, depth: int, declared: Optional[Field] = None
    ) -> FieldValue:
        """
        :param held: What a field holds.
        :param depth: How many levels below the first individual the holder is.
        :param declared: The field holding it, when it is held directly by a field.
        :return: The value in a form a language model can read.
        """
        if isinstance(held, PLAIN_TYPES):
            return PlainValue(value=held, is_default=self.is_default(held, declared))
        if isinstance(held, SymbolicMathType):
            return self.symbolic_value(held)
        if id(held) in self.identifiers:
            return RepeatedIndividual(identifier=self.identifiers[id(held)])
        if is_individual(held):
            if (
                depth >= self.limits.depth
                or len(self.identifiers) >= self.limits.individuals
            ):
                return UndescribedValue(type_name=type(held).__name__)
            return NestedIndividual(description=self.describe(held, depth + 1))
        if isinstance(held, COLLECTION_TYPES):
            return CollectionValue(
                size=len(held),
                items=tuple(
                    self.value(item, depth)
                    for item in islice(held, self.limits.items_per_collection)
                ),
            )
        return UndescribedValue(type_name=type(held).__name__)

    @staticmethod
    def is_default(held: Any, declared: Optional[Field]) -> bool:
        """
        :return: Whether a plain value is the default its field declares. A default made
            by a factory is not compared, since making one could create individuals.
        """
        return (
            declared is not None
            and declared.default is not MISSING
            and held == declared.default
        )

    @staticmethod
    def symbolic_value(held: SymbolicMathType) -> FieldValue:
        """
        :return: The numbers of a constant symbolic value, or the state variables a
            varying one depends on.
        """
        if held.is_constant():
            return NumericValue(values=tuple(held.to_np().flatten().tolist()))
        return WorldStateDependentValue(
            variables=tuple(variable.name for variable in held.free_variables())
        )


# %% referrers


@dataclass(frozen=True)
class ReferrerIndex:
    """
    Which live symbols hold which individuals in their content fields.
    """

    referrers: Dict[int, List[Referrer]]
    """
    The individuals holding each individual, by its object identity.
    """

    @classmethod
    def of_live_symbols(cls) -> ReferrerIndex:
        """
        :return: The index over every live symbol in krrood's symbol graph.
        """
        referrers: Dict[int, List[Referrer]] = defaultdict(list)
        for wrapped in SymbolGraph().wrapped_instances:
            holder = wrapped.instance
            if not is_individual(holder):
                continue
            documented = DocumentedClass.of(type(holder))
            for declared in documented.content_fields:
                held = documented.value_of(holder, declared)
                targets = held if isinstance(held, COLLECTION_TYPES) else (held,)
                for target in targets:
                    if is_individual(target):
                        referrers[id(target)].append(
                            Referrer(
                                type_name=type(holder).__name__,
                                field_name=declared.name,
                                summary=documented.summary,
                            )
                        )
        return cls(referrers=dict(referrers))

    def referrers_of(self, individual: Any) -> List[Referrer]:
        """
        :return: The individuals holding the given one.
        """
        return self.referrers.get(id(individual), [])


# %% describer


@dataclass(frozen=True)
class IndividualDescriber:
    """
    Describes world individuals from their dataclass fields.
    """

    limits: WalkLimits = field(default_factory=WalkLimits)
    """
    How far a description reaches.
    """

    def describe(self, individual: Any) -> IndividualDescription:
        """
        :param individual: An instance of a dataclass.
        :return: Its description, with the individuals holding it.
        """
        description = IndividualWalk(limits=self.limits).describe(individual, depth=0)
        return replace(
            description,
            referrers=ReferrerIndex.of_live_symbols().referrers_of(individual),
        )
