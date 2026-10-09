"""
Describing a world individual by walking its dataclass fields.

The walk knows nothing about the world model an individual belongs to: what a class is
comes from its docstring, what a field means from the field's docstring, and which
individuals hold another from the fields of the individuals in the world.
"""

from __future__ import annotations

from dataclasses import Field, dataclass, field, replace

from typing_extensions import Any, Optional, Sequence, Set

from llm_reasoner.world.classes import DocumentedClass
from llm_reasoner.world.description import (
    FieldDescription,
    IndividualDescription,
    Referrer,
)
from llm_reasoner.world.holders import HolderIndex
from llm_reasoner.world.naming import IndividualNaming, LocalNumbering
from llm_reasoner.world.sources import SymbolGraphSource, WorldSource
from llm_reasoner.world.value_forms import (
    COLLECTION_TYPES,
    DEFAULT_VALUE_FORMS,
    ValueForm,
)
from llm_reasoner.world.values import FieldValue, UndescribedValue


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


@dataclass
class IndividualWalk:
    """
    One walk from an individual through the individuals its fields hold.
    """

    limits: WalkLimits
    """
    How far the walk reaches.
    """

    naming: IndividualNaming
    """
    Names the individuals the walk describes or refers to.
    """

    forms: Sequence[ValueForm]
    """
    The forms values are read in, the first that reads a value reading it.
    """

    described: Set[int] = field(default_factory=set)
    """
    The identities of the individuals described so far.
    """

    def describe(self, individual: Any, depth: int) -> IndividualDescription:
        """
        :param individual: The individual to describe.
        :param depth: How many levels below the first individual it is.
        :return: Its description, describing what its fields hold in turn.
        """
        self.described.add(id(individual))
        documented = DocumentedClass.of(type(individual))
        described_fields = []
        for declared in documented.content_fields_of(individual):
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
            identifier=self.naming.identifier_of(individual),
            lineage=documented.lineage(self.limits.lineage_length),
            fields=described_fields,
        )

    def value(
        self, held: Any, depth: int, declared: Optional[Field] = None
    ) -> FieldValue:
        """
        :param held: What a field holds.
        :param depth: How many levels below the first individual the holder is.
        :param declared: The field holding it, when a field holds it directly.
        :return: The value in the first form that reads it.
        """
        form = next((form for form in self.forms if form.reads(held)), None)
        if form is None:
            return UndescribedValue(type_name=type(held).__name__)
        return form.read(held, self, depth, declared)

    def may_describe(self, individual: Any, depth: int) -> bool:
        """
        :return: Whether an individual is still to be described rather than referred
            to: not described already, and within the limits of the walk.
        """
        return (
            id(individual) not in self.described
            and depth < self.limits.depth
            and len(self.described) < self.limits.individuals
        )


@dataclass(frozen=True)
class IndividualDescriber:
    """
    Describes world individuals from their dataclass fields.
    """

    limits: WalkLimits = field(default_factory=WalkLimits)
    """
    How far a description reaches.
    """

    source: WorldSource = field(default_factory=SymbolGraphSource)
    """
    The world whose individuals may hold the described one.
    """

    forms: Sequence[ValueForm] = DEFAULT_VALUE_FORMS
    """
    The forms values are read in.
    """

    def describe(
        self, individual: Any, naming: Optional[IndividualNaming] = None
    ) -> IndividualDescription:
        """
        :param individual: An instance of a dataclass.
        :param naming: Names the individuals of the description; by default they are
            numbered within it.
        :return: Its description, with the individuals holding it.
        """
        walk = IndividualWalk(
            limits=self.limits,
            naming=naming if naming is not None else LocalNumbering(),
            forms=self.forms,
        )
        description = walk.describe(individual, depth=0)
        holdings = HolderIndex.of(self.source.individuals()).holdings_of(individual)
        return replace(
            description,
            referrers=[
                Referrer(
                    type_name=type(holding.holder).__name__,
                    field_name=holding.field_name,
                    summary=DocumentedClass.of(type(holding.holder)).summary,
                )
                for holding in holdings
            ],
        )

    def value_of(
        self, held: Any, naming: IndividualNaming, declared: Optional[Field] = None
    ) -> FieldValue:
        """
        :return: A single value read on its own, an individual referred to rather than
            described.
        """
        walk = IndividualWalk(
            limits=replace(self.limits, depth=0), naming=naming, forms=self.forms
        )
        return walk.value(held, depth=0, declared=declared)
