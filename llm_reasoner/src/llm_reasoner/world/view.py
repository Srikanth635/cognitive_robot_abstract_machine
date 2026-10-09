"""
The world injector's questions about the live world, answered as typed results and with
individuals named by handles that stay the same across questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Mapping, Optional, Sequence

from llm_reasoner.world.classes import DocumentedClass
from llm_reasoner.world.describer import IndividualDescriber, WalkLimits
from llm_reasoner.world.description import IndividualDescription
from llm_reasoner.world.finding import Condition, FoundIndividuals, IndividualFinder
from llm_reasoner.world.holders import Holder, HolderIndex
from llm_reasoner.world.kinds import KindCatalogue, KindDescription, KindSummary
from llm_reasoner.world.naming import HandleRegistry
from llm_reasoner.world.paths import FieldPath
from llm_reasoner.world.predicates import (
    Argument,
    PredicateCatalogue,
    PredicateDefinition,
)
from llm_reasoner.world.sources import SymbolGraphSource, WorldSource
from llm_reasoner.world.values import FieldValue

FIND_LIMIT = 20
"""
How many individuals a search names unless asked for another number.
"""


@dataclass
class WorldView:
    """
    Questions about the live world: which kinds and individuals it has, what they are,
    what holds them, what their fields reach, and what the world model's predicates say
    about them.
    """

    source: WorldSource = field(default_factory=SymbolGraphSource)
    """
    The world asked about.
    """

    handles: HandleRegistry = field(default_factory=HandleRegistry)
    """
    Names individuals by handles that stay the same across questions.
    """

    limits: WalkLimits = field(default_factory=WalkLimits)
    """
    How far a description reaches unless asked otherwise.
    """

    catalogue: KindCatalogue = field(init=False)
    """
    Answers questions about kinds.
    """

    finder: IndividualFinder = field(init=False)
    """
    Finds individuals by conditions.
    """

    predicate_catalogue: PredicateCatalogue = field(init=False)
    """
    Lists and evaluates the world model's predicates.
    """

    def __post_init__(self):
        self.catalogue = KindCatalogue(source=self.source)
        self.finder = IndividualFinder(source=self.source, handles=self.handles)
        self.predicate_catalogue = PredicateCatalogue(source=self.source)

    # %% kinds

    def kinds(
        self, within: Optional[str] = None, present_only: bool = False
    ) -> List[KindSummary]:
        """
        :param within: The name of a kind, to list only it and its specialisations.
        :param present_only: Only kinds with live individuals.
        :return: The kinds the world has.
        """
        return self.catalogue.kinds(
            within=None if within is None else self.catalogue.resolve(within),
            present_only=present_only,
        )

    def describe_kind(self, kind: str) -> KindDescription:
        """
        :param kind: The name of a kind.
        :return: What the kind is and what its individuals hold.
        """
        return self.catalogue.describe(self.catalogue.resolve(kind))

    # %% individuals

    def find(
        self, kind: str, conditions: Sequence[Condition] = (), limit: int = FIND_LIMIT
    ) -> FoundIndividuals:
        """
        :param kind: The name of a kind.
        :param conditions: Conditions on fields the individuals must meet.
        :param limit: How many of the individuals found to name.
        :return: The individuals found, with the search stated in English.
        """
        return self.finder.find(self.catalogue.resolve(kind), conditions, limit)

    def describe(
        self, handle: str, limits: Optional[WalkLimits] = None
    ) -> IndividualDescription:
        """
        :param handle: The handle of an individual.
        :param limits: How far the description reaches; the view's limits if ``None``.
        :return: What the individual is and holds, individuals named by their handles.
        """
        describer = IndividualDescriber(
            limits=limits if limits is not None else self.limits, source=self.source
        )
        return describer.describe(
            self.handles.individual_of(handle), naming=self.handles
        )

    def holders(self, handle: str) -> List[Holder]:
        """
        :param handle: The handle of an individual.
        :return: The individuals holding it, which often say what it is.
        """
        individual = self.handles.individual_of(handle)
        return [
            Holder(
                handle=self.handles.handle_of(holding.holder),
                type_name=type(holding.holder).__name__,
                field_name=holding.field_name,
                summary=DocumentedClass.of(type(holding.holder)).summary,
            )
            for holding in HolderIndex.of(self.source.individuals()).holdings_of(
                individual
            )
        ]

    def follow(self, handle: str, path: str) -> FieldValue:
        """
        :param handle: The handle of an individual.
        :param path: Field names separated by dots, such as ``inertial.mass``.
        :return: What the path reaches; an individual it reaches is named by its handle.
        """
        reached, holding_field = FieldPath(path=path).reach(
            self.handles.individual_of(handle)
        )
        describer = IndividualDescriber(limits=self.limits, source=self.source)
        return describer.value_of(reached, naming=self.handles, declared=holding_field)

    # %% predicates

    def predicates(self, about: Optional[str] = None) -> List[PredicateDefinition]:
        """
        :param about: The name of a kind, to list only predicates taking its individuals.
        :return: The predicates and value functions the world model declares.
        """
        return self.predicate_catalogue.predicates(
            about=None if about is None else self.catalogue.resolve(about)
        )

    def evaluate(self, predicate: str, arguments: Mapping[str, Argument]) -> FieldValue:
        """
        :param predicate: The name of a predicate or value function.
        :param arguments: What it is passed, by argument name.
        :return: What it says about the arguments.
        """
        result = self.predicate_catalogue.evaluate(
            self.catalogue.resolve(predicate), arguments, self.handles
        )
        describer = IndividualDescriber(limits=self.limits, source=self.source)
        return describer.value_of(result, naming=self.handles)
