"""
Which individuals hold which others in their content fields.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from typing_extensions import Any, Dict, Iterable, List

from llm_reasoner.world.classes import DocumentedClass, is_individual
from llm_reasoner.world.value_forms import COLLECTION_TYPES


@dataclass(frozen=True)
class Holding:
    """
    An individual holding another in one of its fields.
    """

    holder: Any
    """
    The individual holding the other.
    """

    field_name: str
    """
    The field holding it.
    """


@dataclass(frozen=True)
class Holder:
    """
    An individual holding another, named by its handle.
    """

    handle: str
    """
    The handle of the holding individual.
    """

    type_name: str
    """
    The name of its kind.
    """

    field_name: str
    """
    The field holding the other individual.
    """

    summary: str
    """
    The first line of the docstring of its kind.
    """


@dataclass(frozen=True)
class HolderIndex:
    """
    The holdings among a set of individuals, by the identity of the individual held.
    """

    holdings: Dict[int, List[Holding]]
    """
    The holdings of each held individual, by its object identity.
    """

    @classmethod
    def of(cls, individuals: Iterable[Any]) -> HolderIndex:
        """
        :return: The index over the content fields of the given individuals.
        """
        holdings: Dict[int, List[Holding]] = defaultdict(list)
        for holder in individuals:
            if not is_individual(holder):
                continue
            documented = DocumentedClass.of(type(holder))
            for declared in documented.content_fields_of(holder):
                held = documented.value_of(holder, declared)
                targets = held if isinstance(held, COLLECTION_TYPES) else (held,)
                for target in targets:
                    if is_individual(target):
                        holdings[id(target)].append(
                            Holding(holder=holder, field_name=declared.name)
                        )
        return cls(holdings=dict(holdings))

    def holdings_of(self, individual: Any) -> List[Holding]:
        """
        :return: The individuals holding the given one, with the field each holds it in.
        """
        return self.holdings.get(id(individual), [])
