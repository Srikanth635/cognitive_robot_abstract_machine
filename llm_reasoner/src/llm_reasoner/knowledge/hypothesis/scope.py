"""Keeping stereotypical knowledge out of the world, and the world out of it.

THE FAILURE THIS PREVENTS IS SILENT
-----------------------------------
A hypothesis holding an episodic entity does not raise. It changes what queries
mean: closed-world quantification ranges over whatever happens to be present, so
a stray world individual does not merely add a row -- it can flip the truth value
of a negation or a universal. The wrong answer looks exactly like the right one.

The boundary is therefore enforced structurally, at the point of admission,
rather than trusted to whoever writes the next fragment. ``ConceptLevel`` on
every declared class is what makes the check possible: OCCURRENCE-level classes
are the resolver's business and have no place in a stereotype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Optional, Type, TypeVar

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    KnowledgeEntity,
    OntologyRegistry,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel

T = TypeVar("T")


class ScopeViolation(LLMReasonerError):
    """Raised when something episodic is put into a stereotype.

    An exception rather than a logged warning: admitting the fragment and
    reporting it would leave the scope contaminated, and every query over it
    silently wrong from that point on.
    """


@dataclass
class KnowledgeScope:
    """A container of description-level fragments, closed to everything else.

    One scope holds one hypothesis's fragments. It is the unit that gets cached,
    versioned, queried and discarded, and the boundary the resolver's world
    model never crosses.
    """

    label: str = ""
    registry: OntologyRegistry = REGISTRY
    _fragments: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- admission ---------------------------------------------------------

    def admit(self, fragment: Any) -> Any:
        """Add one fragment, refusing anything that does not belong.

        :param fragment: A description-level instance of a declared class.
        :raises ScopeViolation: if the fragment is undeclared, occurrence-level,
            or carries no stable identity.
        """
        self._reject_if_foreign(fragment)
        self._fragments[fragment.local_id] = fragment
        return fragment

    def admit_all(self, fragments: Iterable[Any]) -> None:
        """Add many, refusing the whole batch if any one does not belong.

        All-or-nothing because a partially admitted hypothesis is worse than a
        rejected one: it looks complete and is not.
        """
        checked = list(fragments)
        for fragment in checked:
            self._reject_if_foreign(fragment)
        for fragment in checked:
            self._fragments[fragment.local_id] = fragment

    def _reject_if_foreign(self, fragment: Any) -> None:
        if isinstance(fragment, Enum):
            raise ScopeViolation(
                f"{type(fragment).__name__}.{fragment.name} is a value from a "
                "value partition, not a fragment. Enum members are held by "
                "fragments; they have no identity of their own and nothing "
                "cites them."
            )
        meta = self.registry.meta_for(type(fragment))
        if meta is None:
            raise ScopeViolation(
                f"{type(fragment).__name__} is not a declared class; only "
                "described microworld content may enter a knowledge scope"
            )
        if meta.level is ConceptLevel.OCCURRENCE:
            raise ScopeViolation(
                f"{type(fragment).__name__} is occurrence-level and belongs to "
                "the world model, not to a stereotype. If this is a role, a "
                "hypothesis holds an unbound RoleSlot; the bound role is "
                "created by the resolver at binding time."
            )
        if not isinstance(fragment, KnowledgeEntity):
            raise ScopeViolation(
                f"{type(fragment).__name__} has no stable identity, so it "
                "cannot be cached, cited or round-tripped"
            )

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._fragments)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._fragments.values())

    def __contains__(self, item: Any) -> bool:
        key = item if isinstance(item, str) else getattr(item, "local_id", None)
        return key in self._fragments

    def by_id(self, local_id: str) -> Optional[Any]:
        """:return: The fragment with that identity, or ``None``."""
        return self._fragments.get(local_id)

    def of_type(self, cls: Type[T]) -> list[T]:
        """:return: Every fragment of a type, subclasses included."""
        return [f for f in self._fragments.values() if isinstance(f, cls)]

    @property
    def fragments(self) -> list[Any]:
        """:return: Every fragment, for handing to a constraint suite."""
        return list(self._fragments.values())


def collect_reachable(root: Any, registry: OntologyRegistry = REGISTRY) -> list[Any]:
    """Walk every declared fragment reachable from a root.

    Generation produces a graph rather than a flat list -- a structure pointing at
    phases pointing at subgoals -- and both the constraint suite and the scope
    need the whole graph, not just the root. Traversal follows declared fields
    only, so undescribed attributes cannot smuggle anything in.

    :param root: The fragment to start from, typically the phase structure.
    :return: Every reachable declared fragment, root first, without repetition.
    """
    from llm_reasoner.knowledge.ontological.decorators import property_meta

    seen: dict[int, Any] = {}
    frontier = [root]
    while frontier:
        current = frontier.pop()
        if isinstance(current, Enum):
            # a value partition member: held BY a fragment, never itself one
            continue
        if id(current) in seen or registry.meta_for(type(current)) is None:
            continue
        seen[id(current)] = current
        for field_name in property_meta(type(current)):
            value = getattr(current, field_name, None)
            if value is None:
                continue
            values = (
                value
                if isinstance(value, (list, tuple, set, frozenset))
                else [value]
            )
            frontier.extend(v for v in values if v is not None)
    return list(seen.values())


__all__ = ["KnowledgeScope", "ScopeViolation", "collect_reachable"]