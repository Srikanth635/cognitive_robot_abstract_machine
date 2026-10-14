"""
Following dotted field paths through the declared concepts.

An acceptance rule reaches into a hypothesis by naming fields, such as
``"from_phase.terminated_by"``. Resolving the same path against the declared field types
finds a misspelling before any hypothesis exists, so a rule can never silently read a field
that is not there.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Iterable, Optional

from llm_reasoner.knowledge_updated.ontological.ontology import Ontology
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver

# %% path references


@dataclass(frozen=True)
class PathReference:
    """
    A path that a rule follows, and where the path starts.
    """

    path: str
    """
    Dotted field names to follow.
    """

    relative_to: str = ""
    """
    Path from the subject of the rule to the members that :attr:`path` starts from.

    Empty when the path starts at the subject itself.
    """


@dataclass(frozen=True)
class PathResolution:
    """
    Where following a path ended.
    """

    types: tuple[type, ...] = ()
    """
    The types the path can end on. Empty when the path cannot be verified.
    """

    unknown_step: Optional[str] = None
    """
    The first step that no candidate type declares as a field, or ``None`` when every
    step was found or the path cannot be verified.
    """


# %% resolution


@dataclass
class FieldPathResolver:
    """
    Follows field paths through the concepts an ontology declares.

    A step is found when the current type, or any declared subtype of it, has a dataclass
    field of that name. Subtypes count because a field's declared range is often a base
    concept whose specialisations introduce further fields.
    """

    ontology: Ontology
    """
    The declarations that supply subtypes.
    """

    ranges: RangeResolver = field(default_factory=RangeResolver)
    """
    Reads what the type hint of each field says.
    """

    def resolve(self, start: Iterable[type], path: str) -> PathResolution:
        """
        Follow a path from the given types.

        :param start: The types the first step is taken from.
        :param path: Dotted field names.
        """
        current = tuple(start)
        for step in path.split("."):
            declaring = self._declaring_classes(current, step)
            if not declaring:
                return PathResolution(unknown_step=step)
            reached: list[type] = []
            for declarer in declaring:
                property_range = self.ranges.resolve(declarer, step)
                if not property_range.resolved:
                    return PathResolution()
                reached.extend(property_range.element_types)
            current = tuple(dict.fromkeys(reached))
            if not current:
                return PathResolution()
        return PathResolution(types=current)

    def _declaring_classes(self, current: tuple[type, ...], step: str) -> list[type]:
        """
        Return the classes that write the field in their own body, for every current type
        and its declared subtypes.
        """
        declaring: dict[type, None] = {}
        for base in current:
            for candidate in self._with_subtypes(base):
                declarer = next(
                    (
                        klass
                        for klass in candidate.__mro__
                        if self._declares(klass, step)
                    ),
                    None,
                )
                if declarer is not None:
                    declaring[declarer] = None
        return list(declaring)

    def _with_subtypes(self, base: type) -> tuple[type, ...]:
        """
        Return a type and every declared concept that specialises it.
        """
        return (
            base,
            *(
                meta.cls
                for meta in self.ontology.concepts()
                if meta.cls is not base and issubclass(meta.cls, base)
            ),
        )

    @staticmethod
    def _declares(klass: type, step: str) -> bool:
        """
        Return whether a dataclass writes a field of this name in its own body.
        """
        return (
            is_dataclass(klass)
            and step in inspect.get_annotations(klass)
            and any(declared.name == step for declared in fields(klass))
        )
