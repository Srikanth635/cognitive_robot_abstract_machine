"""
Following dotted field paths through the declared classes.

A constraint reaches into a hypothesis by naming fields, such as
``"from_phase.terminated_by"``. Evaluation reads a missing field as an absent value, so a
misspelled path never reports a violation. Resolving the same path against the declared
field types finds the misspelling before any hypothesis exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Iterable

from llm_reasoner.knowledge.ontological.decorators import OntologyRegistry
from llm_reasoner.knowledge.ontological.validation import _range_types, _resolved_hints

# %% path references


@dataclass(frozen=True)
class PathReference:
    """
    A path that a constraint follows, and where the path starts.
    """

    path: str
    """
    Dotted field names to follow.
    """

    relative_to: str = ""
    """
    Path from the constrained object to the members that :attr:`path` starts from.

    Empty when the path starts at the constrained object.
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

    unknown_step: str | None = None
    """
    The first step that no candidate type declares as a field, or ``None`` when every step
    was found or the path cannot be verified.
    """


# %% resolution


@dataclass
class FieldPathResolver:
    """
    Follows field paths through the classes declared in a registry.

    A step is found when the current type, or any declared subtype of it, has a dataclass
    field of that name. Subtypes count because a field's declared range is often a base
    class whose specialisations introduce further fields.
    """

    registry: OntologyRegistry
    """
    The declarations that supply subtypes.
    """

    _hints_by_class: dict[type, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Resolved field annotations, kept because resolving them evaluates every annotation.
    """

    def resolve(self, start: Iterable[type], path: str) -> PathResolution:
        """
        Follow a path from the given types.

        :param start: The types the first step is taken from.
        :param path: Dotted field names.
        """
        current = tuple(start)
        for step in path.split("."):
            declaring = [
                candidate
                for base in current
                for candidate in self._with_subtypes(base)
                if self._declares(candidate, step)
            ]
            if not declaring:
                return PathResolution(unknown_step=step)
            current = self._ranges_of(declaring, step)
            if not current:
                return PathResolution()
        return PathResolution(types=current)

    def _with_subtypes(self, base: type) -> tuple[type, ...]:
        """
        Return a type and every declared subtype of it.
        """
        declared = (
            metadata.cls
            for metadata in self.registry.all()
            if isinstance(metadata.cls, type)
            and metadata.cls is not base
            and issubclass(metadata.cls, base)
        )
        return (base, *declared)

    @staticmethod
    def _declares(candidate: type, name: str) -> bool:
        """
        Return whether a class declares a dataclass field of this name.
        """
        return is_dataclass(candidate) and any(
            declared.name == name for declared in fields(candidate)
        )

    def _ranges_of(self, declaring: Iterable[type], name: str) -> tuple[type, ...]:
        """
        Return the types a field can hold across the declaring classes.

        Empty when any declaring class leaves the field's annotation unresolved, since
        the path can then not be followed further.
        """
        ranges: list[type] = []
        for candidate in declaring:
            hint = self._hints_of(candidate).get(name)
            if hint is None:
                return ()
            ranges.extend(held for held in _range_types(hint) if isinstance(held, type))
        return tuple(dict.fromkeys(ranges))

    def _hints_of(self, candidate: type) -> dict[str, Any]:
        """
        Return a class's resolved field annotations, resolving each class once.
        """
        if candidate not in self._hints_by_class:
            self._hints_by_class[candidate] = _resolved_hints(candidate)
        return self._hints_by_class[candidate]


__all__ = ["FieldPathResolver", "PathReference", "PathResolution"]
