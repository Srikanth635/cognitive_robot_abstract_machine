"""
Questions about the kinds a world has: which there are, what each is, and which kind a
name means.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from typing_extensions import Any, List, Mapping, Optional

from llm_reasoner.world.classes import DocumentedClass, FieldDefinition, identifier_of
from llm_reasoner.world.description import ClassSummary
from llm_reasoner.world.exceptions import AmbiguousKind, UnknownKind
from llm_reasoner.world.sources import WorldSource

QUALIFIER_SEPARATOR = "."
"""
What separates the module of a kind from its name, and an enclosing class from a nested
one.
"""

LINEAGE_LENGTH = 8
"""
How many ancestors the description of a kind names.
"""


@dataclass(frozen=True)
class KindSummary:
    """
    A kind as a listing names it.
    """

    identifier: str
    """
    The module and qualified name of the kind.
    """

    name: str
    """
    The name of the kind.
    """

    summary: str
    """
    The first line of its docstring.
    """

    individual_count: int
    """
    How many live individuals are of the kind or of a kind specialising it.
    """


@dataclass(frozen=True)
class KindDescription:
    """
    What a kind is and what its individuals hold.
    """

    identifier: str
    """
    The module and qualified name of the kind.
    """

    name: str
    """
    The name of the kind.
    """

    summary: str
    """
    The first line of its docstring.
    """

    ancestors: List[ClassSummary]
    """
    The kinds it specialises, most specific first.
    """

    specialisations: List[str]
    """
    The identifiers of the loaded kinds specialising it directly.
    """

    fields: List[FieldDefinition]
    """
    The content fields it declares.
    """


@dataclass(frozen=True)
class KindCatalogue:
    """
    The kinds of a world.
    """

    source: WorldSource
    """
    The world whose kinds are listed.
    """

    def kinds(
        self, within: Optional[type] = None, present_only: bool = False
    ) -> List[KindSummary]:
        """
        :param within: Only this kind and the kinds specialising it; every kind if
            ``None``.
        :param present_only: Only kinds with live individuals.
        :return: The kinds, ordered by identifier.
        """
        summaries = []
        for kind in self.source.kinds():
            if within is not None and not issubclass(kind, within):
                continue
            count = len(self.source.individuals_of(kind))
            if present_only and count == 0:
                continue
            summaries.append(
                KindSummary(
                    identifier=identifier_of(kind),
                    name=kind.__name__,
                    summary=DocumentedClass.of(kind).summary,
                    individual_count=count,
                )
            )
        return sorted(summaries, key=lambda summary: summary.identifier)

    def describe(self, kind: type) -> KindDescription:
        """
        :return: What the kind is, what it specialises and is specialised by, and the
            fields it declares.
        """
        documented = DocumentedClass.of(kind)
        lineage = documented.lineage(LINEAGE_LENGTH)
        return KindDescription(
            identifier=identifier_of(kind),
            name=kind.__name__,
            summary=documented.summary,
            ancestors=lineage[1:],
            specialisations=sorted(
                identifier_of(specialisation)
                for specialisation in self.source.kinds()
                if kind in specialisation.__bases__
            ),
            fields=documented.field_definitions(),
        )

    def resolve(self, name: str) -> type:
        """
        :param name: The identifier of a loaded class, or the plain name of a kind that
            no other kind shares.
        :return: The class it names.
        :raises AmbiguousKind: When several kinds share the plain name.
        :raises UnknownKind: When nothing loaded has the name.
        """
        qualified = self.loaded_class(name)
        if qualified is not None:
            return qualified
        named = [kind for kind in self.source.kinds() if kind.__name__ == name]
        if len(named) > 1:
            raise AmbiguousKind(
                name=name, identifiers=sorted(identifier_of(kind) for kind in named)
            )
        if not named:
            raise UnknownKind(name=name)
        return named[0]

    @staticmethod
    def loaded_class(identifier: str) -> Optional[type]:
        """
        :return: The class an identifier names in a module already loaded, or ``None``
            when no loaded module declares it.
        """
        parts = identifier.split(QUALIFIER_SEPARATOR)
        for split in range(len(parts) - 1, 0, -1):
            module_name = QUALIFIER_SEPARATOR.join(parts[:split])
            if module_name in sys.modules:
                return KindCatalogue.nested_class(
                    vars(sys.modules[module_name]), parts[split:]
                )
        return None

    @staticmethod
    def nested_class(namespace: Mapping[str, Any], names: List[str]) -> Optional[type]:
        """
        :return: The class the names reach in a namespace, each a class declared in the
            one before; ``None`` when one of them is not.
        """
        found = namespace.get(names[0])
        if not isinstance(found, type):
            return None
        if len(names) == 1:
            return found
        return KindCatalogue.nested_class(vars(found), names[1:])
