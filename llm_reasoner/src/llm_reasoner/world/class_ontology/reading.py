"""
Reading the classes a selection names into descriptions: each class with its
documentation, the described classes it specialises and its fields; each enumeration
with its members.
"""

from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from enum import Enum

from typing_extensions import Dict, FrozenSet, List, Optional, Set, Tuple

from llm_reasoner.world.class_ontology.field_types import FieldType, FieldTypeReader
from llm_reasoner.world.class_ontology.selection import ClassSelection, SkippedModule
from llm_reasoner.world.classes import PLAIN_TYPES, DocumentedClass, identifier_of

# %% descriptions


@dataclass(frozen=True)
class FieldDescription:
    """
    A field of a class.
    """

    name: str
    """
    The field's name.
    """

    documentation: str
    """
    Its whole docstring; empty when it has none.
    """

    field_type: FieldType
    """
    What it holds.
    """

    default: Optional[str]
    """
    Its default written as Python, when it is a plain value; none otherwise.
    """


@dataclass(frozen=True)
class ClassDescription:
    """
    A class of the code base.
    """

    identifier: str
    """
    Its module and name.
    """

    name: str
    """
    Its name.
    """

    documentation: str
    """
    Its whole docstring; empty when it has none.
    """

    parents: Tuple[str, ...]
    """
    The nearest described classes it specialises, by module and name.
    """

    fields: Tuple[FieldDescription, ...]
    """
    The fields it adds to those of the described classes it specialises; none for a
    class described only because a field refers to it.
    """


@dataclass(frozen=True)
class EnumerationDescription:
    """
    An enumeration of the code base.
    """

    identifier: str
    """
    Its module and name.
    """

    name: str
    """
    Its name.
    """

    documentation: str
    """
    Its whole docstring; empty when it has none.
    """

    members: Tuple[str, ...]
    """
    The names of its members, in the order they are declared.
    """


@dataclass(frozen=True)
class ClassOntology:
    """
    The classes and enumerations of a code base a selection names.
    """

    classes: Tuple[ClassDescription, ...]
    """
    The classes, by module and name.
    """

    enumerations: Tuple[EnumerationDescription, ...]
    """
    The enumerations, by module and name.
    """

    skipped_modules: Tuple[SkippedModule, ...]
    """
    The modules of the selected packages that could not be imported.
    """


# %% reading


@dataclass
class ClassOntologyReader:
    """
    Reads the classes a selection names, and those their fields refer to.
    """

    selection: ClassSelection
    """
    What to read.
    """

    expanded: Set[type] = field(default_factory=set)
    """
    The classes described with their fields.
    """

    referenced: Set[type] = field(default_factory=set)
    """
    The classes described only because a field refers to them.
    """

    enumerations: Set[type] = field(default_factory=set)
    """
    The enumerations a field refers to.
    """

    def read(self) -> ClassOntology:
        """
        :return: The descriptions of the selected classes, of the classes and
            enumerations their fields refer to, and the modules that were skipped.
        """
        skipped = self.selection.import_packages()
        ignored = frozenset(
            name
            for base in self.selection.ignored_bases
            for name in self.field_names(self.selection.class_named(base))
        )
        pending = self.selected_classes()
        field_types: Dict[type, Dict[str, FieldType]] = {}
        while pending:
            described = pending.pop()
            if described in self.expanded:
                continue
            self.expanded.add(described)
            field_types[described] = self.read_field_types(described, ignored)
            pending.extend(self.newly_referenced(field_types[described].values()))
        self.referenced -= self.expanded
        return ClassOntology(
            classes=tuple(
                sorted(
                    [
                        self.describe(described, field_types)
                        for described in self.expanded
                    ]
                    + [self.describe(described, {}) for described in self.referenced],
                    key=lambda description: description.identifier,
                )
            ),
            enumerations=tuple(
                sorted(
                    (self.describe_enumeration(found) for found in self.enumerations),
                    key=lambda description: description.identifier,
                )
            ),
            skipped_modules=tuple(skipped),
        )

    def selected_classes(self) -> List[type]:
        """
        :return: The roots and their subclasses in the selected packages.
        """
        selected: List[type] = []
        for root in map(self.selection.class_named, self.selection.roots):
            selected.append(root)
            pending = [root]
            while pending:
                for subclass in pending.pop().__subclasses__():
                    if subclass not in selected and self.selection.selects(subclass):
                        selected.append(subclass)
                        pending.append(subclass)
        return selected

    @staticmethod
    def field_names(described: type) -> FrozenSet[str]:
        """
        :return: The names of a class's fields, inherited ones included.
        """
        if not is_dataclass(described):
            return frozenset()
        return frozenset(declared.name for declared in fields(described))

    def read_field_types(
        self, described: type, ignored: FrozenSet[str]
    ) -> Dict[str, FieldType]:
        """
        :return: What each content field of a class holds, by name, leaving out the
            fields of the ignored bases.
        """
        reader = FieldTypeReader(described=described)
        return {
            declared.name: reader.read(declared)
            for declared in DocumentedClass.of(described).declared_content_fields
            if declared.name not in ignored
        }

    def newly_referenced(self, found: List[FieldType]) -> List[type]:
        """
        Note the classes and enumerations fields refer to.

        :return: Those of them to describe with their fields, not yet noted.
        """
        to_expand: List[type] = []
        for field_type in found:
            referenced = field_type.referenced
            if referenced is None:
                continue
            if issubclass(referenced, Enum):
                self.enumerations.add(referenced)
            elif self.selection.expands(referenced):
                to_expand.append(referenced)
            else:
                self.referenced.add(referenced)
        return to_expand

    def described_classes(self) -> Set[type]:
        """
        :return: Every class described, with its fields or without.
        """
        return self.expanded | self.referenced

    def parents_of(self, described: type) -> Tuple[str, ...]:
        """
        :return: The nearest described classes a class specialises, through those that
            are not described, such as mixins.
        """
        parents: List[str] = []
        pending = list(described.__bases__)
        while pending:
            base = pending.pop(0)
            if base in self.described_classes():
                if identifier_of(base) not in parents:
                    parents.append(identifier_of(base))
            else:
                pending.extend(base.__bases__)
        return tuple(parents)

    def describe(
        self, described: type, field_types: Dict[type, Dict[str, FieldType]]
    ) -> ClassDescription:
        """
        :return: The description of a class, with the fields it adds to the described
            classes it specialises, and those whose type it narrows, when it is described
            with its fields.
        """
        documented = DocumentedClass.of(described)
        own_types = field_types.get(described, {})
        inherited = frozenset(
            name
            for ancestor in described.__mro__[1:]
            if ancestor in self.expanded
            for name, field_type in field_types.get(ancestor, {}).items()
            if own_types.get(name) == field_type
        )
        declared_fields = (
            {declared.name: declared for declared in fields(described)}
            if (is_dataclass(described))
            else {}
        )
        return ClassDescription(
            identifier=identifier_of(described),
            name=described.__name__,
            documentation=documented.documentation,
            parents=self.parents_of(described),
            fields=tuple(
                FieldDescription(
                    name=name,
                    documentation=documented.meanings.get(name, ""),
                    field_type=field_type,
                    default=self.default_of(declared_fields[name]),
                )
                for name, field_type in own_types.items()
                if name not in inherited
            ),
        )

    @staticmethod
    def default_of(declared: Field) -> Optional[str]:
        """
        :return: A field's default written as Python, when it is a plain value.
        """
        if declared.default is MISSING or not isinstance(declared.default, PLAIN_TYPES):
            return None
        return repr(declared.default)

    @staticmethod
    def describe_enumeration(found: type) -> EnumerationDescription:
        """
        :return: The description of an enumeration.
        """
        return EnumerationDescription(
            identifier=identifier_of(found),
            name=found.__name__,
            documentation=DocumentedClass.of(found).documentation,
            members=tuple(member.name for member in found),
        )
