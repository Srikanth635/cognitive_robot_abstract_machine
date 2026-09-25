from __future__ import annotations

import dataclasses
import enum
import re
import sys
import types
import typing
from dataclasses import dataclass, field

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.match import AttributeMatch, Match
from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)
from krrood.utils import recursive_subclasses
from typing_extensions import Any, Dict, List, Optional, Sequence

# %% slot classification

GENERATABLE_TYPES = (bool, int, float, str)
"""Types whose values a model may invent outright."""

UNION_ORIGINS = (typing.Union, types.UnionType)
"""Origins identifying a union annotation, covering both spellings of one."""


class SlotKind(enum.Enum):
    """The kind of knowledge a slot is missing."""

    VALUE = enum.auto()
    """A self-identifying value that does not exist yet and may be generated."""

    REFERENT = enum.auto()
    """An individual that must be selected from an extension, never invented."""

    SORTAL = enum.auto()
    """
    A *kind* rather than an individual, declared as ``Type[T]``.

    Selected from ``T`` and its subclasses, which reflection derives without a world.
    Choosing a kind is abduction over categories -- "something to drink from" is a cup,
    a mug or a glass -- which is what a language model is good at, and unlike an
    individual a kind can be named from the class hierarchy alone.
    """

    BOUND = enum.auto()
    """Already supplied by the caller, and therefore context rather than a question."""


def union_alternatives(declared_type: Any) -> Optional[List[type]]:
    """
    Split a union annotation into its branches.

    Both spellings are recognised, and ``None`` is dropped because a slot's absence is
    expressed by leaving it out rather than by filling it with null.

    :param declared_type: An annotated type.
    :return: The branches, or ``None`` when the annotation is not a union.
    """
    if typing.get_origin(declared_type) not in UNION_ORIGINS:
        return None
    return [
        argument
        for argument in typing.get_args(declared_type)
        if argument is not type(None)
    ]


SORTAL_ANNOTATION = re.compile(r"^(?:typing\.)?[Tt]ype\[\s*([\w.]+)\s*]$")
"""Matches a string annotation of the form ``Type[X]``, capturing ``X``."""


def raw_annotation_of(owner: type, attribute_name: str) -> Any:
    """
    Read one annotation exactly as it was written.

    Annotations are read from the class dictionaries directly rather than through
    ``typing.get_type_hints``, which evaluates *every* annotation on a class: a single
    name that exists only under ``TYPE_CHECKING`` would then break reflection for
    unrelated attributes.

    :param owner: The class declaring the attribute.
    :param attribute_name: The field name on the owner.
    :return: The annotation, a string when the module defers evaluation, or ``None``
        when the attribute is not annotated.
    """
    for ancestor in owner.__mro__:
        annotations = ancestor.__dict__.get("__annotations__", {})
        if attribute_name in annotations:
            return annotations[attribute_name]
    return None


def module_namespace_of(owner: type) -> Dict[str, Any]:
    """
    :param owner: The class whose defining module to look up.
    :return: That module's namespace, empty when the module is unavailable.
    """
    module = sys.modules.get(owner.__module__)
    if module is None:
        return {}
    return vars(module)


def sortal_base(raw_annotation: Any, namespace: Dict[str, Any]) -> Optional[type]:
    """
    Recognise a slot that asks for a kind rather than a value or an individual.

    Deferred annotations arrive as strings, so a string of the form ``Type[X]`` is
    matched and ``X`` resolved against the declaring module. A name that resolves to
    nothing yields no sortal rather than an error, because such a name is one that
    exists only for type checkers and can carry no extension.

    :param raw_annotation: The annotation as written.
    :param namespace: The declaring module's namespace, used to resolve a string.
    :return: The base class the slot ranges over, or ``None``.
    """
    if isinstance(raw_annotation, str):
        matched = SORTAL_ANNOTATION.match(raw_annotation.strip())
        if matched is None:
            return None
        resolved = namespace.get(matched.group(1).split(".")[-1])
        return resolved if isinstance(resolved, type) else None
    if typing.get_origin(raw_annotation) is not type:
        return None
    arguments = typing.get_args(raw_annotation)
    if not arguments or not isinstance(arguments[0], type):
        return None
    return arguments[0]


def sortal_domain(base: type) -> List[type]:
    """
    :param base: The class a sortal slot ranges over.
    :return: That class and every subclass of it, which is the extension of kinds the
        model may choose from.
    """
    return [base] + recursive_subclasses(base)


def is_generatable(declared_type: Any) -> bool:
    """
    :param declared_type: An annotated type.
    :return: Whether a model may invent a value of this type outright.
    """
    if not isinstance(declared_type, type):
        return False
    return issubclass(declared_type, enum.Enum) or issubclass(
        declared_type, GENERATABLE_TYPES
    )


def type_display_name(declared_type: Any, alternatives: Optional[List[type]]) -> str:
    """
    :param declared_type: An annotated type.
    :param alternatives: The union branches, when the annotation is a union.
    :return: A name suitable for a prompt.
    """
    if alternatives is not None:
        return " or ".join(alternative.__name__ for alternative in alternatives)
    return declared_type.__name__


@dataclass
class Slot:
    """One leaf of a match, described entirely by reflection over the query."""

    path: str
    """Dotted access path, which is the shared vocabulary across query and prompt."""

    declared_type: Any
    """The annotated type of the attribute, which may be a union."""

    kind: SlotKind
    """What kind of knowledge is missing."""

    attribute_match: AttributeMatch
    """The krrood node this slot was derived from."""

    owner: type
    """The class that declares the attribute."""

    attribute_name: str
    """The bare field name on the owner."""

    docstring: Optional[str] = None
    """The attribute's own docstring, when it has one."""

    default: Any = None
    """The attribute's declared default, when it has one."""

    has_default: bool = False
    """Whether the attribute declares a default at all."""

    alternatives: Optional[List[type]] = None
    """The union branches, when the declared type is a union."""

    sortal_base: Optional[type] = None
    """The class a sortal slot ranges over, when the slot asks for a kind."""

    members: Optional[List[Any]] = None
    """The enum members, when the declared type is an enum."""

    domain: Optional[List[Any]] = None
    """The candidate individuals, when the slot is a referent with an extension."""

    bound_value: Any = None
    """The caller-supplied value, when the slot is bound."""

    group_path: Optional[str] = None
    """
    Access path of the nested description this slot belongs to, or ``None`` when the
    slot sits directly on the constructed type.
    """

    group_type: Optional[type] = None
    """The type of the nested description, when the slot sits inside one."""

    group_docstring: Optional[str] = None
    """
    Docstring of the attribute holding the nested description.

    Carried because the nesting node contributes no slot of its own, so without this
    the sentence explaining what the nested description is *for* would be lost.
    """

    @property
    def type_name(self) -> str:
        """:return: The declared type's name, joining union branches with "or"."""
        if self.sortal_base is not None:
            return f"a kind of {self.sortal_base.__name__}"
        return type_display_name(self.declared_type, self.alternatives)

    @property
    def is_enum(self) -> bool:
        """:return: Whether the declared type is a single enumeration."""
        return isinstance(self.declared_type, type) and issubclass(
            self.declared_type, enum.Enum
        )

    @property
    def selects_from_domain(self) -> bool:
        """:return: Whether the slot is answered by a token into its extension."""
        return self.kind in (SlotKind.REFERENT, SlotKind.SORTAL)

    @property
    def decodable_types(self) -> List[type]:
        """
        :return: The types a value for this slot may take, in declaration order. A
            union offers several; anything else offers exactly one.
        """
        if self.alternatives is not None:
            return self.alternatives
        return [self.declared_type]


# %% reflection


@dataclass
class SlotReflector:
    """Flattens a match into the slots a model has to answer for."""

    docstrings_by_owner: Dict[type, Dict[str, str]] = field(default_factory=dict)
    """Attribute docstrings, cached per declaring class."""

    namespaces_by_owner: Dict[type, Dict[str, Any]] = field(default_factory=dict)
    """
    Declaring modules' namespaces, cached per class, used to resolve deferred
    annotations.

    krrood unwraps ``Type[T]`` to ``T`` when it builds a variable, so the annotation as
    written is the only place a sortal slot can still be told apart from a slot asking
    for an individual of the same type.
    """

    def reflect(self, expression: Match) -> List[Slot]:
        """
        Describe every leaf of a match.

        Nested matches contribute their own leaves; the nesting node is not itself a
        slot.

        :param expression: The match to reflect over.
        :return: One slot per leaf, in breadth-first order.
        """
        return [
            self._slot_for(attribute_match)
            for attribute_match in expression.matches_with_variables
        ]

    def _slot_for(self, attribute_match: AttributeMatch) -> Slot:
        """
        :param attribute_match: One leaf of a match.
        :return: The slot describing it.
        """
        declared_type = attribute_match.assigned_variable._type_
        assigned_value = attribute_match.assigned_value
        owner = attribute_match.parent.type
        attribute_name = attribute_match.variable._attribute_name_
        alternatives = union_alternatives(declared_type)
        base = self._sortal_base_for(owner, attribute_name, assigned_value)
        kind = self._classify(declared_type, alternatives, assigned_value, base)
        is_enum = isinstance(declared_type, type) and issubclass(
            declared_type, enum.Enum
        )

        default, has_default = self._default_for(owner, attribute_name)
        group_path, group_type, group_docstring = self._group_for(attribute_match)

        return Slot(
            path=attribute_match.name_from_variable_access_path,
            declared_type=declared_type,
            kind=kind,
            attribute_match=attribute_match,
            owner=owner,
            attribute_name=attribute_name,
            docstring=self._docstring_for(owner, attribute_name),
            default=default,
            has_default=has_default,
            alternatives=alternatives,
            sortal_base=base,
            members=list(declared_type) if is_enum else None,
            domain=(
                sortal_domain(base)
                if base is not None
                else self._domain_of(assigned_value)
            ),
            bound_value=assigned_value if kind is SlotKind.BOUND else None,
            group_path=group_path,
            group_type=group_type,
            group_docstring=group_docstring,
        )

    def _sortal_base_for(
        self, owner: type, attribute_name: str, assigned_value: Any
    ) -> Optional[type]:
        """
        Decide whether an unfilled slot asks for a kind.

        Only the one annotation is read, and only for slots that would otherwise be
        treated as individuals, so a class carrying names that exist solely for type
        checkers still reflects correctly.

        :param owner: The class declaring the attribute.
        :param attribute_name: The field name on the owner.
        :param assigned_value: What the caller wrote in the slot.
        :return: The class the slot ranges over, or ``None``.
        """
        if not isinstance(assigned_value, type(Ellipsis)):
            return None
        annotation = raw_annotation_of(owner, attribute_name)
        if annotation is None:
            return None
        if owner not in self.namespaces_by_owner:
            self.namespaces_by_owner[owner] = module_namespace_of(owner)
        return sortal_base(annotation, self.namespaces_by_owner[owner])

    @staticmethod
    def _classify(
        declared_type: Any,
        alternatives: Optional[List[type]],
        assigned_value: Any,
        sortal_base_class: Optional[type] = None,
    ) -> SlotKind:
        """
        Decide what kind of knowledge is missing, from the slot's contents alone.

        A union of generatable branches is a value rather than a referent: a model can
        choose the branch from context, which is why the branches are kept rather than
        narrowed away.

        :param declared_type: The annotated type of the attribute.
        :param alternatives: The union branches, when the annotation is a union.
        :param assigned_value: What the caller wrote in the slot.
        :param sortal_base_class: The class a sortal slot ranges over, when it is one.
        :return: The slot kind.
        """
        if isinstance(assigned_value, SymbolicExpression):
            return SlotKind.REFERENT
        if not isinstance(assigned_value, type(Ellipsis)):
            return SlotKind.BOUND
        if sortal_base_class is not None:
            return SlotKind.SORTAL
        if alternatives is not None:
            if all(is_generatable(alternative) for alternative in alternatives):
                return SlotKind.VALUE
            return SlotKind.REFERENT
        if is_generatable(declared_type):
            return SlotKind.VALUE
        return SlotKind.REFERENT

    @staticmethod
    def _domain_of(assigned_value: Any) -> Optional[List[Any]]:
        """
        Materialise the extension behind a krrood variable.

        A variable without an explicit domain takes its extension from the symbol
        graph, so world individuals reach the caller without this package importing a
        world.

        .. note::
            The symbol graph holds its members weakly by design. An individual that
            nothing else references is collected and drops out of the extension, so an
            empty result can mean the individuals were collected rather than that none
            exist. Whoever creates them keeps them alive.

        :param assigned_value: What the caller wrote in the slot.
        :return: The candidate individuals, or ``None`` when the slot holds no variable.
        """
        if not isinstance(assigned_value, SymbolicExpression):
            return None
        return list(assigned_value.tolist())

    def _group_for(self, attribute_match: AttributeMatch) -> tuple:
        """
        Describe the nested description a slot belongs to.

        :param attribute_match: One leaf of a match.
        :return: The nesting attribute's access path, its type and its docstring, or
            three ``None`` when the leaf sits directly on the constructed type.
        """
        nesting_node = attribute_match.parent
        if not isinstance(nesting_node, AttributeMatch):
            return None, None, None
        return (
            nesting_node.name_from_variable_access_path,
            nesting_node.type,
            self._docstring_for(
                nesting_node.parent.type, nesting_node.variable._attribute_name_
            ),
        )

    @staticmethod
    def _default_for(owner: type, attribute_name: str) -> tuple:
        """
        Find an attribute's declared default.

        A default is worth showing to a model because it states the conventional value,
        which is information the type and the docstring do not carry.

        :param owner: The class declaring the attribute.
        :param attribute_name: The field name on the owner.
        :return: The default and whether one is declared.
        """
        if not dataclasses.is_dataclass(owner):
            return None, False
        for declared_field in dataclasses.fields(owner):
            if declared_field.name != attribute_name:
                continue
            if declared_field.default is not dataclasses.MISSING:
                return declared_field.default, True
            if declared_field.default_factory is not dataclasses.MISSING:
                return declared_field.default_factory(), True
            return None, False
        return None, False

    def _docstring_for(self, owner: type, attribute_name: str) -> Optional[str]:
        """
        :param owner: The class declaring the attribute.
        :param attribute_name: The field name on the owner.
        :return: The attribute's docstring, when it has one.
        """
        if owner not in self.docstrings_by_owner:
            self.docstrings_by_owner[owner] = _attribute_docstrings(owner)
        return self.docstrings_by_owner[owner].get(attribute_name)


def reflect_slots(expression: Match) -> List[Slot]:
    """
    Flatten a match into slots.

    :param expression: The match to reflect over.
    :return: One slot per leaf.
    """
    return SlotReflector().reflect(expression)


def open_slots(slots: Sequence[Slot]) -> List[Slot]:
    """
    :param slots: Every reflected slot.
    :return: Only the slots a model has to answer for.
    """
    return [slot for slot in slots if slot.kind is not SlotKind.BOUND]