"""
Logical statements a concept makes about its own instances.

Axioms are always true of every instance of a concept, which separates them from the
constraints that check relationships between several objects of a hypothesis. They are
recorded on the concept, so the schema, the prompt and the checks all read the same
statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping, Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% invalid axioms


class AxiomFault(StrEnum):
    """
    Why an axiom cannot hold of the class that declares it.
    """

    UNKNOWN_PROPERTY = "unknown_property"
    """
    The axiom names a property the class does not declare.
    """

    NOT_A_FUNCTIONAL_ATTRIBUTE = "not_a_functional_attribute"
    """
    The axiom restricts a property that is not a plain attribute with at most one value.
    """

    PROPERTY_FIXED_TWICE = "property_fixed_twice"
    """
    The class fixes the same property more than once.
    """

    NO_ALLOWED_VALUES = "no_allowed_values"
    """
    The class restricts a property to an empty set of values, which nothing satisfies.
    """

    PROPERTY_RESTRICTED_TWICE = "property_restricted_twice"
    """
    The class restricts the same property to allowed values more than once.
    """

    FIXED_VALUE_NOT_ALLOWED = "fixed_value_not_allowed"
    """
    The class fixes a property to a value that its own allowed values exclude.
    """


@dataclass
class InvalidAxiom(LLMReasonerError):
    """
    A class declares an axiom that cannot hold of it.
    """

    reason: AxiomFault
    """
    Why the axiom is invalid.
    """

    subject: str
    """
    The property the axiom names.
    """

    def __str__(self) -> str:
        return f"invalid axiom on '{self.subject}': {self.reason.value}"


# %% axioms


@dataclass(frozen=True)
class FixedValue:
    """
    A property whose value is the same for every instance of a concept.
    """

    property_name: str
    """
    Name of the functional attribute the concept fixes.
    """

    value: Any
    """
    The value every instance has.
    """


@dataclass(frozen=True)
class AllowedValues:
    """
    The values a concept permits a property to take, out of all it could.
    """

    property_name: str
    """
    Name of the functional attribute the concept restricts.
    """

    values: frozenset[Any]
    """
    The values an instance may have. A single value is the same as fixing it.
    """


@dataclass(frozen=True)
class ClassAxioms:
    """
    Everything a concept states about its instances beyond its properties.
    """

    disjoint_with: tuple[str, ...] = ()
    """
    Names of the concepts that share no instance with this one. Names rather than classes,
    so a concept can be declared disjoint from one defined later in the same file.
    """

    disjoint_siblings: bool = False
    """
    Whether the direct subconcepts of this concept share no instance with each other.
    """

    may_overlap_with: tuple[str, ...] = ()
    """
    Names of the concepts that may share instances with this one, by intent. Saying so
    settles the question of whether two sibling concepts should be disjoint.
    """

    fixed_values: tuple[FixedValue, ...] = ()
    """
    The properties whose value is the same for every instance.
    """

    allowed_values: tuple[AllowedValues, ...] = ()
    """
    The properties restricted to some of the values they could take.
    """

    @classmethod
    def of(
        cls,
        disjoint_with: Iterable[str] = (),
        disjoint_siblings: bool = False,
        may_overlap_with: Iterable[str] = (),
        fixed_values: Optional[Mapping[str, Any]] = None,
        allowed_values: Optional[Mapping[str, Iterable[Any]]] = None,
    ) -> ClassAxioms:
        """
        Build axioms from the arguments of a concept declaration.

        :param fixed_values: Property names with the value each always has.
        :param allowed_values: Property names with the values each may have.
        """
        return cls(
            disjoint_with=tuple(disjoint_with),
            disjoint_siblings=disjoint_siblings,
            may_overlap_with=tuple(may_overlap_with),
            fixed_values=tuple(
                FixedValue(property_name=name, value=value)
                for name, value in (fixed_values or {}).items()
            ),
            allowed_values=tuple(
                AllowedValues(property_name=name, values=frozenset(values))
                for name, values in (allowed_values or {}).items()
            ),
        )

    def fixed_value_of(self, property_name: str) -> Optional[Any]:
        """
        Return the value this concept fixes for a property, or ``None`` when it fixes none.
        """
        return next(
            (
                fixed.value
                for fixed in self.fixed_values
                if fixed.property_name == property_name
            ),
            None,
        )

    def allowed_values_of(self, property_name: str) -> Optional[frozenset[Any]]:
        """
        Return the values this concept allows a property, or ``None`` when it restricts none.
        """
        return next(
            (
                allowed.values
                for allowed in self.allowed_values
                if allowed.property_name == property_name
            ),
            None,
        )

    def check_against(self, owner: type) -> None:
        """
        Check that the class can hold the axioms it declares.

        :param owner: The class declaring the axioms. Properties it inherits count.
        :raises InvalidAxiom: on the first axiom that cannot hold.
        """
        declared = PropertyMeta.declared_on(owner)
        already_fixed: set[str] = set()
        for fixed in self.fixed_values:
            if fixed.property_name in already_fixed:
                raise InvalidAxiom(AxiomFault.PROPERTY_FIXED_TWICE, fixed.property_name)
            already_fixed.add(fixed.property_name)
            self._check_restrictable(declared, fixed.property_name)
        already_restricted: set[str] = set()
        for allowed in self.allowed_values:
            if allowed.property_name in already_restricted:
                raise InvalidAxiom(
                    AxiomFault.PROPERTY_RESTRICTED_TWICE, allowed.property_name
                )
            already_restricted.add(allowed.property_name)
            self._check_restrictable(declared, allowed.property_name)
            if not allowed.values:
                raise InvalidAxiom(AxiomFault.NO_ALLOWED_VALUES, allowed.property_name)
        for fixed in self.fixed_values:
            allowed_values = self.allowed_values_of(fixed.property_name)
            if allowed_values is not None and fixed.value not in allowed_values:
                raise InvalidAxiom(
                    AxiomFault.FIXED_VALUE_NOT_ALLOWED, fixed.property_name
                )

    @staticmethod
    def _check_restrictable(
        declared: Mapping[str, PropertyMeta], property_name: str
    ) -> None:
        """
        Check that a property exists and is a plain attribute with at most one value.
        """
        meta = declared.get(property_name)
        if meta is None:
            raise InvalidAxiom(AxiomFault.UNKNOWN_PROPERTY, property_name)
        if meta.kind is not PropertyKind.ATTRIBUTE or not meta.functional:
            raise InvalidAxiom(AxiomFault.NOT_A_FUNCTIONAL_ATTRIBUTE, property_name)
