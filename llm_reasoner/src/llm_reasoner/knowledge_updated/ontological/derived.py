"""
Checks on a hypothesis that come from the declarations and are never written by hand.

The declarations already say how many values a property needs, which properties are
alternatives, what a concept fixes or allows, what a reference may name, and where each
kind of fragment is declared. Reading them back means a rule cannot drift from the
declaration it enforces, and a property added to a concept is checked without anyone
remembering to write its rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from operator import attrgetter
from typing import Any, Iterable

from llm_reasoner.knowledge_updated.ontological.acceptance import (
    AcceptanceViolation,
    EvaluationContext,
    InstanceCheck,
    follow,
    name_of,
)
from llm_reasoner.knowledge_updated.ontological.concepts import ConceptMeta
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.ontology import (
    Ontology,
    PropertyDeclaration,
)
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import (
    PropertyRange,
    RangeResolver,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    PropertyKind,
    Severity,
)

# %% the rules


class DerivedRule(StrEnum):
    """
    A rule read back from the declarations.
    """

    MINIMUM_COUNT_NOT_MET = "minimum_count_not_met"
    """
    A property holds fewer values than its declaration requires, so the fragment is
    incomplete.
    """

    EXCLUSIVE_GROUP_NOT_EXACTLY_ONE = "exclusive_group_not_exactly_one"
    """
    Of a group of alternative properties, none or several are given, so the fragment does
    not say which alternative holds.
    """

    VALUE_OF_WRONG_TYPE = "value_of_wrong_type"
    """
    A property holds a value of a type other than the one it is declared to hold.
    """

    FIXED_VALUE_NOT_HELD = "fixed_value_not_held"
    """
    A property does not have the value its concept fixes for every instance.
    """

    VALUE_NOT_ALLOWED = "value_not_allowed"
    """
    A property holds a value that its concept does not allow.
    """

    REFERENCE_OUTSIDE_CONCEPT_ROOT = "reference_outside_concept_root"
    """
    A reference names a term that is not a concrete descendant of the concept it is closed
    over, so the resolver could not tell what kind of thing to look for.
    """

    ABSTRACT_CONCEPT_INSTANTIATED = "abstract_concept_instantiated"
    """
    A fragment is an instance of a concept that only groups more specific ones, so it
    discards the specific concept needed for querying.
    """

    REFERENCE_NOT_DECLARED = "reference_not_declared"
    """
    A fragment refers to another that its structure does not declare, so the hypothesis is
    not self-contained and cannot be cached, versioned or shipped as a unit.
    """


# %% what a property is declared to be


@dataclass(frozen=True)
class PropertySpec:
    """
    A property as declared, with the range its type hint gives.
    """

    meta: PropertyMeta
    """
    The property's characteristics.
    """

    range: PropertyRange
    """
    What the property's type hint says it holds.
    """


@dataclass(frozen=True)
class ValueRestrictions:
    """
    What a concept, with the concepts it specialises, requires of property values.
    """

    fixed: dict[str, Any] = field(default_factory=dict)
    """
    The value each fixed property must have.
    """

    allowed: dict[str, frozenset[Any]] = field(default_factory=dict)
    """
    The values each restricted property may have.
    """


# %% the checks


@dataclass
class DerivedChecks(InstanceCheck):
    """
    Checks every fragment of a hypothesis against the declarations of its concept.
    """

    ontology: Ontology
    """
    The declarations the fragments are instances of.
    """

    ranges: RangeResolver = field(default_factory=RangeResolver)
    """
    Reads what the type hint of each property says.
    """

    _declarations_by_owner: dict[type, list[PropertyDeclaration]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The properties each concept writes, grouped once because the ontology is fixed.
    """

    _specs: dict[type, dict[str, PropertySpec]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The properties of each concept including inherited ones.
    """

    def check(self, context: EvaluationContext) -> Iterable[AcceptanceViolation]:
        for fragment in context.instances:
            if not self._is_described(type(fragment)):
                continue
            concept = self.ontology.find_concept(type(fragment))
            if concept is not None:
                yield from self._abstract(fragment, concept)
            yield from self._properties(fragment)
            yield from self._restrictions(fragment)
        yield from self._ownership_closure(context)

    def _is_described(self, cls: type) -> bool:
        """
        Return whether a class is a declared concept or specialises one.

        A specialisation nobody declared is checked as the declared concept it inherits
        from.
        """
        return any(self.ontology.find_concept(klass) for klass in cls.__mro__)

    # -- instantiation ----------------------------------------------------

    @staticmethod
    def _abstract(fragment: Any, concept: ConceptMeta) -> Iterable[AcceptanceViolation]:
        """
        Report an instance of a concept that is never instantiated.
        """
        if concept.abstract:
            yield AcceptanceViolation(
                DerivedRule.ABSTRACT_CONCEPT_INSTANTIATED,
                Severity.ERROR,
                name_of(fragment),
                f"{concept.local_name} is abstract, so a more specific concept is needed",
            )

    # -- properties -------------------------------------------------------

    def _properties(self, fragment: Any) -> Iterable[AcceptanceViolation]:
        """
        Check the values of each property against its declaration.
        """
        specs = self._specs_of(type(fragment))
        for name, spec in specs.items():
            values = follow(fragment, name)
            if len(values) < spec.meta.minimum_count:
                yield self._violation(
                    DerivedRule.MINIMUM_COUNT_NOT_MET,
                    fragment,
                    f"'{name}' has {len(values)} value(s), needs at least "
                    f"{spec.meta.minimum_count}",
                )
            for value in values:
                if spec.range.element_types and not self._fits(value, spec):
                    yield self._violation(
                        DerivedRule.VALUE_OF_WRONG_TYPE,
                        fragment,
                        f"'{name}' holds {type(value).__name__}, expected "
                        + " or ".join(t.__name__ for t in spec.range.element_types),
                    )
                if spec.meta.concept_root is not None and not self._names_a_descendant(
                    value, spec.meta.concept_root
                ):
                    yield self._violation(
                        DerivedRule.REFERENCE_OUTSIDE_CONCEPT_ROOT,
                        fragment,
                        f"'{name}' must name a concrete "
                        f"{spec.meta.concept_root.__name__} concept",
                    )
        yield from self._alternatives(fragment, specs)

    def _alternatives(
        self, fragment: Any, specs: dict[str, PropertySpec]
    ) -> Iterable[AcceptanceViolation]:
        """
        Check that exactly one member of each group of alternatives is given.
        """
        groups: dict[str, list[str]] = {}
        for name, spec in specs.items():
            if spec.meta.exclusive_group is not None:
                groups.setdefault(spec.meta.exclusive_group, []).append(name)
        for members in groups.values():
            given = [name for name in members if follow(fragment, name)]
            if len(given) != 1:
                yield self._violation(
                    DerivedRule.EXCLUSIVE_GROUP_NOT_EXACTLY_ONE,
                    fragment,
                    f"exactly one of ({', '.join(members)}) must be given; found "
                    + (", ".join(given) if given else "none"),
                )

    @staticmethod
    def _fits(value: Any, spec: PropertySpec) -> bool:
        """
        Return whether a value has one of the types a property is declared to hold.
        """
        if spec.meta.kind is PropertyKind.RELATION:
            return isinstance(value, spec.range.element_types)
        if spec.meta.kind is PropertyKind.REFERENCE:
            return isinstance(value, ConceptReference)
        return any(
            DerivedChecks._is_plain_value(value, held)
            for held in spec.range.element_types
        )

    @staticmethod
    def _is_plain_value(value: Any, held: type) -> bool:
        """
        Return whether a value is a number, string, boolean or enumeration member of a type.
        """
        if held is bool:
            return isinstance(value, bool)
        if held is int:
            return isinstance(value, int) and not isinstance(value, bool)
        if held is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if issubclass(held, Enum):
            return isinstance(value, held)
        return isinstance(value, held)

    def _names_a_descendant(self, value: Any, root: type) -> bool:
        """
        Return whether a reference names a concrete concept that is or specialises a root.
        """
        if not isinstance(value, ConceptReference):
            return False
        named = self.ontology.find_concept_by_iri(value.iri)
        return named is not None and not named.abstract and issubclass(named.cls, root)

    # -- restrictions -----------------------------------------------------

    def _restrictions(self, fragment: Any) -> Iterable[AcceptanceViolation]:
        """
        Check the values that the concept fixes or allows.
        """
        restrictions = self._restrictions_of(type(fragment))
        for name, fixed in restrictions.fixed.items():
            current = attrgetter(name)(fragment)
            if current != fixed:
                yield self._violation(
                    DerivedRule.FIXED_VALUE_NOT_HELD,
                    fragment,
                    f"'{name}' is {current!r} but the concept fixes it to {fixed!r}",
                )
        for name, allowed in restrictions.allowed.items():
            current = attrgetter(name)(fragment)
            if current is not None and current not in allowed:
                yield self._violation(
                    DerivedRule.VALUE_NOT_ALLOWED,
                    fragment,
                    f"'{name}' is {current!r}, which the concept does not allow",
                )

    def _restrictions_of(self, cls: type) -> ValueRestrictions:
        """
        Return what a concept and its ancestors fix and allow, nearest concept winning.
        """
        fixed: dict[str, Any] = {}
        allowed: dict[str, frozenset[Any]] = {}
        for klass in reversed(cls.__mro__):
            concept = self.ontology.find_concept(klass)
            if concept is None:
                continue
            for restriction in concept.axioms.fixed_values:
                fixed[restriction.property_name] = restriction.value
            for permitted in concept.axioms.allowed_values:
                name = permitted.property_name
                allowed[name] = (
                    allowed[name] & permitted.values
                    if name in allowed
                    else permitted.values
                )
        return ValueRestrictions(fixed=fixed, allowed=allowed)

    # -- declaring fragments ----------------------------------------------

    def _ownership_closure(
        self, context: EvaluationContext
    ) -> Iterable[AcceptanceViolation]:
        """
        Check that every fragment a structure refers to is declared by that structure.

        A fragment is declared when it is in a collection its structure owns. Kinds that
        no owned collection ranges over are nested inside the fragment that carries them,
        and need no declaration.
        """
        for structure in context.instances:
            owning = {
                name: spec
                for name, spec in self._specs_of(type(structure)).items()
                if spec.meta.owns
            }
            if not owning:
                continue
            declared = {id(m) for name in owning for m in follow(structure, name)}
            declarable = tuple(
                held for spec in owning.values() for held in spec.range.element_types
            )
            for fragment in self._reachable(structure):
                yield from self._undeclared_references(
                    fragment, structure, declared, declarable
                )

    def _undeclared_references(
        self,
        fragment: Any,
        structure: Any,
        declared: set[int],
        declarable: tuple[type, ...],
    ) -> Iterable[AcceptanceViolation]:
        """
        Report references from a fragment to declarable fragments its structure lacks.
        """
        for name, spec in self._specs_of(type(fragment)).items():
            if spec.meta.kind is not PropertyKind.RELATION or spec.meta.owns:
                continue
            for referenced in follow(fragment, name):
                if (
                    isinstance(referenced, declarable)
                    and id(referenced) not in declared
                ):
                    yield self._violation(
                        DerivedRule.REFERENCE_NOT_DECLARED,
                        fragment,
                        f"'{name}' refers to {name_of(referenced)}, which "
                        f"{name_of(structure)} does not declare",
                    )

    def _reachable(self, root: Any) -> list[Any]:
        """
        Return the root and every fragment reachable from it through any relation.
        """
        seen: dict[int, Any] = {id(root): root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for name, spec in self._specs_of(type(current)).items():
                if spec.meta.kind is not PropertyKind.RELATION:
                    continue
                for neighbour in follow(current, name):
                    if (
                        isinstance(neighbour, KnowledgeEntity)
                        and id(neighbour) not in seen
                    ):
                        seen[id(neighbour)] = neighbour
                        frontier.append(neighbour)
        return list(seen.values())

    # -- reading the declarations -----------------------------------------

    def _specs_of(self, cls: type) -> dict[str, PropertySpec]:
        """
        Return the properties of a concept including those it inherits.
        """
        if cls not in self._specs:
            self._specs[cls] = {
                declaration.field_name: PropertySpec(
                    declaration.meta,
                    self.ranges.resolve(declaration.owner, declaration.field_name),
                )
                for klass in reversed(cls.__mro__)
                for declaration in self._declared_by(klass)
            }
        return self._specs[cls]

    def _declared_by(self, owner: type) -> list[PropertyDeclaration]:
        """
        Return the properties a concept writes in its own body.
        """
        if not self._declarations_by_owner:
            for declaration in self.ontology.property_declarations():
                self._declarations_by_owner.setdefault(declaration.owner, []).append(
                    declaration
                )
        return self._declarations_by_owner.get(owner, [])

    @staticmethod
    def _violation(
        rule: DerivedRule, fragment: Any, message: str
    ) -> AcceptanceViolation:
        """
        Report a derived rule broken by a fragment.
        """
        return AcceptanceViolation(rule, Severity.ERROR, name_of(fragment), message)
