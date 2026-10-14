"""
Checks on an ontology's declarations as a whole.

Each declaration is already checked alone when it is made. The checks here need to see
several declarations together: what a property ranges over, whether two properties that
share an identifier agree, and whether a concept contradicts the ones it specialises.
They read the declarations and never the instances of a hypothesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Iterable, Optional, Sequence

from llm_reasoner.knowledge_updated.ontological.concepts import ConceptMeta
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
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
    ConceptLevel,
    PropertyKind,
    Rigidity,
    Severity,
)

PRIMITIVE_TYPES = (str, int, float, bool)
"""
The types an attribute may hold without a declared value partition.
"""

# %% findings


class Check(StrEnum):
    """
    A rule an ontology is checked against.
    """

    UNRESOLVABLE_ANNOTATION = "unresolvable_annotation"
    """
    A property's type hint uses a name that is defined nowhere it can see.
    """

    RANGE_KIND_MISMATCH = "range_kind_mismatch"
    """
    A property's range does not suit its kind: a relation must range over declared
    concepts, an attribute over primitives or declared value partitions, and a reference
    over term references.
    """

    RANGE_FROM_UNDECLARED_DEPENDENCY = "range_from_undeclared_dependency"
    """
    A property ranges over a declaration of a module that its own module does not depend
    on.
    """

    COLLECTION_MISMATCH = "collection_mismatch"
    """
    A property's type hint disagrees with whether it has at most one value.
    """

    MISSING_OPTIONAL = "missing_optional"
    """
    A property that starts empty is typed as if a value were always present.
    """

    INVERSE_UNRESOLVED = "inverse_unresolved"
    """
    A relation's inverse names something that is not a relation of the class it ranges
    over.
    """

    ASSUMPTION_MARKER_NOT_BOOLEAN = "assumption_marker_not_boolean"
    """
    A property that marks an assumption does not hold a boolean.
    """

    EXCLUSIVE_GROUP_TOO_SMALL = "exclusive_group_too_small"
    """
    A group of alternatives has a single member, so there is nothing to choose between.
    """

    DESCRIPTION_REFERS_TO_OCCURRENCE = "description_refers_to_occurrence"
    """
    A concept that classifies kinds has a relation to a concept of actual things.
    """

    PROPERTY_IRI_CONFLICT = "property_iri_conflict"
    """
    Properties that share an identifier differ in kind, functionality or range.
    """

    PROPERTY_REDECLARED_UNDER_OTHER_IRI = "property_redeclared_under_other_iri"
    """
    A concept writes an inherited field again under a different identifier, so one field
    stands for two properties.
    """

    DISJOINT_NAME_UNKNOWN = "disjoint_name_unknown"
    """
    A concept is declared disjoint from a name that no concept has.
    """

    DISJOINT_NAME_AMBIGUOUS = "disjoint_name_ambiguous"
    """
    A concept is declared disjoint from a name that more than one concept has.
    """

    DISJOINT_WITH_SELF = "disjoint_with_self"
    """
    A concept is declared disjoint from itself.
    """

    DISJOINT_RELATED_BY_INHERITANCE = "disjoint_related_by_inheritance"
    """
    A concept is declared disjoint from one it specialises or that specialises it.
    """

    OVERLAP_NAME_UNKNOWN = "overlap_name_unknown"
    """
    A concept is declared to overlap with a name that no concept has.
    """

    OVERLAP_CONTRADICTS_DISJOINTNESS = "overlap_contradicts_disjointness"
    """
    A concept is declared both disjoint from and overlapping with the same name.
    """

    FIXED_VALUE_CONFLICT = "fixed_value_conflict"
    """
    A concept fixes a property to a value different from the one a concept it
    specialises fixes.
    """

    RESTRICTION_WIDENS_INHERITED = "restriction_widens_inherited"
    """
    A concept allows or fixes a value that a concept it specialises does not allow, so
    an instance would belong to the ancestor without satisfying it.
    """

    SIBLINGS_OVERLAP = "siblings_overlap"
    """
    Rigid subconcepts of one concept may share instances, and nothing says whether they
    should.
    """

    ABSTRACT_WITHOUT_SUBCONCEPT = "abstract_without_subconcept"
    """
    An abstract concept that nothing specialises can never appear in a hypothesis.
    """

    UNDOCUMENTED_CONCEPT = "undocumented_concept"
    """
    A concept has no docstring.
    """

    UNDOCUMENTED_PROPERTY = "undocumented_property"
    """
    A property has no documentation under its field.
    """

    UNDOCUMENTED_MEMBER = "undocumented_member"
    """
    A member of a value partition has no documentation.
    """


@dataclass(frozen=True)
class Violation:
    """
    One rule an ontology breaks.
    """

    check: Check
    """
    The rule that was broken.
    """

    severity: Severity
    """
    How seriously the break counts.
    """

    subject: str
    """
    The declaration that breaks the rule, such as ``Liftoff`` or ``Liftoff.threshold``.
    """

    message: str
    """
    What is wrong, in terms of the declaration.
    """


@dataclass(frozen=True)
class ValidationReport:
    """
    Everything found wrong with an ontology.
    """

    violations: tuple[Violation, ...] = ()
    """
    The violations, in the order the checks found them.
    """

    @property
    def errors(self) -> tuple[Violation, ...]:
        """
        The violations that make the ontology inadmissible.
        """
        return tuple(v for v in self.violations if v.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Violation, ...]:
        """
        The violations that are legal but probably unintended.
        """
        return tuple(v for v in self.violations if v.severity is Severity.WARNING)

    @property
    def clean(self) -> bool:
        """
        Whether the report holds no errors.
        """
        return not self.errors

    def of(self, check: Check) -> tuple[Violation, ...]:
        """
        Return the violations of one check.
        """
        return tuple(v for v in self.violations if v.check is check)


# %% what a check can see


@dataclass
class ValidationContext:
    """
    The ontology under validation, and what the checks share while reading it.
    """

    ontology: Ontology
    """
    The ontology being checked.
    """

    module: Optional[str]
    """
    The module whose declarations are reported, or ``None`` for all of them.
    """

    reader: DocumentationReader
    """
    Reads the documentation written in the source.
    """

    resolver: RangeResolver
    """
    Reads what property type hints say.
    """

    _ranges: dict[PropertyDeclaration, PropertyRange] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Resolved ranges, kept because resolving evaluates annotations.
    """

    def concepts(self) -> tuple[ConceptMeta, ...]:
        """
        Return the concepts in scope.
        """
        return self.ontology.concepts(self.module)

    def declarations(self) -> tuple[PropertyDeclaration, ...]:
        """
        Return the property declarations in scope.
        """
        return self.ontology.property_declarations(self.module)

    def all_declarations(self) -> tuple[PropertyDeclaration, ...]:
        """
        Return every property declaration of the ontology, in scope or not.
        """
        return self.ontology.property_declarations()

    def in_scope(self, module: str) -> bool:
        """
        Return whether a module's declarations are reported.
        """
        return self.module is None or self.module == module

    def range_of(self, declaration: PropertyDeclaration) -> PropertyRange:
        """
        Return what a property's type hint says it holds.
        """
        if declaration not in self._ranges:
            self._ranges[declaration] = self.resolver.resolve(
                declaration.owner, declaration.field_name
            )
        return self._ranges[declaration]

    def allowed_modules(self, module: str) -> set[str]:
        """
        Return the modules whose declarations a module may refer to: itself and its
        dependencies.
        """
        declaring = next(m for m in self.ontology.modules() if m.name == module)
        return {
            declaring.name,
            *(dependency.name for dependency in declaring.transitive_dependencies()),
        }

    def module_of(self, cls: type) -> Optional[str]:
        """
        Return the module that declares a class as a concept or value partition.
        """
        concept = self.ontology.find_concept(cls)
        if concept is not None:
            return concept.module
        partition = self.ontology.find_value_partition(cls)
        return None if partition is None else partition.module

    @staticmethod
    def subject_of(declaration: PropertyDeclaration) -> str:
        """
        Return how a violation names a property.
        """
        return f"{declaration.owner.__name__}.{declaration.field_name}"


class ValidationCheckGroup(ABC):
    """
    A set of related checks run together.
    """

    @abstractmethod
    def check(self, context: ValidationContext) -> Iterable[Violation]:
        """
        Return the violations found among the declarations in scope.
        """


# %% property ranges


class PropertyRangeChecks(ValidationCheckGroup):
    """
    Checks that each property's type hint suits what the property is declared to hold.
    """

    def check(self, context: ValidationContext) -> Iterable[Violation]:
        for declaration in context.declarations():
            property_range = context.range_of(declaration)
            if not property_range.resolved:
                yield Violation(
                    Check.UNRESOLVABLE_ANNOTATION,
                    Severity.ERROR,
                    context.subject_of(declaration),
                    "the type hint uses undefined names: "
                    + ", ".join(property_range.unresolved_names),
                )
                continue
            yield from self._check_declaration(context, declaration, property_range)

    def _check_declaration(
        self,
        context: ValidationContext,
        declaration: PropertyDeclaration,
        property_range: PropertyRange,
    ) -> Iterable[Violation]:
        """
        Run every check on a property whose type hint resolved.
        """
        subject = context.subject_of(declaration)
        yield from self._range_kind(context, declaration, property_range, subject)
        yield from self._range_dependencies(
            context, declaration, property_range, subject
        )
        yield from self._cardinality(declaration, property_range, subject)
        yield from self._inverse(context, declaration, property_range, subject)
        yield from self._level(context, declaration, property_range, subject)
        yield from self._assumption_marker(declaration, property_range, subject)

    def _range_kind(
        self,
        context: ValidationContext,
        declaration: PropertyDeclaration,
        property_range: PropertyRange,
        subject: str,
    ) -> Iterable[Violation]:
        """
        Check that the range suits the kind of property.
        """
        if not property_range.element_types:
            yield Violation(
                Check.RANGE_KIND_MISMATCH,
                Severity.ERROR,
                subject,
                "the type hint does not say what the property holds",
            )
            return
        unsuitable = [
            held.__name__
            for held in property_range.element_types
            if not self._suits(context, declaration.meta.kind, held)
        ]
        if unsuitable:
            yield Violation(
                Check.RANGE_KIND_MISMATCH,
                Severity.ERROR,
                subject,
                f"a {declaration.meta.kind.value} cannot range over "
                + ", ".join(unsuitable),
            )

    @staticmethod
    def _suits(context: ValidationContext, kind: PropertyKind, held: type) -> bool:
        """
        Return whether a kind of property may range over a class.
        """
        if kind is PropertyKind.RELATION:
            return context.ontology.find_concept(held) is not None
        if kind is PropertyKind.ATTRIBUTE:
            return (
                held in PRIMITIVE_TYPES
                or context.ontology.find_value_partition(held) is not None
            )
        return held is ConceptReference

    @staticmethod
    def _range_dependencies(
        context: ValidationContext,
        declaration: PropertyDeclaration,
        property_range: PropertyRange,
        subject: str,
    ) -> Iterable[Violation]:
        """
        Check that every declaration the range uses is in a module the property may see.
        """
        allowed = context.allowed_modules(declaration.module)
        outside = [
            held.__name__
            for held in property_range.element_types
            if context.module_of(held) not in (None, *allowed)
        ]
        if outside:
            yield Violation(
                Check.RANGE_FROM_UNDECLARED_DEPENDENCY,
                Severity.ERROR,
                subject,
                f"{declaration.module} does not depend on the modules declaring "
                + ", ".join(outside),
            )

    @staticmethod
    def _cardinality(
        declaration: PropertyDeclaration, property_range: PropertyRange, subject: str
    ) -> Iterable[Violation]:
        """
        Check that the type hint agrees with the property's cardinality and default.
        """
        functional = declaration.meta.functional
        if functional != (not property_range.collection):
            yield Violation(
                Check.COLLECTION_MISMATCH,
                Severity.ERROR,
                subject,
                (
                    "declared single-valued but typed as a collection"
                    if functional
                    else "declared many-valued but typed as a single value"
                ),
            )
        default = next(
            f.default
            for f in fields(declaration.owner)
            if f.name == declaration.field_name
        )
        if functional and default is None and not property_range.optional:
            yield Violation(
                Check.MISSING_OPTIONAL,
                Severity.ERROR,
                subject,
                "starts empty but the type hint does not allow None",
            )

    @staticmethod
    def _inverse(
        context: ValidationContext,
        declaration: PropertyDeclaration,
        property_range: PropertyRange,
        subject: str,
    ) -> Iterable[Violation]:
        """
        Check that a declared inverse is a relation of every class the property ranges over.
        """
        inverse = declaration.meta.inverse_of
        if inverse is None:
            return
        for held in property_range.element_types:
            described = PropertyMeta.declared_on(held).get(inverse)
            if described is None or described.kind is not PropertyKind.RELATION:
                yield Violation(
                    Check.INVERSE_UNRESOLVED,
                    Severity.ERROR,
                    subject,
                    f"{held.__name__} has no relation '{inverse}'",
                )

    @staticmethod
    def _level(
        context: ValidationContext,
        declaration: PropertyDeclaration,
        property_range: PropertyRange,
        subject: str,
    ) -> Iterable[Violation]:
        """
        Check that knowledge about kinds does not refer to actual things.
        """
        owner = context.ontology.find_concept(declaration.owner)
        if owner is None or owner.level is not ConceptLevel.DESCRIPTION:
            return
        for held in property_range.element_types:
            target = context.ontology.find_concept(held)
            if target is not None and target.level is ConceptLevel.OCCURRENCE:
                yield Violation(
                    Check.DESCRIPTION_REFERS_TO_OCCURRENCE,
                    Severity.ERROR,
                    subject,
                    f"a description-level concept refers to the occurrence-level "
                    f"{held.__name__}",
                )

    @staticmethod
    def _assumption_marker(
        declaration: PropertyDeclaration, property_range: PropertyRange, subject: str
    ) -> Iterable[Violation]:
        """
        Check that a property marking an assumption holds a boolean.
        """
        if declaration.meta.marks_assumption and property_range.element_types != (
            bool,
        ):
            yield Violation(
                Check.ASSUMPTION_MARKER_NOT_BOOLEAN,
                Severity.ERROR,
                subject,
                "a property that marks an assumption must be a boolean",
            )


# %% property identity


class PropertyIdentityChecks(ValidationCheckGroup):
    """
    Checks that identifiers name one property each.
    """

    def check(self, context: ValidationContext) -> Iterable[Violation]:
        yield from self._shared_identifiers(context)
        yield from self._redeclarations(context)
        yield from self._exclusive_groups(context)

    def _shared_identifiers(self, context: ValidationContext) -> Iterable[Violation]:
        """
        Check that properties sharing an identifier agree.
        """
        by_iri: dict[str, list[PropertyDeclaration]] = defaultdict(list)
        for declaration in context.all_declarations():
            by_iri[declaration.iri].append(declaration)
        for iri, sharing in by_iri.items():
            if len(sharing) < 2 or not any(context.in_scope(d.module) for d in sharing):
                continue
            kinds = {d.meta.kind for d in sharing}
            functionalities = {d.meta.functional for d in sharing}
            if len(kinds) > 1 or len(functionalities) > 1:
                yield Violation(
                    Check.PROPERTY_IRI_CONFLICT,
                    Severity.ERROR,
                    iri,
                    "used with different kinds or cardinalities by "
                    + ", ".join(context.subject_of(d) for d in sharing),
                )
                continue
            if self._ranges_disagree(context, sharing):
                yield Violation(
                    Check.PROPERTY_IRI_CONFLICT,
                    Severity.WARNING,
                    iri,
                    "used with different ranges by "
                    + ", ".join(context.subject_of(d) for d in sharing),
                )

    @staticmethod
    def _ranges_disagree(
        context: ValidationContext, sharing: Sequence[PropertyDeclaration]
    ) -> bool:
        """
        Return whether properties sharing an identifier differ in range beyond narrowing.

        A concept may restrict the range that a concept it specialises gives the property.
        Declarations that restrict nothing must agree with each other.
        """
        types = {
            d: frozenset(context.range_of(d).element_types)
            for d in sharing
            if context.range_of(d).resolved
        }
        roots = [
            d
            for d in types
            if not any(
                other is not d and issubclass(d.owner, other.owner) for other in types
            )
        ]
        if len({types[d] for d in roots}) > 1:
            return True
        return any(
            not all(
                any(issubclass(held, wide) for wide in types[other])
                for held in types[d]
            )
            for d in types
            for other in types
            if other is not d and issubclass(d.owner, other.owner)
        )

    @staticmethod
    def _exclusive_groups(context: ValidationContext) -> Iterable[Violation]:
        """
        Check that each group of alternatives offers a choice.
        """
        for declaration in context.declarations():
            group = declaration.meta.exclusive_group
            if group is None:
                continue
            members = [
                name
                for name, meta in PropertyMeta.declared_on(declaration.owner).items()
                if meta.exclusive_group == group
            ]
            if len(members) < 2:
                yield Violation(
                    Check.EXCLUSIVE_GROUP_TOO_SMALL,
                    Severity.ERROR,
                    context.subject_of(declaration),
                    f"the group '{group}' has no other member to choose between",
                )

    @staticmethod
    def _redeclarations(context: ValidationContext) -> Iterable[Violation]:
        """
        Check that an inherited field written again keeps its identifier.
        """
        declared = {(d.owner, d.field_name): d for d in context.all_declarations()}
        for declaration in context.declarations():
            for ancestor in declaration.owner.__mro__[1:]:
                inherited = declared.get((ancestor, declaration.field_name))
                if inherited is not None and inherited.iri != declaration.iri:
                    yield Violation(
                        Check.PROPERTY_REDECLARED_UNDER_OTHER_IRI,
                        Severity.WARNING,
                        context.subject_of(declaration),
                        f"also declared by {ancestor.__name__} as {inherited.iri}",
                    )


# %% concept axioms


class ConceptAxiomChecks(ValidationCheckGroup):
    """
    Checks that what concepts state about their instances is coherent.
    """

    def check(self, context: ValidationContext) -> Iterable[Violation]:
        by_name: dict[str, list[ConceptMeta]] = defaultdict(list)
        for meta in context.ontology.concepts():
            by_name[meta.local_name].append(meta)
        for meta in context.concepts():
            yield from self._disjointness(meta, by_name)
            yield from self._intended_overlap(meta, by_name)
            yield from self._fixed_values(context, meta)
            yield from self._allowed_values(context, meta)

    @staticmethod
    def _disjointness(
        meta: ConceptMeta, by_name: dict[str, list[ConceptMeta]]
    ) -> Iterable[Violation]:
        """
        Check that each name a concept is disjoint from names one unrelated concept.
        """
        for name in meta.axioms.disjoint_with:
            matches = by_name.get(name, [])
            if not matches:
                yield Violation(
                    Check.DISJOINT_NAME_UNKNOWN,
                    Severity.ERROR,
                    meta.local_name,
                    f"no concept is named '{name}'",
                )
            elif len(matches) > 1:
                yield Violation(
                    Check.DISJOINT_NAME_AMBIGUOUS,
                    Severity.WARNING,
                    meta.local_name,
                    f"{len(matches)} concepts are named '{name}'",
                )
            elif matches[0].cls is meta.cls:
                yield Violation(
                    Check.DISJOINT_WITH_SELF,
                    Severity.ERROR,
                    meta.local_name,
                    "a concept cannot be disjoint from itself",
                )
            elif issubclass(meta.cls, matches[0].cls) or issubclass(
                matches[0].cls, meta.cls
            ):
                yield Violation(
                    Check.DISJOINT_RELATED_BY_INHERITANCE,
                    Severity.ERROR,
                    meta.local_name,
                    f"'{name}' is related by inheritance, so instances are shared",
                )

    @staticmethod
    def _intended_overlap(
        meta: ConceptMeta, by_name: dict[str, list[ConceptMeta]]
    ) -> Iterable[Violation]:
        """
        Check that each name a concept may overlap with names a concept, once.
        """
        for name in meta.axioms.may_overlap_with:
            if name not in by_name:
                yield Violation(
                    Check.OVERLAP_NAME_UNKNOWN,
                    Severity.ERROR,
                    meta.local_name,
                    f"no concept is named '{name}'",
                )
            elif name in meta.axioms.disjoint_with:
                yield Violation(
                    Check.OVERLAP_CONTRADICTS_DISJOINTNESS,
                    Severity.ERROR,
                    meta.local_name,
                    f"declared both disjoint from and overlapping with '{name}'",
                )

    @staticmethod
    def _fixed_values(
        context: ValidationContext, meta: ConceptMeta
    ) -> Iterable[Violation]:
        """
        Check that no fixed value contradicts one a concept inherits.
        """
        for fixed in meta.axioms.fixed_values:
            for ancestor in meta.cls.__mro__[1:]:
                described = context.ontology.find_concept(ancestor)
                inherited = next(
                    (
                        f
                        for f in (described.axioms.fixed_values if described else ())
                        if f.property_name == fixed.property_name
                    ),
                    None,
                )
                if inherited is not None and inherited.value != fixed.value:
                    yield Violation(
                        Check.FIXED_VALUE_CONFLICT,
                        Severity.ERROR,
                        meta.local_name,
                        f"fixes {fixed.property_name} to {fixed.value!r} but "
                        f"{ancestor.__name__} fixes it to {inherited.value!r}",
                    )
                    break

    @staticmethod
    def _allowed_values(
        context: ValidationContext, meta: ConceptMeta
    ) -> Iterable[Violation]:
        """
        Check that a concept only narrows the values its ancestors allow.
        """
        claimed = {
            allowed.property_name: allowed.values
            for allowed in meta.axioms.allowed_values
        } | {
            fixed.property_name: frozenset({fixed.value})
            for fixed in meta.axioms.fixed_values
        }
        for property_name, values in claimed.items():
            for ancestor in meta.cls.__mro__[1:]:
                described = context.ontology.find_concept(ancestor)
                inherited = (
                    None
                    if described is None
                    else described.axioms.allowed_values_of(property_name)
                )
                if inherited is not None and not values <= inherited:
                    yield Violation(
                        Check.RESTRICTION_WIDENS_INHERITED,
                        Severity.ERROR,
                        meta.local_name,
                        f"{property_name} allows values that {ancestor.__name__} "
                        f"does not: {sorted(map(str, values - inherited))}",
                    )
                    break


# %% taxonomy


class ConceptTaxonomyChecks(ValidationCheckGroup):
    """
    Checks the shape of the hierarchy of concepts.
    """

    def check(self, context: ValidationContext) -> Iterable[Violation]:
        concepts = context.ontology.concepts()
        for parent in concepts:
            yield from self._sibling_overlaps(context, parent, concepts)
        for meta in context.concepts():
            if meta.abstract and not any(
                other.cls is not meta.cls and issubclass(other.cls, meta.cls)
                for other in concepts
            ):
                yield Violation(
                    Check.ABSTRACT_WITHOUT_SUBCONCEPT,
                    Severity.WARNING,
                    meta.local_name,
                    "nothing specialises it, so it can never appear in a hypothesis",
                )

    @staticmethod
    def _sibling_overlaps(
        context: ValidationContext,
        parent: ConceptMeta,
        concepts: Sequence[ConceptMeta],
    ) -> Iterable[Violation]:
        """
        Report rigid subconcepts of a parent that may share instances without saying so.
        """
        if parent.axioms.disjoint_siblings:
            return
        children = [
            m
            for m in concepts
            if parent.cls in m.cls.__bases__ and m.rigidity is Rigidity.RIGID
        ]
        for position, first in enumerate(children):
            for second in children[position + 1 :]:
                relation_stated = (
                    second.local_name in first.axioms.disjoint_with
                    or first.local_name in second.axioms.disjoint_with
                    or second.local_name in first.axioms.may_overlap_with
                    or first.local_name in second.axioms.may_overlap_with
                )
                shared_child = any(
                    other.cls not in (first.cls, second.cls)
                    and issubclass(other.cls, first.cls)
                    and issubclass(other.cls, second.cls)
                    for other in concepts
                )
                if (
                    context.in_scope(first.module)
                    and not relation_stated
                    and not shared_child
                ):
                    yield Violation(
                        Check.SIBLINGS_OVERLAP,
                        Severity.WARNING,
                        first.local_name,
                        f"may share instances with {second.local_name}, and neither "
                        f"is declared disjoint from the other",
                    )


# %% documentation


class DocumentationChecks(ValidationCheckGroup):
    """
    Checks that everything the schema and prompt read is documented.
    """

    def check(self, context: ValidationContext) -> Iterable[Violation]:
        for meta in context.concepts():
            if context.reader.read_class(meta.cls) is None:
                yield Violation(
                    Check.UNDOCUMENTED_CONCEPT,
                    Severity.WARNING,
                    meta.local_name,
                    "the class has no docstring",
                )
        for declaration in context.declarations():
            if (
                context.reader.read_attribute(declaration.owner, declaration.field_name)
                is None
            ):
                yield Violation(
                    Check.UNDOCUMENTED_PROPERTY,
                    Severity.WARNING,
                    context.subject_of(declaration),
                    "no documentation is written under the field",
                )
        for partition in context.ontology.value_partitions(context.module):
            for member in context.reader.find_undocumented_members(
                partition.enumeration
            ):
                yield Violation(
                    Check.UNDOCUMENTED_MEMBER,
                    Severity.WARNING,
                    f"{partition.enumeration.__name__}.{member}",
                    "no documentation is written under the member",
                )


# %% the validator


def default_check_groups() -> tuple[ValidationCheckGroup, ...]:
    """
    Return the check groups every validation runs.
    """
    return (
        PropertyRangeChecks(),
        PropertyIdentityChecks(),
        ConceptAxiomChecks(),
        ConceptTaxonomyChecks(),
        DocumentationChecks(),
    )


@dataclass
class OntologyValidator:
    """
    Checks an ontology's declarations against the rules that need to see several of them.
    """

    ontology: Ontology
    """
    The ontology to check.
    """

    checks: Sequence[ValidationCheckGroup] = field(default_factory=default_check_groups)
    """
    The check groups to run.
    """

    reader: DocumentationReader = field(default_factory=DocumentationReader)
    """
    Reads the documentation written in the source.
    """

    resolver: RangeResolver = field(default_factory=RangeResolver)
    """
    Reads what property type hints say.
    """

    def validate(self, module: Optional[str] = None) -> ValidationReport:
        """
        Run every check group and collect what they find.

        :param module: Report only the declarations of this module. Declarations of the
            others are still read where a check needs to compare against them.
        """
        context = ValidationContext(
            ontology=self.ontology,
            module=module,
            reader=self.reader,
            resolver=self.resolver,
        )
        return ValidationReport(
            violations=tuple(
                violation for group in self.checks for violation in group.check(context)
            )
        )
