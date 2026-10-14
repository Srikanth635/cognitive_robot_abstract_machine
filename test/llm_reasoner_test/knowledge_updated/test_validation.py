from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import pytest
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.ontology import (
    Ontology,
    OntologyModule,
)
from llm_reasoner.knowledge_updated.ontological.properties import (
    attribute,
    reference,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
    ValidationReport,
    Violation,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    ConceptLevel,
    Severity,
)

# %% shared helpers

CORE = "http://example.org/validation/core#"
EXTRA = "http://example.org/validation/extra#"
SHARED_IRI = "urn:test:shared"


def new_ontology() -> tuple[Ontology, OntologyModule]:
    """
    Return an empty ontology with one module.
    """
    ontology = Ontology(name="validation")
    return ontology, ontology.module("core", version="1.0.0", namespace=CORE)


def validate(ontology: Ontology, module: Optional[str] = None) -> ValidationReport:
    """
    Validate an ontology, or one module of it.
    """
    return OntologyValidator(ontology).validate(module)


def subjects(report: ValidationReport, check: Check) -> set[str]:
    """
    Return the subjects a check reported.
    """
    return {violation.subject for violation in report.of(check)}


def severities(report: ValidationReport, check: Check) -> set[Severity]:
    """
    Return the severities a check reported.
    """
    return {violation.severity for violation in report.of(check)}


# %% a consistent ontology


class Direction(StrEnum):
    """
    A way to move.
    """

    UP = "up"
    """
    Towards the ceiling.
    """

    DOWN = "down"
    """
    Towards the floor.
    """


@dataclass(eq=False)
class Widget(KnowledgeEntity):
    """
    A part of a machine.
    """

    size: Optional[float] = attribute(functional=True)
    """
    How large the widget is.
    """


@dataclass(eq=False)
class Gear(Widget):
    """
    A widget with teeth.
    """

    teeth: int = attribute(functional=True, default=0)
    """
    How many teeth it has.
    """

    machine: Optional[Machine] = relation(functional=True)
    """
    The machine it turns in.
    """


@dataclass(eq=False)
class Machine(KnowledgeEntity):
    """
    A machine built from gears.
    """

    direction: Optional[Direction] = attribute(functional=True)
    """
    The way it runs.
    """

    gears: list[Gear] = relation(minimum_count=1, inverse_of="machine")
    """
    The gears it turns.
    """

    kind: Optional[ConceptReference] = reference(functional=True)
    """
    The term naming what kind of machine it is.
    """


class TestConsistentOntology:
    """
    An ontology that follows every rule has nothing to report.
    """

    def test_consistent_documented_ontology_has_no_violations(self) -> None:
        ontology, core = new_ontology()
        core.value_partition()(Direction)
        core.concept()(Widget)
        core.concept()(Gear)
        core.concept()(Machine)

        report = validate(ontology)

        assert report.violations == ()
        assert report.clean


# %% reports


class TestValidationReport:
    """
    A report separates errors from warnings and can be filtered by check.
    """

    ERROR = Violation(
        check=Check.RANGE_KIND_MISMATCH,
        severity=Severity.ERROR,
        subject="A.b",
        message="an error",
    )
    WARNING = Violation(
        check=Check.UNDOCUMENTED_CONCEPT,
        severity=Severity.WARNING,
        subject="A",
        message="a warning",
    )

    def test_errors_and_warnings_are_separated(self) -> None:
        report = ValidationReport(violations=(self.ERROR, self.WARNING))

        assert report.errors == (self.ERROR,)
        assert report.warnings == (self.WARNING,)

    def test_report_with_only_warnings_is_clean(self) -> None:
        assert ValidationReport(violations=(self.WARNING,)).clean

    def test_report_with_an_error_is_not_clean(self) -> None:
        assert not ValidationReport(violations=(self.ERROR, self.WARNING)).clean

    def test_violations_can_be_selected_by_check(self) -> None:
        report = ValidationReport(violations=(self.ERROR, self.WARNING))

        assert report.of(Check.UNDOCUMENTED_CONCEPT) == (self.WARNING,)
        assert report.of(Check.MISSING_OPTIONAL) == ()


# %% ranges


class Orphan(KnowledgeEntity):
    """
    An entity that no ontology declares.
    """


class Silent(StrEnum):
    """
    An enumeration with an undocumented member.
    """

    QUIET = "quiet"


@dataclass(eq=False)
class BadRanges(KnowledgeEntity):
    """
    Fields whose ranges do not suit their kind of property.
    """

    to_orphan: Optional[Orphan] = relation(functional=True)
    """
    Relates to a class the ontology does not declare.
    """

    to_partition: Optional[Direction] = relation(functional=True)
    """
    Relates to an enumeration.
    """

    undeclared_enum: Optional[Silent] = attribute(functional=True)
    """
    Holds an enumeration that is not a declared value partition.
    """

    entity_as_attribute: Optional[Widget] = attribute(functional=True)
    """
    Holds an entity as a plain value.
    """

    int_as_reference: Optional[int] = reference(functional=True)
    """
    Names a term but is typed as an integer.
    """

    unspecified: list = relation()
    """
    Relates to instances of nothing in particular.
    """

    fine: Optional[Widget] = relation(functional=True)
    """
    A correct relation.
    """


class TestRangeKind:
    """
    A property's range must suit what its kind of property holds.
    """

    MISMATCHED = {
        "to_orphan",
        "to_partition",
        "undeclared_enum",
        "entity_as_attribute",
        "int_as_reference",
        "unspecified",
    }

    def test_exactly_the_ill_suited_ranges_are_reported(self) -> None:
        ontology, core = new_ontology()
        core.value_partition()(Direction)
        core.concept()(Widget)
        core.concept()(BadRanges)

        report = validate(ontology)

        assert subjects(report, Check.RANGE_KIND_MISMATCH) == {
            f"BadRanges.{name}" for name in self.MISMATCHED
        }
        assert severities(report, Check.RANGE_KIND_MISMATCH) == {Severity.ERROR}


@dataclass(eq=False)
class Crane(KnowledgeEntity):
    """
    A machine that lifts widgets.
    """

    load: Optional[Widget] = relation(functional=True)
    """
    The widget it is lifting.
    """


class TestRangeDependencies:
    """
    A property may range only over declarations of its module or its dependencies.
    """

    def test_range_from_a_module_that_is_not_a_dependency_is_an_error(self) -> None:
        ontology, core = new_ontology()
        extra = ontology.module("extra", version="1.0.0", namespace=EXTRA)
        core.concept()(Widget)
        extra.concept()(Crane)

        report = validate(ontology)

        assert subjects(report, Check.RANGE_FROM_UNDECLARED_DEPENDENCY) == {
            "Crane.load"
        }
        assert severities(report, Check.RANGE_FROM_UNDECLARED_DEPENDENCY) == {
            Severity.ERROR
        }

    def test_range_from_a_dependency_is_accepted(self) -> None:
        ontology, core = new_ontology()
        extra = ontology.module(
            "extra", version="1.0.0", namespace=EXTRA, depends_on=(core,)
        )
        core.concept()(Widget)
        extra.concept()(Crane)

        assert validate(ontology).of(Check.RANGE_FROM_UNDECLARED_DEPENDENCY) == ()


# %% cardinality


@dataclass(eq=False)
class Cardinality(KnowledgeEntity):
    """
    Fields whose hints disagree with their declared cardinality.
    """

    listed_but_single: list[Widget] = relation(functional=True)
    """
    Declared single-valued but typed as a list.
    """

    single_but_many: Optional[Widget] = relation()
    """
    Declared many-valued but typed as a single value.
    """

    fine_single: Optional[Widget] = relation(functional=True)
    """
    A correct single-valued field.
    """

    fine_many: list[Widget] = relation()
    """
    A correct many-valued field.
    """


@dataclass(eq=False)
class Presence(KnowledgeEntity):
    """
    Fields whose hints do or do not allow their empty default.
    """

    bare_single: Widget = relation(functional=True)
    """
    Starts empty but is typed as always present.
    """

    defaulted: int = attribute(functional=True, default=3)
    """
    Starts at a value, so it needs no optional hint.
    """

    optional_single: Optional[Widget] = relation(functional=True)
    """
    Starts empty and is typed as optional.
    """


class TestCardinality:
    """
    A property's hint must agree with its cardinality and its default.
    """

    def test_hint_that_disagrees_with_functionality_is_an_error(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Widget)
        core.concept()(Cardinality)

        report = validate(ontology)

        assert subjects(report, Check.COLLECTION_MISMATCH) == {
            "Cardinality.listed_but_single",
            "Cardinality.single_but_many",
        }
        assert severities(report, Check.COLLECTION_MISMATCH) == {Severity.ERROR}

    def test_empty_default_needs_an_optional_hint(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Widget)
        core.concept()(Presence)

        report = validate(ontology)

        assert subjects(report, Check.MISSING_OPTIONAL) == {"Presence.bare_single"}
        assert severities(report, Check.MISSING_OPTIONAL) == {Severity.ERROR}


# %% assumption markers


@dataclass(eq=False)
class Presumption(KnowledgeEntity):
    """
    Properties that mark their fragment as an assumption.
    """

    presupposed: bool = attribute(functional=True, default=False, marks_assumption=True)
    """
    A correct marker.
    """

    presupposed_text: Optional[str] = attribute(functional=True, marks_assumption=True)
    """
    A marker that is not a boolean.
    """


class TestAssumptionMarkers:
    """
    A property that marks an assumption must hold a boolean.
    """

    def test_marker_that_is_not_a_boolean_is_an_error(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Presumption)

        report = validate(ontology)

        assert subjects(report, Check.ASSUMPTION_MARKER_NOT_BOOLEAN) == {
            "Presumption.presupposed_text"
        }
        assert severities(report, Check.ASSUMPTION_MARKER_NOT_BOOLEAN) == {
            Severity.ERROR
        }


# %% exclusive groups


@dataclass(eq=False)
class Choice(KnowledgeEntity):
    """
    Two fields that are alternatives.
    """

    by_name: Optional[str] = attribute(
        functional=True, exclusive_group="identification"
    )
    """
    Identifies by name.
    """

    by_number: Optional[int] = attribute(
        functional=True, exclusive_group="identification"
    )
    """
    Identifies by number.
    """


@dataclass(eq=False)
class Lonely(KnowledgeEntity):
    """
    A group with only one member.
    """

    only: Optional[str] = attribute(functional=True, exclusive_group="alone")
    """
    The sole member of its group.
    """


class TestExclusiveGroups:
    """
    A group of alternatives needs at least two members.
    """

    def test_group_with_one_member_is_an_error(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Choice)
        core.concept()(Lonely)

        report = validate(ontology)

        assert subjects(report, Check.EXCLUSIVE_GROUP_TOO_SMALL) == {"Lonely.only"}
        assert severities(report, Check.EXCLUSIVE_GROUP_TOO_SMALL) == {Severity.ERROR}


# %% inverses


@dataclass(eq=False)
class Line(KnowledgeEntity):
    """
    An entry of a ledger that does not know its ledger.
    """


@dataclass(eq=False)
class Ledger(KnowledgeEntity):
    """
    A ledger whose inverse names a field its lines lack.
    """

    lines: list[Line] = relation(inverse_of="owner")
    """
    The lines it holds.
    """


@dataclass(eq=False)
class Tally(KnowledgeEntity):
    """
    A tally whose inverse names an attribute.
    """

    widgets: list[Widget] = relation(inverse_of="size")
    """
    The widgets it counts.
    """


class TestInverses:
    """
    An inverse must name a relation on the class the property ranges over.
    """

    def test_unresolvable_inverses_are_errors(self) -> None:
        ontology, core = new_ontology()
        core.value_partition()(Direction)
        core.concept()(Widget)
        core.concept()(Gear)
        core.concept()(Machine)
        core.concept()(Line)
        core.concept()(Ledger)
        core.concept()(Tally)

        report = validate(ontology)

        assert subjects(report, Check.INVERSE_UNRESOLVED) == {
            "Ledger.lines",
            "Tally.widgets",
        }
        assert severities(report, Check.INVERSE_UNRESOLVED) == {Severity.ERROR}


# %% properties sharing an identifier


@dataclass(eq=False)
class SharedFloat(KnowledgeEntity):
    """
    Holds a number under a shared identifier.
    """

    value: Optional[float] = attribute(functional=True, iri=SHARED_IRI)
    """
    The number.
    """


@dataclass(eq=False)
class SharedFloatAgain(KnowledgeEntity):
    """
    Holds a number under the same shared identifier.
    """

    value: Optional[float] = attribute(functional=True, iri=SHARED_IRI)
    """
    The number.
    """


@dataclass(eq=False)
class SharedText(KnowledgeEntity):
    """
    Holds text under the shared identifier.
    """

    value: Optional[str] = attribute(functional=True, iri=SHARED_IRI)
    """
    The text.
    """


@dataclass(eq=False)
class SharedList(KnowledgeEntity):
    """
    Holds many numbers under the shared identifier.
    """

    value: list[float] = attribute(iri=SHARED_IRI)
    """
    The numbers.
    """


@dataclass(eq=False)
class SharedLink(KnowledgeEntity):
    """
    Relates to a widget under the shared identifier.
    """

    value: Optional[Widget] = relation(functional=True, iri=SHARED_IRI)
    """
    The widget.
    """


NARROWED_IRI = "http://example.org/validation#narrowed"


@dataclass(eq=False)
class SpecialWidget(Widget):
    """
    A widget of a narrower kind.
    """


@dataclass(eq=False)
class AnyWidgetHolder(KnowledgeEntity):
    """
    Relates to any widget.
    """

    held: Optional[Widget] = relation(functional=True, iri=NARROWED_IRI)
    """
    The widget.
    """


@dataclass(eq=False)
class SpecialWidgetHolder(AnyWidgetHolder):
    """
    Restricts the widget it relates to, keeping the identifier.
    """

    held: Optional[SpecialWidget] = relation(functional=True, iri=NARROWED_IRI)
    """
    The special widget.
    """


@dataclass(eq=False)
class SpecialWidgetOwner(KnowledgeEntity):
    """
    Relates to a special widget.
    """

    held: Optional[SpecialWidget] = relation(functional=True, iri=NARROWED_IRI)
    """
    The special widget.
    """


@dataclass(eq=False)
class WidenedWidgetOwner(SpecialWidgetOwner):
    """
    Widens the widget it relates to, keeping the identifier.
    """

    held: Optional[Widget] = relation(functional=True, iri=NARROWED_IRI)
    """
    The widget.
    """


def shared_ontology(*classes: type) -> Ontology:
    """
    Return an ontology declaring the widget and the given classes.
    """
    ontology, core = new_ontology()
    core.concept()(Widget)
    for cls in classes:
        core.concept()(cls)
    return ontology


class TestPropertiesSharingAnIdentifier:
    """
    One identifier names one property, however many concepts use it.
    """

    def test_agreeing_uses_are_accepted(self) -> None:
        report = validate(shared_ontology(SharedFloat, SharedFloatAgain))

        assert report.of(Check.PROPERTY_IRI_CONFLICT) == ()

    def test_different_ranges_are_a_warning(self) -> None:
        report = validate(shared_ontology(SharedFloat, SharedText))

        assert subjects(report, Check.PROPERTY_IRI_CONFLICT) == {SHARED_IRI}
        assert severities(report, Check.PROPERTY_IRI_CONFLICT) == {Severity.WARNING}

    def test_a_subconcept_narrowing_the_range_of_its_parent_is_accepted(self) -> None:
        report = validate(
            shared_ontology(SpecialWidget, AnyWidgetHolder, SpecialWidgetHolder)
        )

        assert report.of(Check.PROPERTY_IRI_CONFLICT) == ()

    def test_a_subconcept_widening_the_range_of_its_parent_is_a_warning(self) -> None:
        report = validate(
            shared_ontology(SpecialWidget, SpecialWidgetOwner, WidenedWidgetOwner)
        )

        assert subjects(report, Check.PROPERTY_IRI_CONFLICT) == {NARROWED_IRI}
        assert severities(report, Check.PROPERTY_IRI_CONFLICT) == {Severity.WARNING}

    def test_unrelated_concepts_with_narrower_and_wider_ranges_still_conflict(
        self,
    ) -> None:
        report = validate(
            shared_ontology(SpecialWidget, AnyWidgetHolder, SpecialWidgetOwner)
        )

        assert subjects(report, Check.PROPERTY_IRI_CONFLICT) == {NARROWED_IRI}

    def test_different_kinds_are_an_error(self) -> None:
        report = validate(shared_ontology(SharedFloat, SharedLink))

        assert subjects(report, Check.PROPERTY_IRI_CONFLICT) == {SHARED_IRI}
        assert severities(report, Check.PROPERTY_IRI_CONFLICT) == {Severity.ERROR}

    def test_different_functionality_is_an_error(self) -> None:
        report = validate(shared_ontology(SharedFloat, SharedList))

        assert subjects(report, Check.PROPERTY_IRI_CONFLICT) == {SHARED_IRI}
        assert severities(report, Check.PROPERTY_IRI_CONFLICT) == {Severity.ERROR}


@dataclass(eq=False)
class BaseAmount(KnowledgeEntity):
    """
    A concept that introduces an amount.
    """

    amount: Optional[float] = attribute(functional=True)
    """
    How much there is.
    """


@dataclass(eq=False)
class DerivedAmount(BaseAmount):
    """
    A concept that writes the inherited amount again.
    """

    amount: Optional[float] = attribute(functional=True)
    """
    How much there is, on this concept.
    """


class TestRedeclaredProperties:
    """
    Writing an inherited field again should not give it a second identifier.
    """

    def test_redeclaration_in_the_same_module_shares_the_identifier(self) -> None:
        ontology, core = new_ontology()
        core.concept()(BaseAmount)
        core.concept()(DerivedAmount)

        assert validate(ontology).of(Check.PROPERTY_REDECLARED_UNDER_OTHER_IRI) == ()

    def test_redeclaration_in_another_module_is_a_warning(self) -> None:
        ontology, core = new_ontology()
        extra = ontology.module(
            "extra", version="1.0.0", namespace=EXTRA, depends_on=(core,)
        )
        core.concept()(BaseAmount)
        extra.concept()(DerivedAmount)

        report = validate(ontology)

        assert subjects(report, Check.PROPERTY_REDECLARED_UNDER_OTHER_IRI) == {
            "DerivedAmount.amount"
        }
        assert severities(report, Check.PROPERTY_REDECLARED_UNDER_OTHER_IRI) == {
            Severity.WARNING
        }


# %% disjointness


@dataclass(eq=False)
class Alpha(KnowledgeEntity):
    """
    A concept other concepts are disjoint from.
    """


@dataclass(eq=False)
class Beta(Alpha):
    """
    A specialisation of the first concept.
    """


@dataclass(eq=False)
class Gamma(KnowledgeEntity):
    """
    A concept unrelated to the others.
    """


class First:
    """
    Holds a class that shares its name with one held by another class.
    """

    @dataclass(eq=False)
    class Twin(KnowledgeEntity):
        """
        The first of two concepts with one name.
        """


class Second:
    """
    Holds a class that shares its name with one held by another class.
    """

    @dataclass(eq=False)
    class Twin(KnowledgeEntity):
        """
        The second of two concepts with one name.
        """


class TestDisjointness:
    """
    Disjointness must name distinct, unrelated concepts.
    """

    def declared(self, disjoint_class: type, disjoint_with: list[str]) -> Ontology:
        ontology, core = new_ontology()
        core.concept()(Alpha)
        core.concept(disjoint_with=disjoint_with)(disjoint_class)
        return ontology

    def test_name_no_concept_has_is_an_error(self) -> None:
        report = validate(self.declared(Gamma, ["Nowhere"]))

        assert subjects(report, Check.DISJOINT_NAME_UNKNOWN) == {"Gamma"}
        assert severities(report, Check.DISJOINT_NAME_UNKNOWN) == {Severity.ERROR}

    def test_concept_cannot_be_disjoint_from_itself(self) -> None:
        report = validate(self.declared(Gamma, ["Gamma"]))

        assert subjects(report, Check.DISJOINT_WITH_SELF) == {"Gamma"}
        assert severities(report, Check.DISJOINT_WITH_SELF) == {Severity.ERROR}

    def test_concept_cannot_be_disjoint_from_its_ancestor(self) -> None:
        report = validate(self.declared(Beta, ["Alpha"]))

        assert subjects(report, Check.DISJOINT_RELATED_BY_INHERITANCE) == {"Beta"}
        assert severities(report, Check.DISJOINT_RELATED_BY_INHERITANCE) == {
            Severity.ERROR
        }

    def test_name_shared_by_two_concepts_is_a_warning(self) -> None:
        ontology, core = new_ontology()
        extra = ontology.module("extra", version="1.0.0", namespace=EXTRA)
        extra.concept()(First.Twin)
        core.concept()(Second.Twin)
        core.concept(disjoint_with=["Twin"])(Gamma)

        report = validate(ontology)

        assert subjects(report, Check.DISJOINT_NAME_AMBIGUOUS) == {"Gamma"}
        assert severities(report, Check.DISJOINT_NAME_AMBIGUOUS) == {Severity.WARNING}

    def test_unrelated_named_concept_is_accepted(self) -> None:
        report = validate(self.declared(Gamma, ["Alpha"]))

        for check in (
            Check.DISJOINT_NAME_UNKNOWN,
            Check.DISJOINT_NAME_AMBIGUOUS,
            Check.DISJOINT_WITH_SELF,
            Check.DISJOINT_RELATED_BY_INHERITANCE,
        ):
            assert report.of(check) == ()


class TestIntendedOverlapNames:
    """
    An intended overlap must name a concept, and must not contradict a disjointness.
    """

    def test_name_no_concept_has_is_an_error(self) -> None:
        ontology, core = new_ontology()
        core.concept(may_overlap_with=["Nowhere"])(Gamma)

        report = validate(ontology)

        assert subjects(report, Check.OVERLAP_NAME_UNKNOWN) == {"Gamma"}
        assert severities(report, Check.OVERLAP_NAME_UNKNOWN) == {Severity.ERROR}

    def test_concept_cannot_be_both_disjoint_from_and_overlapping_another(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Alpha)
        core.concept(disjoint_with=["Alpha"], may_overlap_with=["Alpha"])(Gamma)

        report = validate(ontology)

        assert subjects(report, Check.OVERLAP_CONTRADICTS_DISJOINTNESS) == {"Gamma"}
        assert severities(report, Check.OVERLAP_CONTRADICTS_DISJOINTNESS) == {
            Severity.ERROR
        }

    def test_overlap_with_an_existing_concept_is_accepted(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Alpha)
        core.concept(may_overlap_with=["Alpha"])(Gamma)

        report = validate(ontology)

        assert report.of(Check.OVERLAP_NAME_UNKNOWN) == ()
        assert report.of(Check.OVERLAP_CONTRADICTS_DISJOINTNESS) == ()


# %% fixed values


class Gait(StrEnum):
    """
    How a mover walks.
    """

    WALK = "walk"
    """
    One foot always on the ground.
    """

    RUN = "run"
    """
    Both feet leave the ground.
    """


@dataclass(eq=False)
class Moving(KnowledgeEntity):
    """
    Something that moves in a gait.
    """

    gait: Optional[Gait] = attribute(functional=True)
    """
    The way it moves.
    """


@dataclass(eq=False)
class Walker(Moving):
    """
    Something that always walks.
    """


@dataclass(eq=False)
class Runner(Walker):
    """
    Something that runs although its parent always walks.
    """


@dataclass(eq=False)
class Strider(Walker):
    """
    Something that walks, as its parent does.
    """


class TestFixedValues:
    """
    A concept cannot fix a value differently from the concepts it specialises.
    """

    def declared(self, descendant: type, value: Gait) -> Ontology:
        ontology, core = new_ontology()
        core.value_partition()(Gait)
        core.concept()(Moving)
        core.concept(fixed_values={"gait": Gait.WALK})(Walker)
        core.concept(fixed_values={"gait": value})(descendant)
        return ontology

    def test_contradicting_an_inherited_value_is_an_error(self) -> None:
        report = validate(self.declared(Runner, Gait.RUN))

        assert subjects(report, Check.FIXED_VALUE_CONFLICT) == {"Runner"}
        assert severities(report, Check.FIXED_VALUE_CONFLICT) == {Severity.ERROR}

    def test_repeating_an_inherited_value_is_accepted(self) -> None:
        report = validate(self.declared(Strider, Gait.WALK))

        assert report.of(Check.FIXED_VALUE_CONFLICT) == ()


@dataclass(eq=False)
class Sprinter(Walker):
    """
    Something that moves fast although its parent may only walk.
    """


class TestAllowedValuesAcrossInheritance:
    """
    A concept may narrow the values an ancestor allows, never widen them.
    """

    def declared(self, descendant_allowed: list[Gait]) -> Ontology:
        ontology, core = new_ontology()
        core.value_partition()(Gait)
        core.concept()(Moving)
        core.concept(allowed_values={"gait": [Gait.WALK]})(Walker)
        core.concept(allowed_values={"gait": descendant_allowed})(Sprinter)
        return ontology

    def test_allowing_a_value_the_ancestor_forbids_is_an_error(self) -> None:
        report = validate(self.declared([Gait.WALK, Gait.RUN]))

        assert subjects(report, Check.RESTRICTION_WIDENS_INHERITED) == {"Sprinter"}
        assert severities(report, Check.RESTRICTION_WIDENS_INHERITED) == {
            Severity.ERROR
        }

    def test_narrowing_or_repeating_the_inherited_values_is_accepted(self) -> None:
        report = validate(self.declared([Gait.WALK]))

        assert report.of(Check.RESTRICTION_WIDENS_INHERITED) == ()

    def test_fixing_a_value_the_ancestor_forbids_is_an_error(self) -> None:
        ontology, core = new_ontology()
        core.value_partition()(Gait)
        core.concept()(Moving)
        core.concept(allowed_values={"gait": [Gait.WALK]})(Walker)
        core.concept(fixed_values={"gait": Gait.RUN})(Sprinter)

        report = validate(ontology)

        assert subjects(report, Check.RESTRICTION_WIDENS_INHERITED) == {"Sprinter"}


# %% siblings and abstract concepts


@dataclass(eq=False)
class Shape(KnowledgeEntity):
    """
    A shape.
    """


@dataclass(eq=False)
class Circle(Shape):
    """
    A round shape.
    """


@dataclass(eq=False)
class Square(Shape):
    """
    A four-sided shape.
    """


@dataclass(eq=False)
class SquareCircle(Circle, Square):
    """
    A shape that is both, which makes the two overlap.
    """


@dataclass(eq=False)
class Stub(KnowledgeEntity):
    """
    An abstract concept nothing specialises.
    """


@dataclass(eq=False)
class Stem(KnowledgeEntity):
    """
    An abstract concept with a specialisation.
    """


@dataclass(eq=False)
class Leaf(Stem):
    """
    A specialisation of the abstract concept.
    """


class TestSiblings:
    """
    Sibling concepts should say whether they can share instances.
    """

    def declared(
        self,
        parent_siblings: bool = False,
        circle_disjoint_with: tuple[str, ...] = (),
        with_common_child: bool = False,
    ) -> Ontology:
        ontology, core = new_ontology()
        core.concept(disjoint_siblings=parent_siblings)(Shape)
        core.concept(disjoint_with=circle_disjoint_with)(Circle)
        core.concept()(Square)
        if with_common_child:
            core.concept()(SquareCircle)
        return ontology

    def test_siblings_declared_nothing_about_are_a_warning(self) -> None:
        report = validate(self.declared())

        assert subjects(report, Check.SIBLINGS_OVERLAP) == {"Circle"}
        assert severities(report, Check.SIBLINGS_OVERLAP) == {Severity.WARNING}

    def test_parent_declaring_its_children_disjoint_settles_it(self) -> None:
        assert (
            validate(self.declared(parent_siblings=True)).of(Check.SIBLINGS_OVERLAP)
            == ()
        )

    def test_one_sibling_declaring_the_other_disjoint_settles_it(self) -> None:
        assert (
            validate(self.declared(circle_disjoint_with=("Square",))).of(
                Check.SIBLINGS_OVERLAP
            )
            == ()
        )

    def test_one_sibling_declaring_the_overlap_settles_it(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Shape)
        core.concept(may_overlap_with=["Square"])(Circle)
        core.concept()(Square)

        assert validate(ontology).of(Check.SIBLINGS_OVERLAP) == ()

    def test_siblings_with_a_common_subconcept_overlap_by_construction(self) -> None:
        assert (
            validate(self.declared(with_common_child=True)).of(Check.SIBLINGS_OVERLAP)
            == ()
        )


class TestAbstractConcepts:
    """
    An abstract concept only makes sense when something specialises it.
    """

    def test_abstract_concept_nothing_specialises_is_a_warning(self) -> None:
        ontology, core = new_ontology()
        core.concept(abstract=True)(Stub)

        report = validate(ontology)

        assert subjects(report, Check.ABSTRACT_WITHOUT_SUBCONCEPT) == {"Stub"}
        assert severities(report, Check.ABSTRACT_WITHOUT_SUBCONCEPT) == {
            Severity.WARNING
        }

    def test_abstract_concept_with_a_specialisation_is_accepted(self) -> None:
        ontology, core = new_ontology()
        core.concept(abstract=True)(Stem)
        core.concept()(Leaf)

        assert validate(ontology).of(Check.ABSTRACT_WITHOUT_SUBCONCEPT) == ()


# %% levels


@dataclass(eq=False)
class Episode(KnowledgeEntity):
    """
    Something that actually happened.
    """


@dataclass(eq=False)
class Planner(KnowledgeEntity):
    """
    A generic concept that wrongly points at an actual episode.
    """

    episode: Optional[Episode] = relation(functional=True)
    """
    The episode it refers to.
    """


class TestLevels:
    """
    Knowledge about kinds must not refer to actual things.
    """

    def test_description_referring_to_an_occurrence_is_an_error(self) -> None:
        ontology, core = new_ontology()
        core.concept(level=ConceptLevel.OCCURRENCE)(Episode)
        core.concept()(Planner)

        report = validate(ontology)

        assert subjects(report, Check.DESCRIPTION_REFERS_TO_OCCURRENCE) == {
            "Planner.episode"
        }
        assert severities(report, Check.DESCRIPTION_REFERS_TO_OCCURRENCE) == {
            Severity.ERROR
        }

    def test_description_referring_to_a_description_is_accepted(self) -> None:
        ontology, core = new_ontology()
        core.concept()(Episode)
        core.concept()(Planner)

        assert validate(ontology).of(Check.DESCRIPTION_REFERS_TO_OCCURRENCE) == ()


# %% unresolvable annotations


@dataclass
class Dangling:
    """
    An occurrence-level class, not an entity, whose annotation names something undefined.

    It cannot be a symbol: an undefined name in a symbol's annotation would break the
    symbol graph every test shares.
    """

    target: Optional[NotDefinedAnywhere] = relation(functional=True)
    """
    Relates to something that does not exist.
    """


class TestUnresolvableAnnotations:
    """
    An annotation naming something undefined is reported, and its field is not checked
    further.
    """

    def test_undefined_name_is_an_error(self) -> None:
        ontology, core = new_ontology()
        core.concept(level=ConceptLevel.OCCURRENCE)(Dangling)

        report = validate(ontology)

        assert subjects(report, Check.UNRESOLVABLE_ANNOTATION) == {"Dangling.target"}
        assert severities(report, Check.UNRESOLVABLE_ANNOTATION) == {Severity.ERROR}

    def test_unresolvable_field_is_not_checked_for_anything_else(self) -> None:
        ontology, core = new_ontology()
        core.concept(level=ConceptLevel.OCCURRENCE)(Dangling)

        report = validate(ontology)

        assert {violation.check for violation in report.errors} == {
            Check.UNRESOLVABLE_ANNOTATION
        }


# %% documentation


@dataclass(eq=False)
class Bare(KnowledgeEntity):
    value: Optional[float] = attribute(functional=True)


class TestDocumentation:
    """
    Concepts, properties and partition members should be documented, since the schema and
    the prompt read that documentation.
    """

    def test_undocumented_declarations_are_warnings(self) -> None:
        ontology, core = new_ontology()
        core.value_partition()(Silent)
        core.concept()(Bare)

        report = validate(ontology)

        assert subjects(report, Check.UNDOCUMENTED_CONCEPT) == {"Bare"}
        assert subjects(report, Check.UNDOCUMENTED_PROPERTY) == {"Bare.value"}
        assert subjects(report, Check.UNDOCUMENTED_MEMBER) == {"Silent.QUIET"}
        assert severities(report, Check.UNDOCUMENTED_CONCEPT) == {Severity.WARNING}
        assert severities(report, Check.UNDOCUMENTED_PROPERTY) == {Severity.WARNING}
        assert severities(report, Check.UNDOCUMENTED_MEMBER) == {Severity.WARNING}


# %% scope


class TestModuleScope:
    """
    Validating one module reports only what that module declares.
    """

    def test_only_the_named_modules_declarations_are_reported(self) -> None:
        ontology, core = new_ontology()
        extra = ontology.module("extra", version="1.0.0", namespace=EXTRA)
        core.concept()(Widget)
        extra.concept()(Cardinality)

        assert validate(ontology, module="core").of(Check.COLLECTION_MISMATCH) == ()
        assert subjects(validate(ontology, module="extra"), Check.COLLECTION_MISMATCH)


# %% documentation of the validation vocabulary


class TestValidationVocabularyIsDocumented:
    """
    Every field and member of the validation vocabulary carries documentation.
    """

    @pytest.mark.parametrize("dataclass_type", [Violation, ValidationReport])
    def test_every_field_is_documented(self, dataclass_type: type) -> None:
        assert DocumentationReader().find_undocumented_fields(dataclass_type) == ()

    def test_every_check_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(Check) == ()
