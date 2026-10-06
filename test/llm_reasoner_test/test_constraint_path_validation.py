from __future__ import annotations

import pytest
from llm_reasoner.knowledge.microworlds.flanagan.constraints import CONSTRAINTS
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ActionPhaseStructure,
    MagnitudeRange,
    ManipulationRole,
    PhaseTransition,
    RoleSlot,
)
from llm_reasoner.knowledge.ontological.constraints import (
    Closure,
    ConceptReferenceType,
    Constraint,
    ConstraintPathError,
    ConstraintSuite,
    Disjoint,
    Distinct,
    DistinctAcross,
    ExactlyOne,
    FlagOnPath,
    Forbidden,
    GraphShape,
    Grounded,
    Inclusion,
    MostSpecificTypes,
    NonEmptyString,
    NotAmong,
    Ordered,
    SharedValue,
    TypeCoverage,
    TypeOrder,
)

# %% fixtures

UNKNOWN_FIELD = "no_such_field"
"""
A field that no ontology class declares.
"""

RATIONALE = "declared by a path validation test"


def suite_of(constraint: Constraint) -> ConstraintSuite:
    """
    Build a suite holding one constraint, which validates the constraint's paths.
    """
    return ConstraintSuite(name="path-validation", constraints=[constraint])


# One constraint per path-bearing position of every kind. Each carries exactly one
# unknown step and otherwise valid paths, so a rejection is attributable to that position.
UNKNOWN_PATH_CASES: dict[str, Constraint] = {
    "inclusion-source": Inclusion(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        source=UNKNOWN_FIELD,
        target="from_phase",
    ),
    "inclusion-target": Inclusion(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        source="from_phase",
        target=UNKNOWN_FIELD,
    ),
    "disjoint-left": Disjoint(
        id="T",
        applies_to=ActionPhase,
        rationale=RATIONALE,
        left=UNKNOWN_FIELD,
        right="terminated_by",
    ),
    "disjoint-right": Disjoint(
        id="T",
        applies_to=ActionPhase,
        rationale=RATIONALE,
        left="initiated_by",
        right=UNKNOWN_FIELD,
    ),
    "distinct-left": Distinct(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        left=UNKNOWN_FIELD,
        right="to_phase",
    ),
    "distinct-right": Distinct(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        left="from_phase",
        right=UNKNOWN_FIELD,
    ),
    "flag-on-path-path": FlagOnPath(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        path=UNKNOWN_FIELD,
        flag="is_failure_event",
    ),
    "flag-on-path-flag": FlagOnPath(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        path="gated_by",
        flag=UNKNOWN_FIELD,
    ),
    "ordered-lower": Ordered(
        id="T",
        applies_to=MagnitudeRange,
        rationale=RATIONALE,
        lower=UNKNOWN_FIELD,
        upper="upper_bound",
    ),
    "ordered-upper": Ordered(
        id="T",
        applies_to=MagnitudeRange,
        rationale=RATIONALE,
        lower="lower_bound",
        upper=UNKNOWN_FIELD,
    ),
    "non-empty-string-field": NonEmptyString(
        id="T",
        applies_to=RoleSlot,
        rationale=RATIONALE,
        field_name=UNKNOWN_FIELD,
    ),
    "exactly-one-fields": ExactlyOne(
        id="T",
        applies_to=MagnitudeRange,
        rationale=RATIONALE,
        fields=("lower_bound", UNKNOWN_FIELD),
    ),
    "closure-container": Closure(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container=UNKNOWN_FIELD,
        reached_via=("has_transition.from_phase",),
    ),
    "closure-reached-via": Closure(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container="defines_phase",
        reached_via=(UNKNOWN_FIELD,),
    ),
    "not-among-value-path": NotAmong(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        value_path=UNKNOWN_FIELD,
        collection="defines_phase",
    ),
    "not-among-collection": NotAmong(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        value_path="decomposes_task",
        collection=UNKNOWN_FIELD,
    ),
    "shared-value-left": SharedValue(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        left=UNKNOWN_FIELD,
        right="to_phase",
    ),
    "shared-value-right": SharedValue(
        id="T",
        applies_to=PhaseTransition,
        rationale=RATIONALE,
        left="from_phase",
        right=UNKNOWN_FIELD,
    ),
    "distinct-across-collection": DistinctAcross(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        collection=UNKNOWN_FIELD,
        key_path="terminated_by",
    ),
    "distinct-across-key-path": DistinctAcross(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        collection="defines_phase",
        key_path=UNKNOWN_FIELD,
    ),
    "graph-shape-container": GraphShape(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container=UNKNOWN_FIELD,
        edges="has_transition",
    ),
    "graph-shape-edges": GraphShape(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container="defines_phase",
        edges=UNKNOWN_FIELD,
    ),
    "graph-shape-tail": GraphShape(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container="defines_phase",
        edges="has_transition",
        tail=UNKNOWN_FIELD,
    ),
    "graph-shape-head": GraphShape(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container="defines_phase",
        edges="has_transition",
        head=UNKNOWN_FIELD,
    ),
    "grounded-container": Grounded(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container=UNKNOWN_FIELD,
        needs="has_condition_requirement",
        supplies=("defines_phase.aims_for",),
    ),
    "grounded-needs": Grounded(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container="defines_phase",
        needs=UNKNOWN_FIELD,
        supplies=("defines_phase.aims_for",),
    ),
    "grounded-supplies": Grounded(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        container="defines_phase",
        needs="has_condition_requirement",
        supplies=(UNKNOWN_FIELD,),
    ),
    "most-specific-types-collection": MostSpecificTypes(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        collection=UNKNOWN_FIELD,
    ),
    "concept-reference-type-path": ConceptReferenceType(
        id="T",
        applies_to=RoleSlot,
        rationale=RATIONALE,
        path=UNKNOWN_FIELD,
        concept_root=ManipulationRole,
    ),
    "type-coverage-collection": TypeCoverage(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        collection=UNKNOWN_FIELD,
        required_types=(ActionPhase,),
    ),
    "type-order-edges": TypeOrder(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        edges=UNKNOWN_FIELD,
        expected_types=(ActionPhase,),
    ),
    "type-order-tail": TypeOrder(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        edges="has_transition",
        tail=UNKNOWN_FIELD,
        expected_types=(ActionPhase,),
    ),
    "type-order-head": TypeOrder(
        id="T",
        applies_to=ActionPhaseStructure,
        rationale=RATIONALE,
        edges="has_transition",
        head=UNKNOWN_FIELD,
        expected_types=(ActionPhase,),
    ),
}

KINDS_WITHOUT_PATHS = {Forbidden}
"""
Constraint kinds whose behaviour is a guard and that name no field.
"""


def all_constraint_kinds(kind: type[Constraint] = Constraint) -> set[type[Constraint]]:
    """
    Return every concrete constraint kind derived from a base.
    """
    kinds: set[type[Constraint]] = set()
    for subclass in kind.__subclasses__():
        kinds.add(subclass)
        kinds |= all_constraint_kinds(subclass)
    return kinds


# %% unknown paths


class TestUnknownPathRejection:
    """
    A constraint naming a field its subject cannot reach is rejected when the suite is
    built.
    """

    @pytest.mark.parametrize("case", UNKNOWN_PATH_CASES)
    def test_every_path_position_is_checked(self, case: str) -> None:
        constraint = UNKNOWN_PATH_CASES[case]

        with pytest.raises(ConstraintPathError) as rejection:
            suite_of(constraint)

        assert rejection.value.constraint_id == constraint.id
        assert rejection.value.unknown_step == UNKNOWN_FIELD
        assert rejection.value.subject_type is constraint.applies_to

    def test_every_constraint_kind_is_covered(self) -> None:
        covered = {type(constraint) for constraint in UNKNOWN_PATH_CASES.values()}

        assert covered | KINDS_WITHOUT_PATHS == all_constraint_kinds()

    def test_error_reports_the_declared_path_and_the_first_unknown_step(self) -> None:
        constraint = Inclusion(
            id="T",
            applies_to=PhaseTransition,
            rationale=RATIONALE,
            source=f"from_phase.{UNKNOWN_FIELD}",
            target="from_phase",
        )

        with pytest.raises(ConstraintPathError) as rejection:
            suite_of(constraint)

        assert rejection.value.path == constraint.source
        assert rejection.value.unknown_step == UNKNOWN_FIELD


# %% accepted paths


class TestValidPathAcceptance:
    """
    Paths that reach a declared field are accepted, however the field is declared.
    """

    def test_step_declared_only_by_a_subtype_of_the_range_is_accepted(self) -> None:
        """
        ``crosses_threshold`` belongs to threshold-crossing events, a subtype of the
        declared range of ``defines_event_type``.
        """
        constraint = Closure(
            id="T",
            applies_to=ActionPhaseStructure,
            rationale=RATIONALE,
            container="defines_parameter",
            reached_via=("defines_event_type.crosses_threshold",),
        )

        assert suite_of(constraint).constraints == [constraint]

    def test_guard_constraint_without_paths_is_accepted(self) -> None:
        constraint = Forbidden(
            id="T",
            applies_to=ActionPhase,
            rationale=RATIONALE,
            when=lambda phase: False,
            message=RATIONALE,
        )

        assert suite_of(constraint).constraints == [constraint]

    def test_flanagan_acceptance_suite_paths_all_resolve(self) -> None:
        suite = ConstraintSuite(name="flanagan-rebuilt", constraints=CONSTRAINTS)

        assert len(suite.constraints) == len(CONSTRAINTS)
