from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    PhaseConditionType,
    Subgoal,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    MechanicalEventType,
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ConditionMode,
    ConditionPolarity,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% conditions


class TestPhaseConditionType:
    """
    A condition type describes a kind of world state a phase is concerned with.
    """

    def test_condition_is_a_concrete_core_entity_aligned_as_a_state_type(self) -> None:
        described = FLANAGAN.find_concept(PhaseConditionType)

        assert issubclass(PhaseConditionType, KnowledgeEntity)
        assert described.module == CORE.name
        assert described.abstract is False
        assert described.alignments == (subclass_of(SOMA, "StateType"),)

    def test_kinds_of_condition_share_no_instance(self) -> None:
        assert FLANAGAN.find_concept(PhaseConditionType).axioms.disjoint_siblings

    def test_condition_names_the_roles_it_is_about(self) -> None:
        described = PropertyMeta.declared_on(PhaseConditionType)["condition_on_role"]

        assert described.kind is PropertyKind.RELATION
        assert described.minimum_count == 1
        assert RangeResolver().resolve(
            PhaseConditionType, "condition_on_role"
        ).element_types == (RoleSlot,)

    def test_condition_can_name_the_failure_that_registers_its_loss(self) -> None:
        described = PropertyMeta.declared_on(PhaseConditionType)[
            "violation_signalled_by"
        ]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True
        assert described.minimum_count == 0
        assert RangeResolver().resolve(
            PhaseConditionType, "violation_signalled_by"
        ).element_types == (MechanicalEventType,)

    def test_there_are_no_separate_precondition_or_postcondition_types(self) -> None:
        for name in ("Precondition", "Postcondition"):
            assert FLANAGAN.find_concept_by_iri(CORE.namespace.term(name)) is None


# %% requirements


class TestConditionRequirement:
    """
    A requirement is a phase's demand on a condition, qualified by mode and polarity.
    """

    def test_requirement_is_a_core_entity_aligned_as_a_relation(self) -> None:
        described = FLANAGAN.find_concept(ConditionRequirement)

        assert issubclass(ConditionRequirement, KnowledgeEntity)
        assert described.module == CORE.name
        assert described.alignments == (subclass_of(DUL, "Relation"),)

    def test_requirement_names_exactly_one_condition_and_one_mode(self) -> None:
        declared = PropertyMeta.declared_on(ConditionRequirement)

        assert declared["on_condition"].kind is PropertyKind.RELATION
        assert declared["on_condition"].functional is True
        assert declared["on_condition"].minimum_count == 1
        assert declared["mode"].kind is PropertyKind.ATTRIBUTE
        assert declared["mode"].functional is True
        assert declared["mode"].minimum_count == 1

    def test_requirement_ranges_over_the_declared_types(self) -> None:
        resolver = RangeResolver()

        assert resolver.resolve(ConditionRequirement, "on_condition").element_types == (
            PhaseConditionType,
        )
        assert resolver.resolve(ConditionRequirement, "mode").element_types == (
            ConditionMode,
        )
        assert resolver.resolve(ConditionRequirement, "polarity").element_types == (
            ConditionPolarity,
        )

    def test_a_condition_is_required_to_hold_unless_stated_otherwise(self) -> None:
        assert ConditionRequirement().polarity is ConditionPolarity.MUST_HOLD

    def test_a_requirement_is_not_a_world_assumption_unless_stated_otherwise(
        self,
    ) -> None:
        assert ConditionRequirement().is_world_assumption is False

    def test_the_world_assumption_flag_marks_the_requirement_as_presupposed(
        self,
    ) -> None:
        declared = PropertyMeta.declared_on(ConditionRequirement)

        assert declared["is_world_assumption"].marks_assumption is True


# %% subgoals


class TestSubgoal:
    """
    A subgoal is the state a phase exists to reach, and the event that shows it reached.
    """

    def test_subgoal_is_a_core_entity_but_not_itself_a_condition(self) -> None:
        described = FLANAGAN.find_concept(Subgoal)

        assert issubclass(Subgoal, KnowledgeEntity)
        assert not issubclass(Subgoal, PhaseConditionType)
        assert described.module == CORE.name

    def test_subgoal_targets_exactly_one_condition(self) -> None:
        described = PropertyMeta.declared_on(Subgoal)["targets_condition"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True
        assert described.minimum_count == 1
        assert RangeResolver().resolve(Subgoal, "targets_condition").element_types == (
            PhaseConditionType,
        )

    def test_subgoal_names_exactly_one_event_that_shows_it_reached(self) -> None:
        described = PropertyMeta.declared_on(Subgoal)["attainment_signalled_by"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True
        assert described.minimum_count == 1
        assert RangeResolver().resolve(
            Subgoal, "attainment_signalled_by"
        ).element_types == (PhaseBoundaryEventType,)


# %% documentation


class TestConditionsAreDocumented:
    """
    Every condition concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", [PhaseConditionType, ConditionRequirement, Subgoal])
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
