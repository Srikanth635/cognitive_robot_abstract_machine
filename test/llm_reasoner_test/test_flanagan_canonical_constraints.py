from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ConditionMode,
    ConditionRequirement,
    PhaseTransition,
    RoleSlot,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (
    PouringPhaseStructure,
    SpillEvent,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    ForceLoadingPhase,
    ReachPhase,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% Constraint behavior


class TestMalformedInvariantReporting:
    """
    Malformed generated conditions are reported rather than crashing.
    """

    def test_event_used_as_invariant_condition_produces_a_violation(self) -> None:
        spill = SpillEvent(label="Spill", is_failure_event=True)
        requirement = ConditionRequirement(
            label="Avoid spilling",
            mode=ConditionMode.THROUGHOUT,
            on_condition=spill,
        )

        violations = SUITE.evaluate((spill, requirement))

        assert any(violation.check == "C28" for violation in violations)


class TestCanonicalPhaseClassification:
    """
    Pouring structures reject action-neutral phase instances.
    """

    def test_generic_action_phase_is_not_specific_enough_for_pouring(self) -> None:
        generic_phase = ActionPhase(label="Move the object")
        structure = PouringPhaseStructure(
            label="Pouring phases",
            defines_phase=[generic_phase],
        )

        violations = SUITE.evaluate((structure, generic_phase))

        assert any(violation.check == "C30a" for violation in violations)


class TestCanonicalRoleReference:
    """
    Role slots must reference a declared manipulation-role concept.
    """

    def test_unknown_role_concept_is_rejected(self) -> None:
        role = RoleSlot(
            label="Unknown participant",
            binds_as=ConceptReference(iri="example:UnknownRole"),
        )

        violations = SUITE.evaluate((role,))

        assert any(violation.check == "C30c" for violation in violations)


class TestPouringPhaseCoverage:
    """
    Incomplete canonical phase inventories are reported as warnings.
    """

    def test_missing_flow_regulation_phase_is_reported(self) -> None:
        reach = ReachPhase(label="Reach")
        structure = PouringPhaseStructure(
            label="Incomplete pouring phases",
            defines_phase=[reach],
        )

        violations = SUITE.evaluate((structure, reach))
        coverage = [violation for violation in violations if violation.check == "C31"]

        assert coverage
        assert all(violation.severity == "warning" for violation in coverage)


class TestPouringPhaseOrder:
    """
    Canonical phase types follow their sensorimotor ordering.
    """

    def test_reverse_reach_to_load_order_is_reported(self) -> None:
        reach = ReachPhase(label="Reach")
        load = ForceLoadingPhase(label="ForceLoading")
        transition = PhaseTransition(
            label="Reverse transition",
            from_phase=load,
            to_phase=reach,
        )
        structure = PouringPhaseStructure(
            label="Reversed pouring phases",
            defines_phase=[reach, load],
            has_transition=[transition],
        )

        violations = SUITE.evaluate((structure, reach, load, transition))
        ordering = [violation for violation in violations if violation.check == "C32"]

        assert ordering
        assert all(violation.severity == "warning" for violation in ordering)
