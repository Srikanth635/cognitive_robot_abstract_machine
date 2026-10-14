from __future__ import annotations

from typing import Callable

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    PredictionError,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ContactTransition,
    CouplingRegime,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.control import (
    CorrectiveGripResponse,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.rules import (
    PREHENSION_ACCEPTANCE,
    PrehensionRule,
)
from llm_reasoner.knowledge_updated.ontological.acceptance import AcceptanceViolation
from llm_reasoner.knowledge_updated.ontological.derived import DerivedRule
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

from .prehension_structure import PrehensionParts, build_valid_prehension

# %% helpers


def evaluate(parts: PrehensionParts) -> tuple[AcceptanceViolation, ...]:
    """
    Run the whole acceptance suite of the module over a structure.
    """
    return PREHENSION_ACCEPTANCE.evaluate(parts.fragments())


def fired(parts: PrehensionParts) -> set[str]:
    """
    Return the rules a structure breaks.
    """
    return {violation.rule for violation in evaluate(parts)}


# %% one break for each rule


def break_margin(p: PrehensionParts) -> None:
    p.margin.margin_on = p.load


def break_replace_boundary(p: PrehensionParts) -> None:
    p.replace.terminated_by = [p.unloaded]


def break_unload_boundary(p: PrehensionParts) -> None:
    p.unload.terminated_by = [p.released]


def break_release_boundary(p: PrehensionParts) -> None:
    p.release.terminated_by = [p.withdrawn]


def break_withdraw_start(p: PrehensionParts) -> None:
    p.withdraw.initiated_by = [p.unloaded]


def break_withdraw_end(p: PrehensionParts) -> None:
    p.withdraw.terminated_by = [p.released]


def break_adjustment_trigger(p: PrehensionParts) -> None:
    p.prediction.predicts_event = p.lifted_off


def break_general_phase(p: PrehensionParts) -> None:
    p.structure.defines_phase.append(ActionPhase(label="general phase"))


BREAKS: dict[PrehensionRule, Callable[[PrehensionParts], None]] = {
    PrehensionRule.GRIP_MARGIN_ON_ANOTHER_SETPOINT: break_margin,
    PrehensionRule.REPLACE_PHASE_NOT_ENDED_BY_TOUCHDOWN: break_replace_boundary,
    PrehensionRule.UNLOAD_PHASE_NOT_ENDED_BY_FORCES_UNLOADED: break_unload_boundary,
    PrehensionRule.RELEASE_PHASE_NOT_ENDED_BY_CONTACT_BREAK: break_release_boundary,
    PrehensionRule.WITHDRAWAL_NOT_AFTER_CONTACT_BREAK: break_withdraw_start,
    PrehensionRule.WITHDRAWAL_NOT_ENDED_BY_COMPLETION: break_withdraw_end,
    PrehensionRule.ANTICIPATORY_ADJUSTMENT_NOT_FOR_TOUCHDOWN: break_adjustment_trigger,
    PrehensionRule.PHASE_NOT_A_SPECIFIC_KIND: break_general_phase,
}
"""
For each rule of the module, the smallest change to the valid structure that breaks it.
"""


# %% the valid structure


class TestValidStructure:
    """
    A structure built from prehension concepts can satisfy every rule at once.
    """

    def test_valid_structure_raises_nothing(self) -> None:
        assert evaluate(build_valid_prehension()) == ()

    def test_valid_structure_reaches_every_fragment_it_declares(self) -> None:
        parts = build_valid_prehension()

        assert {
            parts.reach,
            parts.withdraw,
            parts.coupling,
            parts.adjustment,
            parts.load_force,
        } <= set(parts.fragments())


# %% each rule fires


class TestEveryRuleCanFire:
    """
    Breaking one thing in a valid structure makes the rule for it fire.
    """

    @pytest.mark.parametrize("rule", list(PrehensionRule), ids=lambda rule: rule.name)
    def test_rule_fires_when_its_condition_is_broken(
        self, rule: PrehensionRule
    ) -> None:
        parts = build_valid_prehension()
        BREAKS[rule](parts)

        assert rule in fired(parts)

    def test_every_rule_has_a_break(self) -> None:
        assert set(BREAKS) == set(PrehensionRule)

    @pytest.mark.parametrize("rule", list(PrehensionRule), ids=lambda rule: rule.name)
    def test_rule_is_an_error(self, rule: PrehensionRule) -> None:
        parts = build_valid_prehension()
        BREAKS[rule](parts)

        assert {
            violation.severity
            for violation in evaluate(parts)
            if violation.rule is rule
        } == {Severity.ERROR}


# %% what the declarations already say


class TestDerivedFromDeclarations:
    """
    Whatever a declaration states is checked without a written rule.
    """

    def test_a_reach_that_is_coupled_breaks_the_coupling_its_kind_fixes(self) -> None:
        parts = build_valid_prehension()
        parts.reach.coupling_regime = CouplingRegime.COUPLED

        assert DerivedRule.FIXED_VALUE_NOT_HELD in fired(parts)

    def test_a_touchdown_that_breaks_contact_breaks_the_transition_its_kind_fixes(
        self,
    ) -> None:
        parts = build_valid_prehension()
        parts.touched_down.contact_transition = ContactTransition.BREAKS

        assert DerivedRule.FIXED_VALUE_NOT_HELD in fired(parts)

    def test_a_grip_force_in_another_dimension_is_reported(self) -> None:
        parts = build_valid_prehension()
        parts.grip.dimension = PhysicalDimension.LENGTH

        assert DerivedRule.FIXED_VALUE_NOT_HELD in fired(parts)

    def test_liftoff_crossing_another_threshold_is_reported(self) -> None:
        parts = build_valid_prehension()
        parts.lifted_off.crosses_threshold = parts.touched_down.crosses_threshold

        assert DerivedRule.VALUE_OF_WRONG_TYPE in fired(parts)

    def test_a_grip_response_that_changes_load_force_is_reported(self) -> None:
        parts = build_valid_prehension()
        error = PredictionError(label="early liftoff")
        response = CorrectiveGripResponse(
            triggered_by_error=[error], targets_setpoint=[parts.load]
        )
        parts.structure.defines_correction.append(response)
        parts.structure.defines_error.append(error)

        assert DerivedRule.VALUE_OF_WRONG_TYPE in fired(parts)

    def test_an_anticipatory_adjustment_outside_replacing_is_reported(self) -> None:
        parts = build_valid_prehension()
        parts.adjustment.adjustment_active_in = [parts.lift]

        assert DerivedRule.VALUE_OF_WRONG_TYPE in fired(parts)

    def test_a_coupling_the_structure_does_not_declare_is_reported(self) -> None:
        parts = build_valid_prehension()
        parts.structure.defines_force_coupling = []
        parts.structure.defines_parameter.remove(parts.friction)

        assert DerivedRule.REFERENCE_NOT_DECLARED in fired(parts)

    def test_weight_borne_by_an_undeclared_support_is_reported(self) -> None:
        parts = build_valid_prehension()
        parts.structure.defines_role.remove(parts.destination)

        assert DerivedRule.REFERENCE_NOT_DECLARED in fired(parts)


# %% documentation


class TestRulesAreDocumented:
    """
    Every rule says why it exists.
    """

    def test_every_rule_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(PrehensionRule) == ()
