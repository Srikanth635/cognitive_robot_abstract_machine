from __future__ import annotations

from typing import Callable

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    PhaseTransition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ThresholdCrossingEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CouplingRegime,
    PhysicalDimension,
    RegulationMode,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.rules import (
    POURING_ACCEPTANCE,
    PouringRule,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.value_partitions import (
    MaterialRegime,
    TransferTerminationReason,
)
from llm_reasoner.knowledge_updated.ontological.acceptance import AcceptanceViolation
from llm_reasoner.knowledge_updated.ontological.derived import DerivedRule
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

from .pouring_structure import PouringParts, build_valid_pouring

# %% helpers


def evaluate(parts: PouringParts) -> tuple[AcceptanceViolation, ...]:
    """
    Run the whole acceptance suite of the module over a structure.
    """
    return POURING_ACCEPTANCE.evaluate(parts.fragments())


def fired(parts: PouringParts) -> set[str]:
    """
    Return the rules a structure breaks.
    """
    return {violation.rule for violation in evaluate(parts)}


# %% one break for each rule


def break_material(p: PouringParts) -> None:
    p.structure.material_regime = MaterialRegime.GRANULAR


def break_termination(p: PouringParts) -> None:
    p.structure.termination_reason = TransferTerminationReason.SOURCE_EMPTY


def break_without_tilt(p: PouringParts) -> None:
    p.structure.defines_phase.remove(p.tilt)


def break_without_regulation(p: PouringParts) -> None:
    p.structure.defines_phase.remove(p.flow)


def break_without_termination(p: PouringParts) -> None:
    p.structure.defines_phase.remove(p.stop)


def break_order(p: PouringParts) -> None:
    p.structure.defines_transition.append(
        PhaseTransition(
            label="backward", from_phase=p.stop, to_phase=p.tilt, gated_by=[p.cessation]
        )
    )


def break_general_event(p: PouringParts) -> None:
    p.structure.defines_event_type.append(ThresholdCrossingEventType(label="general"))


def break_tilt_boundary(p: PouringParts) -> None:
    p.tilt.terminated_by = [p.target]


def break_stop_boundary(p: PouringParts) -> None:
    p.stop.terminated_by = [p.upright_restored]


BREAKS: dict[PouringRule, Callable[[PouringParts], None]] = {
    PouringRule.MATERIAL_REGIME_EXCLUDES_ESTIMATE: break_material,
    PouringRule.TERMINATION_REASON_HAS_NO_EVENT: break_termination,
    PouringRule.TRANSFER_HAS_NO_TILT_TO_FLOW_PHASE: break_without_tilt,
    PouringRule.TRANSFER_HAS_NO_FLOW_REGULATION_PHASE: break_without_regulation,
    PouringRule.TRANSFER_HAS_NO_FLOW_TERMINATION_PHASE: break_without_termination,
    PouringRule.PHASES_AGAINST_SENSORIMOTOR_ORDER: break_order,
    PouringRule.EVENT_NOT_A_SPECIFIC_KIND: break_general_event,
    PouringRule.TILT_PHASE_NOT_ENDED_BY_FLOW_ONSET: break_tilt_boundary,
    PouringRule.FLOW_TERMINATION_NOT_ENDED_BY_FLOW_CESSATION: break_stop_boundary,
}
"""
For each rule of the module, the smallest change to the valid structure that breaks it.
"""

WARNINGS = {
    PouringRule.TRANSFER_HAS_NO_TILT_TO_FLOW_PHASE,
    PouringRule.TRANSFER_HAS_NO_FLOW_REGULATION_PHASE,
    PouringRule.TRANSFER_HAS_NO_FLOW_TERMINATION_PHASE,
    PouringRule.PHASES_AGAINST_SENSORIMOTOR_ORDER,
}
"""
The rules whose breach is legal but probably unintended, so it is only a warning.
"""


# %% the valid structure


class TestValidStructure:
    """
    A structure built from pouring concepts can satisfy every rule at once.
    """

    def test_valid_structure_raises_nothing(self) -> None:
        assert evaluate(build_valid_pouring()) == ()

    def test_valid_structure_reaches_every_fragment_it_declares(self) -> None:
        parts = build_valid_pouring()

        assert {
            parts.align,
            parts.flow,
            parts.upright,
            parts.viscosity,
            parts.spill,
        } <= set(parts.fragments())


# %% each rule fires


class TestEveryRuleCanFire:
    """
    Breaking one thing in a valid structure makes the rule for it fire.
    """

    @pytest.mark.parametrize("rule", list(PouringRule), ids=lambda rule: rule.name)
    def test_rule_fires_when_its_condition_is_broken(self, rule: PouringRule) -> None:
        parts = build_valid_pouring()
        BREAKS[rule](parts)

        assert rule in fired(parts)

    def test_every_rule_has_a_break(self) -> None:
        assert set(BREAKS) == set(PouringRule)

    @pytest.mark.parametrize("rule", list(PouringRule), ids=lambda rule: rule.name)
    def test_rule_has_the_severity_it_is_meant_to_have(self, rule: PouringRule) -> None:
        parts = build_valid_pouring()
        BREAKS[rule](parts)
        severities = {
            violation.severity
            for violation in evaluate(parts)
            if violation.rule is rule
        }

        assert severities == {Severity.WARNING if rule in WARNINGS else Severity.ERROR}


# %% what the declarations already say


class TestDerivedFromDeclarations:
    """
    Whatever a declaration states is checked without a written rule.
    """

    def test_feedforward_flow_regulation_is_not_allowed(self) -> None:
        parts = build_valid_pouring()
        parts.flow.regulation_mode = RegulationMode.FEEDFORWARD

        assert DerivedRule.VALUE_NOT_ALLOWED in fired(parts)

    def test_uncoupled_flow_regulation_breaks_its_fixed_coupling(self) -> None:
        parts = build_valid_pouring()
        parts.flow.coupling_regime = CouplingRegime.UNCOUPLED

        assert DerivedRule.FIXED_VALUE_NOT_HELD in fired(parts)

    def test_a_quantity_to_pour_in_another_dimension_is_not_allowed(self) -> None:
        parts = build_valid_pouring()
        parts.quantity.dimension = PhysicalDimension.LENGTH

        assert DerivedRule.VALUE_NOT_ALLOWED in fired(parts)

    def test_flow_onset_crossing_a_spill_threshold_is_reported(self) -> None:
        parts = build_valid_pouring()
        parts.onset.crosses_threshold = parts.spill_threshold

        assert DerivedRule.VALUE_OF_WRONG_TYPE in fired(parts)

    def test_a_pouring_estimate_the_structure_does_not_declare_is_reported(
        self,
    ) -> None:
        parts = build_valid_pouring()
        parts.structure.defines_parameter.remove(parts.quantity)

        assert DerivedRule.REFERENCE_NOT_DECLARED in fired(parts)


# %% documentation


class TestRulesAreDocumented:
    """
    Every rule says why it exists.
    """

    def test_every_rule_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(PouringRule) == ()
