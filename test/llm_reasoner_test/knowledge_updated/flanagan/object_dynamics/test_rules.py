from __future__ import annotations

from typing import Callable

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    ForwardModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.rules import CoreRule
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.commands import (
    EfferenceCopy,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    LoadForceEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.rules import (
    OBJECT_DYNAMICS_ACCEPTANCE,
    ObjectDynamicsRule,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.value_partitions import (
    PerturbationKind,
    RepresentationPurpose,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    MeasurementUnit,
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.ontological.acceptance import AcceptanceViolation
from llm_reasoner.knowledge_updated.ontological.derived import DerivedRule
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

from ..core_structure import magnitude
from .object_dynamics_structure import (
    ObjectDynamicsParts,
    build_valid_object_dynamics,
)

# %% helpers


def evaluate(parts: ObjectDynamicsParts) -> tuple[AcceptanceViolation, ...]:
    """
    Run the whole acceptance suite of the module over a structure.
    """
    return OBJECT_DYNAMICS_ACCEPTANCE.evaluate(parts.fragments())


def fired(parts: ObjectDynamicsParts) -> set[str]:
    """
    Return the rules a structure breaks.
    """
    return {violation.rule for violation in evaluate(parts)}


# %% one break for each rule


def break_copy_not_given_to_model(s: ObjectDynamicsParts) -> None:
    s.composition.efference_copy = EfferenceCopy(copies_command=s.command)


def break_prediction_from_other_model(s: ObjectDynamicsParts) -> None:
    s.core.prediction.generated_by = ForwardModel(label="another forward model")


def break_unheld_estimate(s: ObjectDynamicsParts) -> None:
    stray = LoadForceEstimate(
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.FORCE,
        parameter_range=magnitude(1.0, 2.0, MeasurementUnit.NEWTON),
        parameter_on_role=[s.core.manipulandum],
    )
    s.structure.defines_parameter.append(stray)


def break_two_purposes(s: ObjectDynamicsParts) -> None:
    s.perceptual.holds_estimate = [s.load_force]


def break_uncovered_role(s: ObjectDynamicsParts) -> None:
    s.object_model.represents_role = [s.core.actor]


def break_update_outside_memory(s: ObjectDynamicsParts) -> None:
    s.memory.stores_representation = [s.limb_model]


def break_irreciprocal_match(s: ObjectDynamicsParts) -> None:
    s.limb_load.matched_with = []


def break_match_across_kinds(s: ObjectDynamicsParts) -> None:
    s.limb_load.perturbation_kind = PerturbationKind.ROTATION


def break_unmatched_transfer(s: ObjectDynamicsParts) -> None:
    s.object_load.matched_with = []


def break_transfer_to_itself(s: ObjectDynamicsParts) -> None:
    s.transfer.to_perturbation = s.object_load


def break_sensorimotor_estimate(s: ObjectDynamicsParts) -> None:
    s.dissociation.sensorimotor_estimate = s.perceived_heaviness


def break_perceptual_estimate(s: ObjectDynamicsParts) -> None:
    s.dissociation.perceptual_estimate = s.load_force


def break_indistinct_estimates(s: ObjectDynamicsParts) -> None:
    s.dissociation.perceptual_estimate = s.dissociation.sensorimotor_estimate


BREAKS: dict[ObjectDynamicsRule, Callable[[ObjectDynamicsParts], None]] = {
    ObjectDynamicsRule.DYNAMICS_MODEL_NOT_GIVEN_THE_COPY: break_copy_not_given_to_model,
    ObjectDynamicsRule.PREDICTION_NOT_FROM_THE_COMPOSITIONS_FORWARD_MODEL: break_prediction_from_other_model,
    ObjectDynamicsRule.ESTIMATE_HELD_BY_NO_REPRESENTATION: break_unheld_estimate,
    ObjectDynamicsRule.ESTIMATE_SERVES_TWO_PURPOSES: break_two_purposes,
    ObjectDynamicsRule.REPRESENTATION_OMITS_ESTIMATED_ROLE: break_uncovered_role,
    ObjectDynamicsRule.RETAINED_UPDATE_OUTSIDE_ITS_MEMORY: break_update_outside_memory,
    ObjectDynamicsRule.PERTURBATION_MATCH_NOT_RECIPROCAL: break_irreciprocal_match,
    ObjectDynamicsRule.PERTURBATION_MATCHED_ACROSS_KINDS: break_match_across_kinds,
    ObjectDynamicsRule.TRANSFER_BETWEEN_UNMATCHED_PERTURBATIONS: break_unmatched_transfer,
    ObjectDynamicsRule.TRANSFER_WITHIN_ONE_PERTURBATION: break_transfer_to_itself,
    ObjectDynamicsRule.SENSORIMOTOR_ESTIMATE_NOT_HELD: break_sensorimotor_estimate,
    ObjectDynamicsRule.PERCEPTUAL_ESTIMATE_NOT_HELD: break_perceptual_estimate,
    ObjectDynamicsRule.DISSOCIATED_ESTIMATES_NOT_DISTINCT: break_indistinct_estimates,
}
"""
For each rule of the module, the smallest change to the valid structure that breaks it.
"""


# %% the valid structure


class TestValidStructure:
    """
    A structure built from core and object-dynamics concepts can satisfy every rule.
    """

    def test_valid_structure_raises_nothing(self) -> None:
        assert evaluate(build_valid_object_dynamics()) == ()

    def test_valid_structure_reaches_every_fragment_it_declares(self) -> None:
        parts = build_valid_object_dynamics()

        assert {
            parts.command,
            parts.copy,
            parts.object_model,
            parts.limb_model,
            parts.perceptual,
            parts.composition,
            parts.memory,
            parts.update,
            parts.transfer,
            parts.dissociation,
        } <= set(parts.fragments())


# %% each rule fires


class TestEveryRuleCanFire:
    """
    Breaking one thing in a valid structure makes the rule for it fire.
    """

    @pytest.mark.parametrize(
        "rule", list(ObjectDynamicsRule), ids=lambda rule: rule.name
    )
    def test_rule_fires_when_its_condition_is_broken(
        self, rule: ObjectDynamicsRule
    ) -> None:
        parts = build_valid_object_dynamics()
        BREAKS[rule](parts)

        assert rule in fired(parts)

    def test_every_rule_has_a_break(self) -> None:
        assert set(BREAKS) == set(ObjectDynamicsRule)

    @pytest.mark.parametrize(
        "rule", list(ObjectDynamicsRule), ids=lambda rule: rule.name
    )
    def test_rule_is_an_error(self, rule: ObjectDynamicsRule) -> None:
        parts = build_valid_object_dynamics()
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

    def test_a_dynamics_model_serving_judgment_breaks_its_fixed_purpose(self) -> None:
        parts = build_valid_object_dynamics()
        parts.object_model.purpose = RepresentationPurpose.PERCEPTUAL_JUDGMENT

        assert DerivedRule.FIXED_VALUE_NOT_HELD in fired(parts)

    def test_a_perceptual_representation_cannot_stand_for_sensorimotor_control(
        self,
    ) -> None:
        parts = build_valid_object_dynamics()
        parts.dissociation.sensorimotor_representation = parts.perceptual

        assert DerivedRule.VALUE_OF_WRONG_TYPE in fired(parts)

    def test_a_command_the_structure_does_not_declare_is_reported(self) -> None:
        parts = build_valid_object_dynamics()
        parts.structure.defines_motor_command = []

        assert DerivedRule.REFERENCE_NOT_DECLARED in fired(parts)

    def test_a_representation_holding_nothing_is_reported(self) -> None:
        parts = build_valid_object_dynamics()
        parts.limb_model.holds_estimate = []

        assert DerivedRule.MINIMUM_COUNT_NOT_MET in fired(parts)

    def test_a_load_force_estimate_in_another_dimension_is_reported(self) -> None:
        parts = build_valid_object_dynamics()
        parts.load_force.dimension = PhysicalDimension.MASS

        assert DerivedRule.FIXED_VALUE_NOT_HELD in fired(parts)


# %% the core still applies


class TestCoreRulesStillApply:
    """
    A structure of this module is still a core phase structure.
    """

    def test_a_retained_update_changing_an_unheld_estimate_breaks_the_core_rule(
        self,
    ) -> None:
        parts = build_valid_object_dynamics()
        parts.update.updates_estimate = [parts.perceived_heaviness]

        assert CoreRule.MODEL_UPDATE_CHANGES_UNHELD_ESTIMATE in fired(parts)


# %% documentation


class TestRulesAreDocumented:
    """
    Every rule says why it exists.
    """

    def test_every_rule_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(ObjectDynamicsRule) == ()
