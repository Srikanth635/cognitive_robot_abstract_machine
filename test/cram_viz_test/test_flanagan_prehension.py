from __future__ import annotations

from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    CorrectionDirection,
    CouplingRegime,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    AnticipatoryGripAdjustment,
    ContactBreak,
    ContactReleasePhase,
    CorrectiveGripResponse,
    ForceLoadingPhase,
    ForcesUnloaded,
    GripLoadCoupling,
    Liftoff,
    LiftPhase,
    ReplacePhase,
    Touchdown,
    UnloadPhase,
    WithdrawPhase,
)
from llm_reasoner.knowledge.query import HypothesisView

from cram_viz import hypothesis_knowledge

# %% phase semantics


class TestCanonicalPrehensionSequence:
    """
    The reference structure preserves the prehensile control sequence.
    """

    def test_force_loading_transitions_to_lift_at_liftoff(self) -> None:
        structure = build_pouring_phase_structure()
        loading = next(
            phase
            for phase in structure.defines_phase
            if isinstance(phase, ForceLoadingPhase)
        )
        lift = next(
            phase for phase in structure.defines_phase if isinstance(phase, LiftPhase)
        )

        assert len(loading.terminated_by) == 1
        assert isinstance(loading.terminated_by[0], Liftoff)
        assert lift.initiated_by == loading.terminated_by

    def test_lowering_stays_coupled_through_touchdown(self) -> None:
        structure = build_pouring_phase_structure()
        lowering = next(
            phase
            for phase in structure.defines_phase
            if isinstance(phase, ReplacePhase)
        )

        assert lowering.coupling_regime is CouplingRegime.COUPLED
        assert len(lowering.terminated_by) == 1
        assert isinstance(lowering.terminated_by[0], Touchdown)

    def test_unload_release_and_withdraw_have_distinct_boundaries(self) -> None:
        structure = build_pouring_phase_structure()
        unloading = next(
            phase for phase in structure.defines_phase if isinstance(phase, UnloadPhase)
        )
        release = next(
            phase
            for phase in structure.defines_phase
            if isinstance(phase, ContactReleasePhase)
        )
        withdrawal = next(
            phase
            for phase in structure.defines_phase
            if isinstance(phase, WithdrawPhase)
        )

        assert isinstance(unloading.terminated_by[0], ForcesUnloaded)
        assert [
            role.label for role in unloading.terminated_by[0].weight_supported_by
        ] == ["final support: kitchen counter"]
        assert isinstance(release.terminated_by[0], ContactBreak)
        assert withdrawal.initiated_by == release.terminated_by


# %% force control


class TestCanonicalGripControl:
    """
    The reference structure covers reactive and anticipatory grip control.
    """

    def test_transport_is_in_the_functional_grip_load_coupling(self) -> None:
        structure = build_pouring_phase_structure()
        coupling = structure.has_force_coupling[0]

        assert isinstance(coupling, GripLoadCoupling)
        assert coupling.predicted_load is not None
        assert coupling.friction_estimate is not None
        assert coupling.safety_margin.margin_on is coupling.grip_force_response
        assert any(phase.label == "Transport" for phase in coupling.coupling_active_in)

    def test_slip_releases_a_typed_grip_correction(self) -> None:
        structure = build_pouring_phase_structure()
        corrections = [
            correction
            for correction in structure.defines_correction
            if isinstance(correction, CorrectiveGripResponse)
        ]

        assert len(corrections) == 1
        assert corrections[0].direction is CorrectionDirection.INCREASE

    def test_touchdown_has_an_anticipatory_grip_adjustment(self) -> None:
        structure = build_pouring_phase_structure()

        assert len(structure.has_anticipatory_grip_adjustment) == 1
        adjustment = structure.has_anticipatory_grip_adjustment[0]
        assert isinstance(adjustment, AnticipatoryGripAdjustment)
        touchdown = adjustment.triggered_by_prediction.predicts_event
        assert isinstance(touchdown, Touchdown)
        assert (
            touchdown
            in structure.has_internal_model_composition[
                0
            ].predicted_mechanical_consequence
        )
        assert all(
            isinstance(phase, ReplacePhase) for phase in adjustment.adjustment_active_in
        )

    def test_typed_queries_return_coupling_and_anticipatory_adjustment(self) -> None:
        structure = build_pouring_phase_structure()
        record = hypothesis_knowledge.build_hypothesis_record(structure)
        view = HypothesisView(record)

        assert [
            answer.value for answer in view.force_couplings()
        ] == structure.has_force_coupling
        assert [
            answer.value for answer in view.anticipatory_grip_adjustments()
        ] == structure.has_anticipatory_grip_adjustment
