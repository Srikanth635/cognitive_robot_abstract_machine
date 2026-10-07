"""
The acceptance suite of the predictive manipulation microworld.
"""

from __future__ import annotations

from llm_reasoner.acceptance import core, gaze, object_dynamics, pouring, prehension
from llm_reasoner.acceptance.derived import RangesHold, RestrictionsHold
from llm_reasoner.acceptance.rules import AcceptanceSuite


def predictive_manipulation_suite() -> AcceptanceSuite:
    """
    :return: Every acceptance rule of the microworld: the two derived from the ontology,
        then the rules of each module in dependency order.
    """
    return AcceptanceSuite(
        rules=[
            RestrictionsHold(),
            RangesHold(),
            core.RangeBoundsOrdered(),
            core.WindowBoundsOrdered(),
            core.TransitionGateEndsSource(),
            core.TransitionGateStartsTarget(),
            core.TransitionIsNotALoop(),
            core.PhaseHasExtent(),
            core.SubgoalSignalledByPhaseEnd(),
            core.InvariantHasViolationEvent(),
            core.SafetyMarginOnOneParticipant(),
            core.TaskIsNotItsOwnPhase(),
            core.OneEntryPhase(),
            core.NoOrphanPhase(),
            core.PhaseEndingsDistinct(),
            core.PreconditionsGrounded(),
            core.MatchHasObservation(),
            core.MatchHasNoResponse(),
            core.AbsentEventHasNoObservation(),
            core.ErrorHasObservation(),
            core.PermissionFitsTransition(),
            core.UpdatedEstimateIsHeld(),
            core.CorrectionAnswersClassifiedError(),
            core.ModelUpdateAnswersClassifiedError(),
            core.EvidenceCompatible(),
            core.ControlPointPredictsItsEvent(),
            core.ControlPointComparesItsPredictions(),
            core.ControlPointHandlesBothOutcomes(),
            core.MismatchInvokesCorrection(),
            core.ContinuousPredictionComparesItsModality(),
            core.ToleranceMatchesTrajectory(),
            core.ControlledSetpointIsActive(),
            core.FeedbackControlHasContinuousPrediction(),
            core.MechanicalBoundaryMonitored(),
            core.ExemptionNamesNormalBoundary(),
            core.ComponentsDeclared(),
            object_dynamics.CompositionPredictsWithItsForwardModel(),
            object_dynamics.CompositionDynamicsUsesItsCopy(),
            object_dynamics.EstimateHeldByARepresentation(),
            object_dynamics.EstimateServesOnePurpose(),
            object_dynamics.DissociationCoherent(),
            object_dynamics.RetainedUpdateInItsMemory(),
            object_dynamics.TransferComparesMatchedPerturbations(),
            object_dynamics.RepresentationCoversEstimatedRoles(),
            prehension.SafetyMarginOnCouplingResponse(),
            prehension.AdjustmentPrecedesPredictedEvent(),
            prehension.PrehensionPhasesSpecific(),
            pouring.PouringEventsSpecific(),
            pouring.PouringPhasesOrdered(),
            pouring.MaterialFitsFlowDynamics(),
            pouring.TerminationReasonHasEvent(),
            gaze.FixationHeldUntilPhaseGoal(),
            gaze.FixationOnPhaseParticipant(),
        ]
    )
