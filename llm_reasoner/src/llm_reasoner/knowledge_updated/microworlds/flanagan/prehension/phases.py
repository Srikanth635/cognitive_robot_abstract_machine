"""
The phases of a prehensile manipulation, from reaching to withdrawing.

A phase that is coupled or uncoupled by its nature says so in its declaration. The
events that must bound a phase are rules of the module, since they relate the phase to
its events.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CouplingRegime,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import PREHENSION
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@PREHENSION.concept(
    fixed_values={"coupling_regime": CouplingRegime.UNCOUPLED},
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class ReachPhase(ActionPhase):
    """
    Transport of the uncoupled effector toward a pre-contact pose.
    """


@PREHENSION.concept()
@dataclass(eq=False)
class ContactEstablishmentPhase(ActionPhase):
    """
    Controlled establishment of effector-manipulandum contact.
    """


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING)
@dataclass(eq=False)
class ForceLoadingPhase(ActionPhase):
    """
    Coordinated grip and load increase from contact until liftoff.
    """


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING)
@dataclass(eq=False)
class LiftPhase(ActionPhase):
    """
    Vertical transport away from the initial support.
    """


@PREHENSION.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class HoldPhase(ActionPhase):
    """
    Optional maintenance of a stable grasp before transport.
    """


@PREHENSION.concept(
    fixed_values={"coupling_regime": CouplingRegime.COUPLED},
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class TransportPhase(ActionPhase):
    """
    Translation of the coupled effector-manipulandum unit.
    """


@PREHENSION.concept(
    fixed_values={"coupling_regime": CouplingRegime.COUPLED},
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class ReplacePhase(ActionPhase):
    """
    Lowering of the still-coupled manipulandum until touchdown.
    """


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING)
@dataclass(eq=False)
class UnloadPhase(ActionPhase):
    """
    Transfer of weight to support while grip and load forces decrease.
    """


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING)
@dataclass(eq=False)
class ContactReleasePhase(ActionPhase):
    """
    Release of effector-manipulandum contact after force unloading.
    """


@PREHENSION.concept()
@dataclass(eq=False)
class WithdrawPhase(ActionPhase):
    """
    Optional effector withdrawal following contact release.
    """
