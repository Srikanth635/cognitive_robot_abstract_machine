"""
Acceptance constraints owned by the sensory-attenuation module.
"""

from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.core import ScientificStatus
from llm_reasoner.knowledge.microworlds.flanagan.perception import (
    AttenuationControlRelationship,
    EventOrigin,
    SensoryAttenuation,
)
from llm_reasoner.knowledge.ontological.constraints import Forbidden

# %% perception constraints


PERCEPTION_CONSTRAINTS = [
    Forbidden(
        id="PERCEPTION-C64",
        applies_to=AttenuationControlRelationship,
        when=lambda relationship: relationship.scientific_status
        is not ScientificStatus.OPEN_QUESTION,
        message="attenuation-control relationship is asserted as resolved",
        rationale="The reviewed work leaves a shared neural prediction mechanism "
        "unresolved.",
    ),
    Forbidden(
        id="PERCEPTION-C65a",
        applies_to=SensoryAttenuation,
        when=lambda attenuation: attenuation.event_origin
        is not EventOrigin.SELF_GENERATED,
        message="attenuation is assigned to a non-self-generated event",
        rationale="The reported attenuation is centered on predicted self-generated "
        "contact rather than movement or external contact alone.",
    ),
    Forbidden(
        id="PERCEPTION-C65b",
        applies_to=SensoryAttenuation,
        when=lambda attenuation: attenuation.scientific_status
        is not ScientificStatus.EMPIRICAL_FINDING,
        message="sensory attenuation is not marked as an empirical finding",
        rationale="Attenuation around predicted self-generated contact is directly "
        "reported by the reviewed experiments.",
    ),
]


__all__ = ["PERCEPTION_CONSTRAINTS"]
