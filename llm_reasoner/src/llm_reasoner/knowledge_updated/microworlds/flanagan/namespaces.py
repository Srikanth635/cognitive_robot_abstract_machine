"""
The IRI prefixes under which each Flanagan module declares its terms.
"""

from __future__ import annotations

from enum import StrEnum

FLANAGAN_NAMESPACE_ROOT = "http://www.ease-crc.org/ont/llm_reasoner/flanagan/"
"""
The project-owned root shared by every Flanagan module.
"""


class FlanaganNamespace(StrEnum):
    """
    The namespace of each independently versioned Flanagan module.
    """

    CORE = FLANAGAN_NAMESPACE_ROOT + "core#"
    """
    Action-neutral phase and predictive-control vocabulary.
    """

    OBJECT_DYNAMICS = FLANAGAN_NAMESPACE_ROOT + "object-dynamics#"
    """
    Object and limb dynamics vocabulary.
    """

    PREHENSION = FLANAGAN_NAMESPACE_ROOT + "prehension#"
    """
    Prehensile manipulation vocabulary.
    """

    POURING = FLANAGAN_NAMESPACE_ROOT + "pouring#"
    """
    Substance-transfer vocabulary.
    """

    GAZE = FLANAGAN_NAMESPACE_ROOT + "gaze#"
    """
    Gaze-control vocabulary.
    """

    OBSERVATION = FLANAGAN_NAMESPACE_ROOT + "observation#"
    """
    Action-observation vocabulary.
    """

    LEARNING = FLANAGAN_NAMESPACE_ROOT + "learning#"
    """
    Visuomotor-learning vocabulary.
    """

    PERCEPTION = FLANAGAN_NAMESPACE_ROOT + "perception#"
    """
    Perceptual attenuation vocabulary.
    """
