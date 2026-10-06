"""
Stable ontology namespaces for the Flanagan microworld modules.
"""

from enum import StrEnum

FLANAGAN_NAMESPACE = "http://www.ease-crc.org/ont/llm_reasoner/flanagan/"
"""
Project-owned namespace root for the Flanagan microworld.
"""


class FlanaganNamespace(StrEnum):
    """
    The independently versioned vocabulary namespaces.
    """

    CORE = FLANAGAN_NAMESPACE + "core#"
    """
    Action-neutral phase and predictive-control vocabulary.
    """

    OBJECT_DYNAMICS = FLANAGAN_NAMESPACE + "object-dynamics#"
    """
    Object and limb dynamics vocabulary.
    """

    PREHENSION = FLANAGAN_NAMESPACE + "prehension#"
    """
    Prehensile manipulation vocabulary.
    """

    POURING = FLANAGAN_NAMESPACE + "pouring#"
    """
    Substance-transfer vocabulary.
    """

    GAZE = FLANAGAN_NAMESPACE + "gaze#"
    """
    Gaze-control vocabulary.
    """
    OBSERVATION = FLANAGAN_NAMESPACE + "observation#"
    """
    Action-observation vocabulary.
    """
    LEARNING = FLANAGAN_NAMESPACE + "learning#"
    """
    Visuomotor-learning vocabulary.
    """

    PERCEPTION = FLANAGAN_NAMESPACE + "perception#"
    """
    Perceptual attenuation vocabulary.
    """

    CANONICAL = FLANAGAN_NAMESPACE + "canonical#"
    """
    Identifiers for canonical reference structures.
    """


__all__ = ["FLANAGAN_NAMESPACE", "FlanaganNamespace"]
