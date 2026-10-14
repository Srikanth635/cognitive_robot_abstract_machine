"""
The Flanagan microworld's ontology and the modules it is divided into.

Each module declares into the one ontology under its own namespace, and may specialise
only the modules it depends on. The dependencies below are the ones the scientific
contract sets out: the core names no particular action, and the optional capabilities
never reach into the prehension and pouring chain.
"""

from __future__ import annotations

from llm_reasoner.knowledge_updated.microworlds.flanagan.namespaces import (
    FlanaganNamespace,
)
from llm_reasoner.knowledge_updated.ontological.ontology import Ontology

DEVELOPMENT_VERSION = "0.1.0"
"""
The version of a module whose vocabulary is still being redesigned.
"""

FLANAGAN = Ontology(name="flanagan")
"""
The ontology every Flanagan module declares into.
"""

CORE = FLANAGAN.module(
    "core", version=DEVELOPMENT_VERSION, namespace=FlanaganNamespace.CORE
)
"""
Action-neutral phases, roles, events, conditions, quantities, prediction and control.
"""

OBJECT_DYNAMICS = FLANAGAN.module(
    "object_dynamics",
    version=DEVELOPMENT_VERSION,
    namespace=FlanaganNamespace.OBJECT_DYNAMICS,
    depends_on=(CORE,),
)
"""
Represented object and limb dynamics, efference copies and sensorimotor memory.
"""

PREHENSION = FLANAGAN.module(
    "prehension",
    version=DEVELOPMENT_VERSION,
    namespace=FlanaganNamespace.PREHENSION,
    depends_on=(OBJECT_DYNAMICS,),
)
"""
Contact, lift, transport, replacement, release and grip-load control.
"""

POURING = FLANAGAN.module(
    "pouring",
    version=DEVELOPMENT_VERSION,
    namespace=FlanaganNamespace.POURING,
    depends_on=(PREHENSION,),
)
"""
Source and target roles, flow, material regimes and transfer variables.
"""

GAZE = FLANAGAN.module(
    "gaze",
    version=DEVELOPMENT_VERSION,
    namespace=FlanaganNamespace.GAZE,
    depends_on=(CORE,),
)
"""
Proactive and reactive gaze, targets, fixation and multimodal alignment.
"""

OBSERVATION = FLANAGAN.module(
    "observation",
    version=DEVELOPMENT_VERSION,
    namespace=FlanaganNamespace.OBSERVATION,
    depends_on=(GAZE,),
)
"""
Actor-observer evidence, visibility and prediction matching.
"""

LEARNING = FLANAGAN.module(
    "learning",
    version=DEVELOPMENT_VERSION,
    namespace=FlanaganNamespace.LEARNING,
    depends_on=(GAZE, OBJECT_DYNAMICS),
)
"""
Exploration, acquisition, refinement and sensorimotor-map updates.
"""

PERCEPTION = FLANAGAN.module(
    "perception",
    version=DEVELOPMENT_VERSION,
    namespace=FlanaganNamespace.PERCEPTION,
    depends_on=(CORE,),
)
"""
Predicted self-contact attenuation and its unresolved relation to control.
"""
