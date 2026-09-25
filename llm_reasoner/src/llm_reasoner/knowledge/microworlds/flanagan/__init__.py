"""
Flanagan–Johansson action-phase domain model.
"""

from llm_reasoner.knowledge.microworlds.flanagan import (
    core,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.microworlds.flanagan.constraints import (
    CONSTRAINTS,
    SUITE,
)

MODULE_SCOPE: str = "llm_reasoner.knowledge.microworlds.flanagan"
"""
Fully qualified scope of the domain model.
"""

__all__ = [
    "CONSTRAINTS",
    "MODULE_SCOPE",
    "SUITE",
    "core",
    "object_dynamics",
    "pouring",
    "prehension",
]
