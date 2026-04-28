"""Resolution layer — turn LLM output into concrete Python values.

- :mod:`llmr.resolution.grounder` maps an :class:`EntityDescription`
  to a KRROOD :class:`Symbol` instance.
- :mod:`llmr.resolution.resolver` dispatches a :class:`SlotValue`
  to a Python value by field kind (ENTITY/POSE/ENUM/PRIMITIVE/TYPE_REF),
  delegating entity-like kinds to the grounder.
"""

from llmr.resolution.grounder import EntityGrounding, GroundingResult
from llmr.resolution.resolver import (
    coerce_enum,
    coerce_primitive,
    resolve_slot,
    ground_entity_slot,
)

__all__ = [
    "EntityGrounding",
    "GroundingResult",
    "coerce_enum",
    "coerce_primitive",
    "resolve_slot",
    "ground_entity_slot",
]
