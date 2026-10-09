"""
A kind named like one of the world mimics, as two world models can each declare a kind
of the same name.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol


@dataclass(eq=False)
class VesselAnnotation(Symbol):
    """
    A vessel of another world model, unrelated to the mimic of the same name.
    """
