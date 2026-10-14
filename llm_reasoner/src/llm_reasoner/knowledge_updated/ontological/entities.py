"""
The base of everything a hypothesis is made of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from krrood.symbol_graph.symbol_graph import Symbol

LOCAL_ID_LENGTH = 12
"""
Number of characters in a generated identifier.
"""


def generate_local_id() -> str:
    """
    Return a fresh identifier for an entity.
    """
    return uuid4().hex[:LOCAL_ID_LENGTH]


@dataclass(eq=False)
class KnowledgeEntity(Symbol):
    """
    A described thing with a stable identifier and a human-readable label.

    Being a symbol places its instances in the symbol graph, which is what lets the
    entity query language range over them. Identity is by object, so two entities with
    equal content remain different entities.
    """

    local_id: str = field(default_factory=generate_local_id)
    """
    Identifier that survives serialisation, unique within one hypothesis.

    A fragment is identified by its hypothesis together with this identifier.
    """

    label: str = ""
    """
    The name an explanation uses to refer to the entity.
    """
