"""
A module of a world model that nothing imports by default: the functions it declares are
found only once a source is told to load it.

It imports only krrood.
"""

from __future__ import annotations

from krrood.entity_query_language.predicate import symbolic_function


@symbolic_function
def is_below_limit(value: float, limit: float) -> bool:
    """
    Whether a value is below a limit.

    :param value: The value asked about.
    :param limit: The limit it is compared with.
    :return: Whether the value is below the limit.
    """
    return value < limit
