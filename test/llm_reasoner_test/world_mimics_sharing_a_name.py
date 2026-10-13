"""
A kind and a function named like ones of the world mimics, as two world models can each
declare a kind or a function of the same name.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol, symbolic_function


@dataclass(eq=False)
class VesselAnnotation(Symbol):
    """
    A vessel of another world model, unrelated to the mimic of the same name.
    """


@symbolic_function
def weighs_less_than(mass: float, other_mass: float) -> bool:
    """
    Whether a mass is less than another, in a world model unrelated to the mimic of the
    same name.

    :param mass: The mass that may be less.
    :param other_mass: The mass it is compared with.
    :return: Whether the mass is less.
    """
    return mass < other_mass
