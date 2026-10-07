"""
A generic base whose metaclass holds an ``__annotations__`` dictionary of its own, as
``abc.ABCMeta`` does once a library has read its annotations.

Looking up
``__annotations__`` on a subclass that declares none then finds an inherited dictionary
instead of creating one for the subclass.
"""

from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

T = TypeVar("T")
"""
The type a subclass binds.
"""


class MetaclassHoldingAnnotations(ABCMeta):
    """
    A metaclass with an annotations dictionary in its own namespace.
    """

    __annotations__ = {}


@dataclass
class GenericUnderAnnotationHoldingMetaclass(
    Generic[T], SubClassSafeGeneric, metaclass=MetaclassHoldingAnnotations
):
    """
    A generic dataclass created by the metaclass holding annotations.
    """

    value: T
    """
    A value of the bound type.
    """
