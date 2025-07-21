from __future__ import annotations
import os
from abc import ABC
from dataclasses import dataclass, field

from ripple_down_rules import (has, isA, dependsOn, TrackedObjectMixin,
                               Predicate as RDRPredicate, RDRDecorator)
from typing_extensions import Tuple, Type, ClassVar, TYPE_CHECKING, List

if TYPE_CHECKING:
    from semantic_world import View


@dataclass
class Predicate(RDRPredicate, ABC):
    """
    Inherits from the Base class for predicates in the Ripple Down Rules framework.
    This class extends the RDRPredicate to include additional functionality
    specific to the semantic world predicates.
    """
    models_dir: ClassVar[str] = os.path.join(os.path.dirname(__file__), "predicate_models")
    rdr_decorator: ClassVar[RDRDecorator] = field(init=False, default=None)

    @classmethod
    def get_rdr_decorator(cls, output_types: Tuple[Type, ...], mutually_exclusive: bool) -> RDRDecorator:
        """
        Returns the RDRDecorator for this predicate.
        """
        return RDRDecorator(cls.models_dir, output_types, mutually_exclusive, update_existing_rules=False,
                            package_name="semantic_world")

    @classmethod
    def fit_mode(cls, fit: bool = True):
        """
        Put the predicate in fit mode.
        """
        cls.rdr_decorator.fit = fit

    @classmethod
    def update_existing_rules(cls, update_existing_rules: bool = True):
        """
        Update the existing rules with the new rules.
        """
        cls.rdr_decorator.update_existing_rules = update_existing_rules