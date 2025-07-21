import os.path
from abc import ABC
from dataclasses import dataclass

from ripple_down_rules import (Predicate as RDRPredicate, RDRDecorator)
from typing_extensions import ClassVar

from semantic_world import View


@dataclass
class Predicate(RDRPredicate, ABC):
    """
    Inherits from the Base class for predicates in the Ripple Down Rules framework.
    This class extends the RDRPredicate to include additional functionality
    specific to the semantic world predicates.
    """
    models_dir: ClassVar[str] = os.path.join(os.path.dirname(__file__), "predicates_models")


@dataclass
class IsPossibleLocationFor(Predicate):
    """
    Predicate to produce possible locations for an object.
    """
    rdr_decorator: RDRDecorator = RDRDecorator(RDRPredicate.models_dir, (View,), False,
                                               package_name="semantic_world")

    @classmethod
    @Predicate.rdr_decorator((View,), True)
    def evaluate(cls, *args, **kwargs):
        pass
