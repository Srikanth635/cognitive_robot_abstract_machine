"""
Values the vocabulary can hold: numeric ranges, time windows and named categories.

A range keeps only its unit, and the physical dimension is read from the unit. A range
can therefore never claim kilograms and a length at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    TIME_UNITS,
    MeasurementUnit,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, reference
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% numeric values


@CORE.concept(
    alignments=(subclass_of(DUL, "Region"),),
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class MagnitudeRange(KnowledgeEntity):
    """
    An interval of a quantity, with the unit that gives its bounds meaning.
    """

    lower_bound: Optional[float] = attribute(functional=True, minimum_count=1)
    """
    The inclusive lower bound.
    """

    upper_bound: Optional[float] = attribute(functional=True, minimum_count=1)
    """
    The inclusive upper bound.
    """

    unit: Optional[MeasurementUnit] = attribute(functional=True, minimum_count=1)
    """
    The measurement unit shared by both bounds.
    """

    @property
    def physical_dimension(self) -> Optional[PhysicalDimension]:
        """
        The physical dimension the unit measures, or ``None`` while there is no unit.
        """
        return None if self.unit is None else self.unit.physical_dimension


@CORE.concept(
    alignments=(subclass_of(DUL, "Region"),),
    allowed_values={"unit": TIME_UNITS},
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class TemporalWindow(KnowledgeEntity):
    """
    An expected or observed time interval relative to a control event.
    """

    earliest_offset: Optional[float] = attribute(functional=True, minimum_count=1)
    """
    The inclusive beginning of the interval.
    """

    latest_offset: Optional[float] = attribute(functional=True, minimum_count=1)
    """
    The inclusive end of the interval.
    """

    unit: Optional[MeasurementUnit] = attribute(functional=True, minimum_count=1)
    """
    The time unit shared by both offsets.
    """


# %% categorical values


@CORE.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class CategoricalValue(KnowledgeEntity):
    """
    A value that names a category instead of giving a number.
    """

    category: Optional[ConceptReference] = reference(functional=True, minimum_count=1)
    """
    The term identifying the category.
    """
