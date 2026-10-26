"""
The restrictions of their classes that fragments under test violate.
"""

from __future__ import annotations

from llm_reasoner.knowledge.acceptance.derived import RestrictionsHold
from llm_reasoner.knowledge.ontologies.terms import (
    DescriptionEntity,
    PropertyRestriction,
    PropertyTerm,
)

# %% violated restrictions


def violated_on(
    fragment: DescriptionEntity, property_term: PropertyTerm
) -> list[PropertyRestriction]:
    """
    :return: The restrictions on a property that a fragment violates.
    """
    return [
        violation.expression
        for violation in RestrictionsHold().violations([fragment])
        if isinstance(violation.expression, PropertyRestriction)
        and violation.expression.property_iri == property_term.iri
    ]
