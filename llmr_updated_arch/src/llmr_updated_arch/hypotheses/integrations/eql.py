"""Minimal KRROOD EQL bridge for sg_model repositories.

This module is intentionally small. It does not mirror the full KRROOD EQL
factory surface; it only adds the pieces that are sg_model-specific:

- binding a repository-backed default domain for ``variable()``
- binding a repository-backed default domain for ``match_variable()``

All other EQL query builders should be imported directly from
``krrood.entity_query_language.factories``.
"""

from __future__ import annotations

from typing_extensions import TypeVar

from krrood.entity_query_language.core.variable import DomainType
from krrood.entity_query_language.factories import (
    match_variable as _match_variable,
    variable as _variable,
)

from llmr_updated_arch.hypotheses.graph import HypothesisGraph

T = TypeVar("T")


def _resolved_domain(
    type_: type[T],
    graph: HypothesisGraph,
    domain: DomainType | None,
) -> DomainType:
    """Resolve the effective EQL domain for an sg_model type."""

    if domain is not None:
        return domain
    return graph.domain(type_)


def variable(
    type_: type[T],
    graph: HypothesisGraph,
    *,
    domain: DomainType | None = None,
):
    """Create an EQL variable whose default domain comes from *graph*.

    Unlike KRROOD ``variable(type_, domain=None)``, sg_model entities are not
    ``Symbol`` subclasses, so there is no implicit global domain. This helper
    always resolves the domain from the provided repository unless an explicit
    *domain* override is supplied.
    """

    return _variable(type_, domain=_resolved_domain(type_, graph, domain))


def match_variable(
    type_: type[T],
    graph: HypothesisGraph,
    *,
    domain: DomainType | None = None,
):
    """Create an EQL match variable whose default domain comes from *graph*."""

    return _match_variable(type_, domain=_resolved_domain(type_, graph, domain))
