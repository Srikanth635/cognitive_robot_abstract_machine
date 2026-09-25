"""Extracting what a hypothesis presupposes rather than establishes.

WHY THIS LIVES IN GENERATION AND NOT IN ``hypothesis``
-------------------------------------------------------
``HypothesisRecord`` is deliberately microworld-agnostic, so it cannot know that
a Flanagan condition requirement carries the flag that makes it an assumption.
Generation can, because it is generating for a particular microworld.

WHY IT IS NOT HARD-CODED TO FLANAGAN EITHER
--------------------------------------------
Extraction is driven by the ``marks_assumption`` metadata on a field rather than
by recognising a field name. The next microworld will call its equivalent
something else, and a name-matching extractor would silently return nothing for
it -- silently, because an empty assumption list is indistinguishable from a
hypothesis that happens to presuppose nothing.

WHY ASSUMPTIONS MATTER MORE THAN THE REST
------------------------------------------
Everything else in a hypothesis is structure the resolver may take or leave. An
assumption is a claim about the world that must hold before the hypothesis
applies at all, so it is the one thing the resolver has to check rather than
weigh -- and has to report when it cannot.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from llm_reasoner.knowledge.hypothesis.record import Assumption
from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    OntologyRegistry,
    property_meta,
)


def assumption_fields(cls: type) -> list[str]:
    """:return: Field names on a class declared to mark an assumption."""
    return [
        name for name, meta in property_meta(cls).items() if meta.marks_assumption
    ]


def _describe(fragment: Any) -> str:
    """Render an assumption in the microworld's own terms.

    Prefers the label of whatever the fragment is about over the fragment's own,
    because a requirement is usually unlabelled while the condition it points at
    carries the sentence a reader needs.
    """
    for attribute in ("on_condition", "condition", "about"):
        target = getattr(fragment, attribute, None)
        label = getattr(target, "label", None)
        if label:
            return str(label)
    return str(getattr(fragment, "label", "") or type(fragment).__name__)


def _roles_of(fragment: Any) -> tuple[str, ...]:
    """Identities of the role slots an assumption predicates over.

    Carried so the resolver knows what it must have bound before the assumption
    can even be evaluated: an assumption about a receptacle means nothing until
    a receptacle has been chosen.
    """
    found: list[str] = []
    for attribute in ("on_condition", "condition", "about"):
        target = getattr(fragment, attribute, None)
        if target is None:
            continue
        for role_attribute in ("condition_on_role", "parameter_on_role", "on_role"):
            for role in getattr(target, role_attribute, ()) or ():
                identity = getattr(role, "local_id", None)
                if identity and identity not in found:
                    found.append(identity)
    return tuple(found)


def extract_assumptions(
    fragments: Iterable[Any], registry: OntologyRegistry = REGISTRY
) -> tuple[Assumption, ...]:
    """Collect every assumption a hypothesis's fragments carry.

    :param fragments: Every fragment of the hypothesis, not just the root.
    :return: One assumption per flagged fragment, in encounter order.
    """
    out: list[Assumption] = []
    for fragment in fragments:
        if registry.meta_for(type(fragment)) is None:
            continue
        for name in assumption_fields(type(fragment)):
            if not getattr(fragment, name, False):
                continue
            out.append(
                Assumption(
                    fragment_id=getattr(fragment, "local_id", ""),
                    description=_describe(fragment),
                    on_roles=_roles_of(fragment),
                )
            )
    return tuple(out)


__all__ = ["assumption_fields", "extract_assumptions"]