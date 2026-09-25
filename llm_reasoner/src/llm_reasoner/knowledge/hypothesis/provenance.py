from __future__ import annotations
"""Where a hypothesis came from, and what identifies it for caching.

WHY THE WHOLE ATTEMPT LOG IS KEPT
---------------------------------
Not just the exchange that finally worked. A hypothesis that needed two
repairs carries evidence of *why* -- which constraint the model kept violating,
and whether the ontology's own vocabulary was clear enough to correct it. That
is diagnostic information about the microworld, not merely about the model, and
it is unrecoverable once discarded.
"""

"""Where a hypothesis came from, and what identifies it for caching.

WHY THE WHOLE ATTEMPT LOG IS KEPT
---------------------------------
Not just the exchange that finally worked. A hypothesis that needed two
repairs carries evidence of *why* -- which constraint the model kept violating,
and whether the ontology's own vocabulary was clear enough to correct it. That
is diagnostic information about the microworld, not merely about the model, and
it is unrecoverable once discarded.
"""



from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional

from llm_reasoner.clients.base import Exchange


@dataclass(frozen=True)
class Provenance:
    """Everything needed to reproduce a hypothesis, or to know it is stale."""

    generator: str
    """The client identity that produced it.

    Worth making more specific than a class name: two different models behind
    one client class produce hypotheses indistinguishable in the record, so
    version bumps never invalidate them. Include the model string.
    """

    prompt_version: str
    """Which generation prompt produced it. A prompt change is a generator
    change, and hypotheses generated under the old one are stale for the same
    reason a module version bump makes them stale."""

    module_versions: Mapping[str, str] = field(default_factory=dict)
    """Version of every microworld module loaded, keyed by module name. The
    part of the cache key most likely to change, because the ontology is the
    thing still under active design."""

    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    degraded: bool = False
    """Whether only a fallback capability profile matched the action, so no
    action-specific vocabulary was loaded.

    Carried on provenance rather than left in the selection because it is a fact
    about how this hypothesis was produced, and it survives the selection object.
    A degraded hypothesis is valid but weaker evidence -- it used action-neutral
    types and free-text labels where a named class would have been available for
    a known action -- and the query surface should be able to see that rather
    than serving it as though it were equally well grounded."""

    exchanges: tuple[Exchange, ...] = ()
    """Every call to the model, in order, repairs included.

    ``Exchange`` comes from ``clients`` rather than from either half of the
    reasoner: it records a client call, which is a concept neither half owns.
    Importing it from there is what lets this package keep provenance without
    depending on ``resolver``."""

    @property
    def repair_count(self) -> int:
        """:return: How many corrections were needed. Zero means first try."""
        return max(0, len(self.exchanges) - 1)


@dataclass(frozen=True)
class CacheKey:
    """What makes two hypotheses interchangeable.

    Deliberately carries NO episodic component. A hypothesis is generic by
    construction, so the situation it was first generated for has no bearing on
    whether it can be reused -- and including it would defeat caching entirely,
    since no two episodes are identical.
    """

    action_type: str
    """IRI of the task the hypothesis decomposes."""

    module_versions: tuple[tuple[str, str], ...]
    """Sorted module/version pairs, so the key is order-independent."""

    generator: str
    prompt_version: str

    @classmethod
    def of(cls, action_type: str, provenance: Provenance) -> "CacheKey":
        """Derive the key from a hypothesis's own provenance."""
        return cls(
            action_type=action_type,
            module_versions=tuple(sorted(provenance.module_versions.items())),
            generator=provenance.generator,
            prompt_version=provenance.prompt_version,
        )

    def invalidated_by(self, module_versions: Mapping[str, str]) -> bool:
        """Whether a change in loaded module versions makes this key stale.

        Used to mark hypotheses superseded when the ontology moves, rather than
        silently serving knowledge generated against a model that no longer
        holds.

        :param module_versions: The versions now in force.
        """
        return any(
            module_versions.get(name) != version
            for name, version in self.module_versions
        )


__all__ = ["CacheKey", "Exchange", "Provenance"]