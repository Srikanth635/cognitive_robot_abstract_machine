"""
Generating only when the store cannot answer.

WHY THIS IS A WRAPPER AND NOT A FLAG ON THE GENERATOR
------------------------------------------------------
``HypothesisGenerator`` always calls a model; that is its whole job, and a
``use_cache`` flag would make it sometimes not do it, which is a different job
wearing the same name. Keeping the cache outside means the generator stays
testable with a scripted client and no store, and the caching policy can change
without touching generation.

WHAT A HIT IS WORTH
-------------------
A generation costs a large prompt and one or more model calls. A hypothesis is
generic, so the same one serves every episode of that action until the ontology
moves -- which makes the hit rate close to one in steady use, and the cost close
to zero. That is the return on having kept world state out of generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge.generation.backend import (
    GenerationResult,
    HypothesisGenerator,
)
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.hypothesis.store import HypothesisStore, Lookup
from llm_reasoner.knowledge.microworlds.registry import (
    ModuleSelection,
    select_modules,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference


@dataclass(frozen=True)
class CachedResult:
    """
    A hypothesis, and how it was obtained.
    """

    record: HypothesisRecord
    selection: ModuleSelection
    from_cache: bool
    lookup: Lookup
    """
    The full lookup, so a caller can see what was passed over: a stale record the
    ontology outran, or a refuted one that failed the same way before.
    """

    generation: Optional[GenerationResult] = None

    @property
    def servable(self) -> bool:
        return self.record.servable


@dataclass
class CachingGenerator:
    """
    Answers from the store when it can, generates when it cannot.
    """

    generator: HypothesisGenerator
    store: HypothesisStore

    regenerate_refuted: bool = False
    """
    Whether to try again for an action whose previous attempts were refuted.

    Off by default. A hypothesis refuted because it kept violating the same constraint
    will usually be refuted again by the same model under the same prompt, and spending
    the calls to rediscover that is waste. Turning it on is the right move after the
    ontology or the prompt changes -- which is exactly when the cache key has moved
    anyway.
    """

    def get(
        self,
        task: ConceptReference,
        action_label: Optional[str] = None,
        selection: Optional[ModuleSelection] = None,
        force: bool = False,
    ) -> CachedResult:
        """
        Obtain a hypothesis for an action.

        :param force: Generate even on a hit. For deliberately producing a second
            opinion, not for routine use.
        """
        selection = selection or select_modules(task)
        versions = dict(selection.module_versions)
        lookup = self.store.lookup(
            task.iri,
            versions,
            prompt_version=self.generator.prompt_version,
        )

        if lookup.hit and not force:
            return CachedResult(
                record=lookup.record,
                selection=selection,
                from_cache=True,
                lookup=lookup,
            )

        if lookup.refuted and not self.regenerate_refuted and not force:
            # nothing servable, and previous attempts failed under a key that
            # has not moved: returning the refuted record says more than
            # spending calls to fail the same way again
            return CachedResult(
                record=lookup.refuted[-1],
                selection=selection,
                from_cache=True,
                lookup=lookup,
            )

        result = self.generator.generate(
            task, action_label=action_label, selection=selection
        )
        self.store.put(result.record)
        return CachedResult(
            record=result.record,
            selection=selection,
            from_cache=False,
            lookup=lookup,
            generation=result,
        )


__all__ = ["CachedResult", "CachingGenerator"]
