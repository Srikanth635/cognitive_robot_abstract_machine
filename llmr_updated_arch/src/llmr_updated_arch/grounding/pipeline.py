"""Grounding stage orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Any

from llmr_updated_arch.core.contracts import GroundingResolver
from llmr_updated_arch.core.context import ResolutionContext
from llmr_updated_arch.integrations.krrood.match_reader import bind_slot_value
from llmr_updated_arch.grounding.resolvers import SlotGroundingResolver


@dataclass
class GroundingPipeline:
    """Resolve all free match slots with ordered grounding resolvers."""

    resolvers: list[GroundingResolver] = field(
        default_factory=lambda: [SlotGroundingResolver()]
    )

    def ground(self, context: ResolutionContext, unresolved: Any) -> dict[str, Any]:
        grounded: dict[str, Any] = {}
        for slot in context.match_snapshot.free_slots:
            result = None
            for resolver in self.resolvers:
                if resolver.supports(slot, context.semantic_bundle):
                    result = resolver.resolve(
                        slot=slot,
                        context=context,
                        resolved_params=grounded,
                        unresolved=unresolved,
                    )
                    break
            if result is None or not result.resolved or result.value is unresolved:
                continue
            grounded[slot.attribute_name] = result.value
            bind_slot_value(slot, result.value)
        return grounded
