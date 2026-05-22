"""ReferentResolver — two-tier entity grounding.

Tier 1 (annotation-based): resolve semantic_type to a Symbol subclass via the
        SymbolGraph class diagram, then collect its annotated bodies.
Tier 2 (name-based): substring-match description.name across all symbol_type instances.

All SymbolGraph access is delegated to bridge.world so this module stays clean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from minimal_llmr.bridge.world import (
    get_instances,
    resolve_symbol_class,
    symbol_display_name,
)
from minimal_llmr.core.schemas import ReferentDescription

logger = logging.getLogger(__name__)


@dataclass
class ReferentCandidates:
    """Result of a referent grounding attempt."""

    candidates: list[Any] = field(default_factory=list)
    """Symbol instances that match the description, in preference order."""

    warning: Optional[str] = None
    """Non-fatal diagnostic (e.g. multiple matches, fallback used)."""


@dataclass
class ReferentResolver:
    """Grounds a ReferentDescription to Symbol instances in the SymbolGraph.

    :param symbol_graph: Optional SymbolGraph; None uses the singleton.
    """

    symbol_graph: Any = None

    def resolve(
        self,
        description: ReferentDescription,
        expected_type: Optional[type] = None,
    ) -> ReferentCandidates:
        """Resolve *description* to Symbol instances (Tier 1, then Tier 2)."""
        if description.semantic_type:
            result = self._annotation_resolve(description, expected_type=expected_type)
            if result.candidates:
                logger.debug(
                    "Tier 1: semantic_type=%r → %d instance(s)",
                    description.semantic_type,
                    len(result.candidates),
                )
                return result
            logger.debug(
                "Tier 1 found nothing for '%s', falling back to name search.",
                description.semantic_type,
            )

        result = self._name_resolve(description)
        if result.candidates:
            return result

        warning = (
            f"No instances found for '{description.name}' "
            f"(semantic_type={description.semantic_type!r}). "
            "Check that the object exists in the world."
        )
        logger.warning(warning)
        return ReferentCandidates(candidates=[], warning=warning)

    # ── Tier 1 ─────────────────────────────────────────────────────────────────

    def _annotation_resolve(
        self,
        description: ReferentDescription,
        expected_type: Optional[type] = None,
    ) -> ReferentCandidates:
        cls = resolve_symbol_class(description.semantic_type, self.symbol_graph)
        if cls is None:
            cls = expected_type
        if cls is None:
            return ReferentCandidates()

        annotations = get_instances(cls, self.symbol_graph)
        if not annotations:
            return ReferentCandidates()

        # If annotation instances are already of expected_type, return them directly
        # (e.g. Manipulator field → Manipulator instance, not its kinematic bodies).
        if expected_type is not None and isinstance(annotations[0], expected_type):
            candidates = list(annotations)
            if description.name:
                name_lower = description.name.lower()
                filtered = [a for a in candidates if name_lower in symbol_display_name(a).lower()]
                if filtered:
                    candidates = filtered
            candidates = self._refine(candidates, description)
            return ReferentCandidates(
                candidates=candidates,
                warning=self._multi_match_warning(candidates, description.name),
            )

        # Expand annotations to their physical bodies.
        candidates: list[Any] = []
        for ann in annotations:
            try:
                for body in ann.bodies:
                    if body not in candidates:
                        candidates.append(body)
            except AttributeError:
                if ann not in candidates:
                    candidates.append(ann)

        if description.name and candidates:
            name_lower = description.name.lower()
            filtered = [b for b in candidates if name_lower in symbol_display_name(b).lower()]
            if filtered:
                candidates = filtered

        candidates = self._refine(candidates, description)
        return ReferentCandidates(
            candidates=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Tier 2 ─────────────────────────────────────────────────────────────────

    def _name_resolve(self, description: ReferentDescription) -> ReferentCandidates:
        if not description.name:
            return ReferentCandidates()

        name_lower = description.name.lower()
        all_instances = get_instances(symbol_graph=self.symbol_graph)
        candidates = [
            b for b in all_instances if name_lower in symbol_display_name(b).lower()
        ]
        if not candidates:
            return ReferentCandidates()

        candidates = self._refine(candidates, description)
        return ReferentCandidates(
            candidates=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Attribute refinement ────────────────────────────────────────────────────

    def _refine(
        self, candidates: list[Any], description: ReferentDescription
    ) -> list[Any]:
        if description.attributes and len(candidates) > 1:
            refined = self._filter_by_attributes(candidates, description.attributes)
            if refined:
                return refined
        return candidates

    @staticmethod
    def _filter_by_attributes(candidates: list[Any], attributes: dict) -> list[Any]:
        filtered: list[Any] = []
        for body in candidates:
            body_str = symbol_display_name(body).lower()
            ann_type_names = " ".join(
                type(a).__name__.lower()
                for a in getattr(body, "_semantic_annotations", [])
            )
            combined = body_str + " " + ann_type_names
            for value in attributes.values():
                if isinstance(value, str) and value.lower() in combined:
                    filtered.append(body)
                    break
        return filtered if filtered else candidates

    @staticmethod
    def _multi_match_warning(candidates: list[Any], name: Optional[str]) -> Optional[str]:
        if len(candidates) > 1:
            names = [symbol_display_name(b) for b in candidates]
            return (
                f"Grounding for '{name}' returned {len(candidates)} candidates: "
                f"{names}. First candidate will be used."
            )
        return None
